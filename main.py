from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tf_agents import replay_buffers

import os
import matplotlib.pyplot as plt
import tempfile
from tf_agents import specs
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment, BatchedPyEnvironment
from tf_agents.networks import q_network
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from utils import load_agent, load_checkpoint
from metrics import log_eval_metrics, get_eval_metrics
from visualization import visualize_returns
from environment import SafeModeEnv

tempdir = tempfile.gettempdir()

def build_agent(env, config, train_step):
    # The critic network first encodes the observation ("state") and actions seperately,
    # then joins them and passes the jointed combinations through a fully connected network
    # to produce stimates of the Q-values
    observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(env))

    use_gpu = False
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None, # No encoding
            action_fc_layer_params=None, # No encoding
            joint_fc_layer_params=config['critic_joint_fc_layer_params'], # (256, 256) - 2 layers 256 units
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform'
        )
        actor_net = actor_distribution_network.ActorDistributionNetwork(
                    observation_spec,
                    action_spec, # action_spec needs .shape.num_elements()
                    fc_layer_params=config['actor_fc_layer_params'],
                    #continuous_projection_net=( # Can't handle bounded array spec
                    #tanh_normal_projection_network.TanhNormalProjectionNetwork)
            )


    tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=config['actor_learning_rate']),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=config['critic_learning_rate']),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=config['alpha_learning_rate']),
            target_update_tau=config['target_update_tau'],
            target_update_period=config['target_update_period'],
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=config['gamma'],
            reward_scale_factor=config['reward_scale_factor'],
            train_step_counter=train_step)

    tf_agent.initialize()
    return tf_agent

def build_collect_actor(collect_env, tf_agent, train_step, replay_observer, tempdir):
    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
                                collect_env,
                                collect_policy,
                                train_step,
                                steps_per_run=1,
                                metrics=actor.collect_metrics(10),
                                summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
                                observers=[replay_observer, env_step_metric]
                                )
    return collect_actor

def build_initial_collect_actor(collect_env, train_step, replay_observer, initial_collect_steps):
    random_policy = random_py_policy.RandomPyPolicy(
                        collect_env.time_step_spec(), collect_env.action_spec())

    initial_collect_actor = actor.Actor(
                            collect_env,
                            random_policy,
                            train_step,
                            steps_per_run=initial_collect_steps,
                            observers=[replay_observer])
    return initial_collect_actor

def build_eval_actor(eval_env, tf_agent, train_step, num_eval_episodes, tempdir):
    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)


    eval_actor = actor.Actor(
                        eval_env,
                        eval_policy,
                        train_step,
                        episodes_per_run=num_eval_episodes,
                        metrics=actor.eval_metrics(num_eval_episodes),
                        summary_dir=os.path.join(tempdir, 'eval'),
                        )
    return eval_actor

def build_agent_learner(tf_agent, train_step, policy_save_interval, replay_buffer, batch_size):
    dataset = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda : dataset
    
    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
                                tempdir,
                                train_step,
                                tf_agent,
                                experience_dataset_fn,
                                checkpoint_interval=policy_save_interval,
                                triggers=learning_triggers
                                )
    return agent_learner

def train_agent(tf_agent, eval_actor, collect_actor, agent_learner, configs):
    num_iterations = configs['num_iterations']
    eval_interval = configs['eval_interval']
    log_interval = configs['log_interval']

    # Metrics: Actor is currently going to provide average return and average length episode
    metrics = get_eval_metrics(eval_actor)
    log_eval_metrics(0, metrics)

    # Training the Agent: Includes collecting data from environment and optimizing agent networks

    tf_agent.train_step_counter.assign(0)   # Reset the train step
    avg_return = get_eval_metrics(eval_actor)["AverageReturn"]
    returns = [avg_return]

    for _ in range(num_iterations):
        # Training.
        collect_actor.run() # Add to the replay buffer
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics(eval_actor)
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))
    
    visualize_returns(num_iterations, eval_interval, returns)

def main():
    continue_training = False
    training_configs = {
        'num_iterations' : 5000, # Number of iterations for training networks (epochs)
        'eval_interval' : 1000, # How often the average return is calculated during training
        'log_interval' : 100, # How often to print the loss
    }

    policy_save_interval = 1000 # How often to save the policy in training steps
    num_eval_episodes = 20 # Number of times agent is run to evaluate policy

    initial_collect_steps = 256*10 # How many times random policy will be used to fill buffer
    replay_buffer_capacity = 10000
    batch_size = 256 # Batch size used for generating training datasets for networks


    # Network Params
    network_config = {
        "critic_learning_rate" : 3e-4,
        "actor_learning_rate" : 3e-4,
        "alpha_learning_rate" : 3e-4,
        "target_update_tau" : 0.005,
        "target_update_period" : 1,
        "gamma" : 0.99,
        "reward_scale_factor" : 1.0,

        "actor_fc_layer_params" : (20, 20, 20, 20),
        "critic_joint_fc_layer_params" : (20, 20, 20, 20),
    }
    
    collect_env = BatchedPyEnvironment(envs=[SafeModeEnv()])
    eval_env = BatchedPyEnvironment(envs=[SafeModeEnv()])

    train_step = train_utils.create_train_step() # TF variable equal to 0 
    tf_agent = build_agent(collect_env, network_config, train_step)

    # Custom Replay Buffer implementation
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=collect_env.batch_size,
        max_length=replay_buffer_capacity)

    # Add an observer that adds to the replay buffer:
    replay_observer = replay_buffer.add_batch

    ## Actors: Interact between policy and environment. 
    # Actors pass trajectories (S,A,R) to observer. Observer caches and writes traj to buffer
    
    # Policy which randomly selects random actions to seed buffer
    initial_collect_actor = build_initial_collect_actor(collect_env, train_step, replay_observer, initial_collect_steps)
    initial_collect_actor.run()

    # Copy of policy for data collection
    collect_actor = build_collect_actor(collect_env, tf_agent, train_step, replay_observer, tempdir)

    # Main policy for evaluation and deployment
    eval_actor = build_eval_actor(eval_env, tf_agent, train_step, num_eval_episodes, tempdir)

    # Learners: Contains agent and performs gradient step updates to policy variables
    agent_learner = build_agent_learner(tf_agent, train_step, policy_save_interval, replay_buffer, batch_size)

    if continue_training:
        tf_agent, replay_buffer, train_step = load_checkpoint(tf_agent, replay_buffer, train_step, tempdir)
   
    train_agent(tf_agent, eval_actor, collect_actor, agent_learner, training_configs)
    

if __name__ == "__main__":
    main()