import os
import tensorflow as tf
from tf_agents.utils import common

def load_checkpoint(tf_agent, replay_buffer, train_step, tempdir):
    checkpoint_dir = os.path.join(tempdir, 'train/checkpoints')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step    
        )

    train_checkpointer.initialize_or_restore()
    #train_step = tf.compat.v1.train.get_global_step()
    return tf_agent, replay_buffer, train_step

def load_agent(tempdir):
    # Load policy
    policy_dir = os.path.join(tempdir, 'policies/policy')
    tf_agent = tf.saved_model.load(policy_dir)

    return tf_agent
