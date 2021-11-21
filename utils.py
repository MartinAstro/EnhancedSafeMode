import os
import shutil
import tempfile
import glob

def load_checkpoint(tf_agent, replay_buffer, train_step, tempdir):
    import tensorflow as tf
    from tf_agents.utils import common
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
    train_step = tf.compat.v1.train.get_global_step()
    return tf_agent, replay_buffer, train_step

def copy_files(root_src_dir, root_dst_dir):
    import shutil
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                if os.path.samefile(src_file, dst_file):
                    continue
                os.remove(dst_file)
            shutil.copy2(src_file, dst_dir)

def collect_policy_checkpoints():
    tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())
    directories = os.path.join(tempdir, 'policies/checkpoints/**')
    policies = glob.glob(directories)
    for i in range(len(policies)):
        policies[i] = policies[i].split("_")[-1]
    policies.sort()
    return policies

def load_policy(chpt_number, load_dir=None):
    import tensorflow as tf
    from tf_agents.utils import common
    assert len(chpt_number) == 10
    if load_dir is None:
        tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())
    else:
        tempdir = load_dir
    from_dir = os.path.join(tempdir, 'policies/checkpoints/policy_checkpoint_'+chpt_number+"")
    to_dir = os.path.join(tempdir, 'policies/policy')
    copy_files(from_dir, to_dir)
    policy = tf.saved_model.load(os.path.join(tempdir, 'policies/policy'))
    return policy

def save_policy(save_dir):
    # save to Data/Policies/directory
    tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())
    for directory in ['policies/', 'eval/', 'train/']:
        from_dir = os.path.join(tempdir, directory )
        to_dir = os.path.join('Data/Policies/' + save_dir, directory)
        copy_files(from_dir, to_dir)
        shutil.rmtree(from_dir)
    print("Policy directories saved to: %s" % ('Data/Policies/' + save_dir))
