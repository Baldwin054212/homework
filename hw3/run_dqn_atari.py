import argparse
import gym
from gym import wrappers
import os.path as osp
import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                session,
                args, logdir):

    # This is just a rough estimate
    num_iterations = float(args.num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= args.num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env=env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=True,
        logdir=logdir
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):

    set_global_seeds(seed)
    task.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(task, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='SeaquestNoFrameskip-v4')
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=5e7)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


    logdir = args.exp_name + '_' + args.env_name
    logdir = os.path.join('data', logdir)
    logdir = osp.join(logdir, "seed"+str(args.seed))


    task = gym.make(args.env_name)

    # Run training
    seed = args.seed
    print('random seed = %d' % seed)
    env = get_env(task, seed)

    session = get_session()
    atari_learn(env, session, args, logdir)

if __name__ == "__main__":

    main()
