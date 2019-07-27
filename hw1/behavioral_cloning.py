import tensorflow as tf
from tensorflow.python.keras import layers
import gym
import pickle
import numpy as np
import os

print(tf.VERSION)
print(tf.keras.__version__)

ENV_NAME = "Ant-v2"
NUM_ROLLOUTS = 10
MAX_STEPS = None
SEED_NUM = 5
BATCH_SIZE = 64
EPOCHS = 20
ITERATION = 10

if __name__ == '__main__':
    # a = [1, 2, 3]
    # b = [4, 5, 6]
    # d = {"mean": a, "std": b}
    # with open(os.path.join("test_data", "behavior_cloning_" + ENV_NAME+".pkl"), "wb") as f:
    #     pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    #
    env = gym.make(ENV_NAME)
    act_dim = env.action_space.shape[0]

    file = "expert_data/"+ENV_NAME+".pkl"
    with open(file, "rb") as f:
        data = pickle.load(f)

    train = data["observations"]
    label = data["actions"]
    label = np.squeeze(label)
    max_steps = MAX_STEPS or env.spec.timestep_limit

    means = []
    stds = []
    for seed in range(SEED_NUM):

        mean = []
        std = []

        model = tf.keras.Sequential()
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(act_dim, activation="tanh"))

        model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss="mse", metrics=['mae'])

        tf.set_random_seed(seed*10)
        np.random.seed(seed*10)

        for iter in range(ITERATION):
            print("iter:", iter)
            model.fit(train, label, batch_size=BATCH_SIZE, epochs=EPOCHS)

            roll_reward = []
            for roll in range(NUM_ROLLOUTS):
                s = env.reset()
                done = False
                reward = 0
                step = 0
                while not done:
                    a = model.predict(s[np.newaxis, :])
                    s, r, done, _ = env.step(a)
                    reward += r
                    # env.render()
                    step += 1
                    if step >= max_steps:
                        break

                roll_reward.append(reward)

            mean.append(np.mean(roll_reward))
            std.append(np.std(roll_reward))

        means.append(mean)
        stds.append(std)

        # print('returns', epoch_reward)
        print('mean return', mean)
        print('std of return', std)

    d = {"mean": means, "std": stds}
    with open(os.path.join("test_data", "behavior_cloning_" + ENV_NAME+".pkl"), "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
