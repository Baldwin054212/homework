import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

ENV_NAME = "Ant-v2"
ITERATION = 10


def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

    return smooth_data


if __name__ == '__main__':
    file = "behavior_cloning_" + ENV_NAME+".pkl"
    with open(os.path.join("test_data", file), "rb") as f:
        data = pickle.load(f)

    x1 = data["mean"]
    # x1 = smooth(x1, sm=2)

    file = "dagger_" + ENV_NAME+".pkl"
    with open(os.path.join("test_data", file), "rb") as f:
        data = pickle.load(f)

    x2 = data["mean"]
    x2 = smooth(x2, sm=2)

    time = range(10)

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(time=time, data=x1, color="r", condition="behavior_cloning")
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")

    plt.ylabel("Reward")
    plt.xlabel("Iteration Number")
    plt.title("Imitation Learning")

    plt.show()