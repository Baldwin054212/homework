import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
SEED_NUM = 10
ENV = "halfcheetah"

if __name__ == '__main__':
    file = os.path.join("ppo/ppo_s0", "progress.txt")
    algo = ["ddpg_" + ENV, "td3_" + ENV, "ppo_" + ENV, "trpo_" + ENV, "vpg_" + ENV, "sac_" + ENV]
    data = []
    for i in range(len(algo)):
        for seed in range(SEED_NUM):
            file = os.path.join(os.path.join(algo[i], algo[i] + "_s" + str(seed*10)), "progress.txt")

            pd_data = pd.read_table(file)
            pd_data.insert(len(pd_data.columns), "Unit", seed)
            pd_data.insert(len(pd_data.columns), "Condition", algo[i])

            data.append(pd_data)

    data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="TotalEnvInteracts", value="AverageEpRet", condition="Condition", unit="Unit")

    xscale = np.max(data["TotalEnvInteracts"]) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.legend(loc='best').set_draggable(True)
    plt.tight_layout(pad=0.5)
    plt.show()








