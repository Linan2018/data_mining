# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("-k", help="For cross-validation")
parser.add_argument("-t", "--threshold_step", help="Threshold step")
parser.add_argument("--min", help="Use a small amount of data", action="store_true")
args = parser.parse_args()

warnings.filterwarnings("ignore")

np.random.seed(0)


# colab_path = "drive/My Drive/hw1/"
colab_path = ""

if args.min:
    file_path = colab_path + "ml-1m/ratings.dat"

    with open(file_path) as f:
        data = f.read().strip().split()

    with open("data_small.dat", "w") as f:
        for line in data:
            s = line.strip().split('::')
            if int(s[0]) <= 300 and int(s[1]) <= 200:
                f.writelines(line)
                f.writelines('\n')
    file_path = colab_path + "data_small.dat"
    n_user, n_item = 300, 200

else:
    file_path = colab_path + "ml-1m/ratings.dat"
    n_user, n_item = 6040, 3952




with open(file_path) as f:
    data = f.read().strip().split()
random.shuffle(data)

k = int(args.k)
threshold_step = int(args.threshold_step)
n_step = len(data)//k
result = {'tp': [0], 'fp': [0], 'fn': [0], 'tn': [0]}
index = 0

fig_data = {
    'x': [],
    'y': []
}

print("Done 0")

for rating_threshold in np.arange(0, 5 + threshold_step, threshold_step):
    for episode in range(k):
        print("************ rating_threshold: {}, episode: {} ************".format(rating_threshold, episode))
        val_data = data[episode * n_step: (episode+1) * n_step]
        train_data = data[:episode * n_step] + data[(episode+1) * n_step:]
        r_ndarray = 3 * np.ones((n_user, n_item))
        r_v_ndarray = np.zeros((n_user, n_item))

        for line in train_data:
            s = line.strip().split('::')
            r_ndarray[int(s[0])-1][int(s[1])-1] = int(s[2])
            r_v_ndarray[int(s[0]) - 1][int(s[1]) - 1] = int(s[2])

        exist = (r_ndarray > rating_threshold) * 1.0
        n_item2user = [np.sum(exist[:, _]) for _ in range(n_item)]

        print("Done 1")

        s_item_ndarray = np.ones((n_item, n_item))

        for i in range(n_item):
            for j in range(i + 1, n_item):
                n_i = n_item2user[i]
                n_j = n_item2user[j]
                n_ij = np.sum(np.dot(exist[:, i], exist[:, j]))
                sq = np.sqrt(n_i*n_j)
                if sq:
                    s_item_ndarray[i][j] = n_ij / sq
                else:
                    s_item_ndarray[i][j] = 0
                if np.isnan(s_item_ndarray[i][j]):
                    s_item_ndarray[i][j] = 0
                s_item_ndarray[j][i] = s_item_ndarray[i][j]

        print("Done 2")

        for line in val_data:
            s = line.strip().split('::')
            user, item, rating = int(s[0]) - 1, int(s[1]) - 1, int(s[2])
            real = rating > 3

            temp = (r_v_ndarray[user, :] > 0) * 1.0

            t = np.multiply(temp, s_item_ndarray[:, item])

            if np.sum(t) < 1e-5:
                p = 3
            else:
                p = np.dot(t/np.sum(t), r_ndarray[user, :])

            prediction = p > rating_threshold

            if real == prediction:
                if real:
                    result['tp'][index] += 1
                else:
                    result['tn'][index] += 1
            else:
                if real:
                    result['fn'][index] += 1
                else:
                    result['fp'][index] += 1
    # print("tp:{}, fp:{}, fn:{}, tn:{}".format(
    #     result['tp'][index], result['fp'][index], result['fn'][index], result['tn'][index]))

    try:
        fpr = result['fp'][index] / (result['fp'][index] + result['tn'][index])
    except ZeroDivisionError:
        if result['fp'][index] == 0:
            fpr = 1
        else:
            fpr = (result['fp'][index] + 0.1) / (result['fp'][index] + result['tn'][index] + 0.2)
    try:
        tpr = result['tp'][index] / (result['tp'][index] + result['fn'][index])
    except ZeroDivisionError:
        if result['tp'][index] == 0:
            tpr = 1
        else:
            tpr = (result['tp'][index] + 0.1) / (result['tp'][index] + result['fn'][index] + 0.2)

    fig_data['x'].append(fpr)
    fig_data['y'].append(tpr)

    print("FPR: {}, TPR :{}".format(fpr, tpr))

    index += 1
    result['tp'].append(0)
    result['tn'].append(0)
    result['fn'].append(0)
    result['fp'].append(0)


result['tp'].pop()
result['tn'].pop()
result['fn'].pop()
result['fp'].pop()
pd.DataFrame(result).to_csv(colab_path + "result.csv")
print("csv saved at{}".format(colab_path + "result.csv"))


#
# for i in range(index):
#
#     # print(fpr, tpr)


plt.figure(figsize=(6, 6))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])

plt.plot(fig_data['x'], fig_data['y'], color='darkorange')
plt.savefig(colab_path + 'result.png')
print("figure saved at{}".format(colab_path + 'result.png'))
plt.show()