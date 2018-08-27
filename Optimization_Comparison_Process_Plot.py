import os
import json
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
palette = ['#539caf', colors["darkmagenta"], colors["crimson"], colors["teal"]]
styles = ['-', '--', '-.', ':']

Algorithms = ['RA', 'BA', 'GA', 'CMA']
X = {}
Y = {}

for algorithm in Algorithms:
    X[algorithm] = {}
    Y[algorithm] = {}
    path = 'KEEL_Cross_Folder_XGBoost_Para_From_' + algorithm + '_Timeline_Record'
    Dir = os.listdir(path)

    for file in Dir:
        name = path + '/' + file
        data_set_name = file.split('_')[0]
        print("Optimization Algorithm", algorithm, "; Data Set Name: ", data_set_name)
        X[algorithm][data_set_name] = []
        Y[algorithm][data_set_name] = []

        with open(name, 'r') as para_json:
            para_value = json.load(para_json)
        y_max = 0
        for key, val in para_value.items():
            X[algorithm][data_set_name].append(float(key))
            y_current = float(list(val.keys())[0])
            if y_current > y_max:
                y_max = y_current
            Y[algorithm][data_set_name].append(y_max)

data_set_list = list(X[Algorithms[0]].keys())
fig = plt.figure(figsize=(8,72))
Title = 'Optimization_Process_Compare'
fig.canvas.set_window_title(Title)
fig.subplots_adjust(hspace=1.2)
for i in range(len(data_set_list)):
    ax = plt.subplot(len(data_set_list) + 1, 1, i+1)
    for j in range(len(Algorithms)):
        ax.plot(X[Algorithms[j]][data_set_list[i]], Y[Algorithms[j]][data_set_list[i]],
                color=palette[j], linestyle=styles[j], label=Algorithms[j])
    plt.legend(loc='upper right')
    plt.title(data_set_list[i])

fig_file = Title + '.png'
fig.savefig(fig_file)





