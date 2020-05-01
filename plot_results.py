import pickle
import numpy as np
import matplotlib.pyplot as plt
# set big font
import seaborn as sns
sns.set_context("notebook", font_scale=1.8)
plt.style.use('fivethirtyeight')

test_case = "1d_C6"
n_arm = 6
budget = 40
batch_list = [1]

methods = ["BanditBO", "OnehotEncoding", "MerchanLobato", "SMAC", "TPE"]
batch_name = "_".join(map(str, batch_list))

if "BanditBO" in methods:
    with open("{}_banditbo_b{}.pickle".format(test_case, batch_name), "rb") as f:
        banditbo = pickle.load(f)
# baselines
if "OnehotEncoding" in methods:
    with open("{}_onehot_b{}.pickle".format(test_case, batch_name), "rb") as f:
        onehot = pickle.load(f)
if "MerchanLobato" in methods:
    with open("{}_merchanlobato_b{}.pickle".format(test_case, batch_name), "rb") as f:
        merchanlobato = pickle.load(f)
if "SMAC" in methods:
    with open("{}_smac_b{}.pickle".format(test_case, batch_name), "rb") as f:
        smac = pickle.load(f)
if "TPE" in methods:
    with open("{}_tpe_b{}.pickle".format(test_case, batch_name), "rb") as f:
        tpe = pickle.load(f)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ["o", "s", "v", "^", "*", "d", "h", "p", "x", "+"]
marker_size = 4
lw = 3

# plot best function values
x_range = np.arange(budget)
for idx, batch in enumerate(batch_list):
    fig, ax = plt.subplots(figsize=(7, 5))
    if "BanditBO" in methods:
        ax.plot(x_range, banditbo.mean_bestVals_batch[:budget, idx], linewidth=lw, label="Bandit-BO", color=colors[0])
        ax.fill_between(x_range, banditbo.mean_bestVals_batch[:budget, idx] - banditbo.mean_errVals_batch[:budget, idx],
                        banditbo.mean_bestVals_batch[:budget, idx] + banditbo.mean_errVals_batch[:budget, idx],
                        color=colors[0], alpha=0.2)
    if "OnehotEncoding" in methods:
        ax.plot(x_range, onehot.mean_bestVals_batch[:budget, idx], linewidth=lw, label="One-hot-Encoding", color=colors[2])
        ax.fill_between(x_range, onehot.mean_bestVals_batch[:budget, idx] - onehot.mean_errVals_batch[:budget, idx],
                        onehot.mean_bestVals_batch[:budget, idx] + onehot.mean_errVals_batch[:budget, idx],
                        color=colors[2], alpha=0.2)
    if "MerchanLobato" in methods:
        ax.plot(x_range, merchanlobato.mean_bestVals_batch[:budget, idx], linewidth=lw, label="Merchan-Lobato", color=colors[3])
        ax.fill_between(x_range, merchanlobato.mean_bestVals_batch[:budget, idx] - merchanlobato.mean_errVals_batch[:budget, idx],
                        merchanlobato.mean_bestVals_batch[:budget, idx] + merchanlobato.mean_errVals_batch[:budget, idx],
                        color=colors[3], alpha=0.2)
    if "SMAC" in methods:
        ax.plot(x_range, smac.mean_bestVals_batch[:budget, idx], linewidth=lw, label="SMAC", color=colors[4])
        ax.fill_between(x_range, smac.mean_bestVals_batch[:budget, idx] - smac.mean_errVals_batch[:budget, idx],
                        smac.mean_bestVals_batch[:budget, idx] + smac.mean_errVals_batch[:budget, idx],
                        color=colors[4], alpha=0.2)
    if "TPE" in methods:
        ax.plot(x_range, tpe.mean_bestVals_batch[:budget, idx], linewidth=lw, label="TPE", color=colors[5])
        ax.fill_between(x_range, tpe.mean_bestVals_batch[:budget, idx] - tpe.mean_errVals_batch[:budget, idx],
                        tpe.mean_bestVals_batch[:budget, idx] + tpe.mean_errVals_batch[:budget, idx],
                        color=colors[5], alpha=0.2)
    ax.set_xlabel("Iterations (Function evaluations)")
    ax.set_ylabel("Best function value")
    if test_case in ["cnn_tune_mnist", "cnn_tune_cifar10"]:
        ax.set_ylabel("Accuracy")
    if test_case in ["nas_optimization_protein_structure", "nas_optimization_slice_localization",
                     "nas_optimization_naval_propulsion", "nas_optimization_parkinsons_telemonitoring"]:
        ax.set_ylabel("Negative MSE")
    plt.legend()
    if batch_name == "1_5_10":
        ax.set_title("Batch Size = {}".format(batch))
        filename = test_case + "_compare_batch_" + str(batch) + ".pdf"
        plt.savefig(filename, bbox_inches="tight")
        if batch == 5:
            ax.set_title("# of Categories = {}".format(n_arm))
            filename = test_case + "_compare_batch_" + str(batch) + "_new.pdf"
            plt.savefig(filename, bbox_inches="tight")
    elif batch_name == "5":
        ax.set_title("Batch Size = {}".format(batch))
        filename = test_case + "_compare_batch_" + str(batch) + ".pdf"
        plt.savefig(filename, bbox_inches="tight")
        ax.set_title("# of Categories = {}".format(n_arm))
        filename = test_case + "_compare_batch_" + str(batch) + "_new.pdf"
        plt.savefig(filename, bbox_inches="tight")
    else:
        ax.set_title("Batch Size = {}".format(batch))
        filename = test_case + "_compare_batch_" + str(batch) + ".pdf"
        plt.savefig(filename, bbox_inches="tight")



