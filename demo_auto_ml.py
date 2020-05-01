import numpy as np
import pickle
import timeit
import datetime
import argparse
from sklearn.metrics import accuracy_score
from hyperopt import hp, tpe, fmin, Trials
from mab_diff.BanditBO_Diff import BanditBO_Diff
import test_functions.auto_ml

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="iris", type=str, nargs='?', help='dataset')
parser.add_argument('--budget', default=60, type=int, nargs='?', help='number of iterations to run BO')
parser.add_argument('--method', default="banditbo", type=str, nargs='?', help='method to run optimization')

args = parser.parse_args()
dataset = args.dataset
method = args.method
print("dataset: {}, budget: {}, method: {}".format(dataset, args.budget, method))

np.random.seed(123)

f_bo = test_functions.auto_ml.obj_func_bo
f_tpe = test_functions.auto_ml.obj_func_tpe

c_bound_dim = 14
x_dims = np.array([2, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 3])
# original bounds
x_bounds_org = [      [[50, 100], [0.01, 2]],
                      [[0.01, 1], [0.01, 1], [0.1, 1]],
                      [[0, 2]],
                      [[0, 1]],
                      [[10, 50], [0, 1]],
                      [[1e-2, 100]],
                      [[1e-2, 100]],
                      [[0, 1]],
                      [[0, 1]],
                      [[0.03125, 32768]],
                      [[0.03125, 32768], [3.0517578125e-05, 8]],
                      [[1e-5, 10]],
                      [[1e-7, 1e-1], [1e-9, 1], [1e-7, 1e-1]],
                      [[128, 256], [1e-7, 1e-1], [1e-4, 1e-1]]]
# new bounds with log transform
x_bounds = np.array([[[50, 100], [-4.61, 0.69]],
                     [[-4.61, 0], [0.01, 1], [0.1, 1]],
                     [[0, 2]],
                     [[0, 1]],
                     [[10, 50], [0, 1]],
                     [[-4.61, 4.61]],
                     [[-4.61, 4.61]],
                     [[0, 1]],
                     [[0, 1]],
                     [[-3.47, 10.40]],
                     [[-3.47, 10.40], [-10.40, 2.08]],
                     [[-11.51, 2.3]],
                     [[-16.12, -2.3], [-20.72, 0], [-16.12, -2.3]],
                     [[4.85, 5.55], [-16.12, -2.3], [-9.21, -2.3]]])

n_run = 10 # 10
n_arm = c_bound_dim
n_dim = x_dims
budget = args.budget # 50
batch_list = [1] # [1, 5, 10]
n_init = n_dim + 1

# BanditBO
bounds_mab = []
for arm in range(n_arm):
    x_bound = [{"name": "x{}".format(d+1), "type": "continuous", "domain": (x_bounds[arm][d][0], x_bounds[arm][d][1])}
               for d in range(x_dims[arm])]
    bounds_mab.append(x_bound)

# TPE
# use "_classifier" instead of "classifier" to make sure it is the first item in dictionary
bounds_tpe = hp.choice('_classifier', [
    {'model': "Adaboost", 'param': {'ada_n_estimators': hp.uniform('ada_n_estimators', x_bounds[0][0][0], x_bounds[0][0][1]),
                                    'ada_lr_rate': hp.uniform('ada_lr_rate', x_bounds[0][1][0], x_bounds[0][1][1])}},
    {'model': "GradientBoosting", 'param': {'gb_lr_rate': hp.uniform('gb_lr_rate', x_bounds[1][0][0], x_bounds[1][0][1]),
                                            'gb_subsample': hp.uniform('gb_subsample', x_bounds[1][1][0], x_bounds[1][1][1]),
                                            'gb_max_features': hp.uniform('gb_max_features', x_bounds[1][2][0], x_bounds[1][2][1])}},
    {'model': "DecisionTree", 'param': {'dt_max_depth': hp.uniform('dt_max_depth', x_bounds[2][0][0], x_bounds[2][0][1])}},
    {'model': "ExtraTrees", 'param': {'et_max_features': hp.uniform('et_max_features', x_bounds[3][0][0], x_bounds[3][0][1])}},
    {'model': "RandomForest", 'param': {'rf_n_estimators': hp.uniform('rf_n_estimators', x_bounds[4][0][0], x_bounds[4][0][1]),
                                        'rf_max_features': hp.uniform('rf_max_features', x_bounds[4][1][0], x_bounds[4][1][1])}},
    {'model': "BernoulliNB", 'param': {'ber_alpha': hp.uniform('ber_alpha', x_bounds[5][0][0], x_bounds[5][0][1])}},
    {'model': "MultinomialNB", 'param': {'mul_alpha': hp.uniform('mul_alpha', x_bounds[6][0][0], x_bounds[6][0][1])}},
    {'model': "LDA", 'param': {'lda_shrinkage': hp.uniform('lda_shrinkage', x_bounds[7][0][0], x_bounds[7][0][1])}},
    {'model': "QDA", 'param': {'qda_reg_param': hp.uniform('qda_reg_param', x_bounds[8][0][0], x_bounds[8][0][1])}},
    {'model': "LinearSVC", 'param': {'lin_C': hp.uniform('lin_C', x_bounds[9][0][0], x_bounds[9][0][1])}},
    {'model': "SVC", 'param': {'svm_C': hp.uniform('svm_C', x_bounds[10][0][0], x_bounds[10][0][1]),
                               'svm_gamma': hp.uniform('svm_gamma', x_bounds[10][1][0], x_bounds[10][1][1])}},
    {'model': "PassiveAggressive", 'param': {'pa_C': hp.uniform('pa_C', x_bounds[11][0][0], x_bounds[11][0][1])}},
    {'model': "SGD", 'param': {'sgd_alpha': hp.uniform('sgd_alpha', x_bounds[12][0][0], x_bounds[12][0][1]),
                               'sgd_l1_ratio': hp.uniform('sgd_l1_ratio', x_bounds[12][1][0], x_bounds[12][1][1]),
                               'sgd_eta0': hp.uniform('sgd_eta0', x_bounds[12][2][0], x_bounds[12][2][1])}},
    {'model': "NeuralNetwork", 'param': {'nn_units': hp.uniform('nn_units', x_bounds[13][0][0], x_bounds[13][0][1]),
                                         'nn_alpha': hp.uniform('nn_alpha', x_bounds[13][1][0], x_bounds[13][1][1]),
                                         'nn_lr_rate': hp.uniform('nn_lr_rate', x_bounds[13][2][0], x_bounds[13][2][1])}}
])

# save models to files
batch_name = "_".join(map(str, batch_list))

datasets_all = ["iris", "digits", "wine", "breast_cancer", "analcatdata_authorship",
                "blood_transfusion", "monks1", "monks2", "steel_plates_fault", "qsar_biodeg",
                "phoneme", "diabetes", "hill_valley", "eeg_eye_state", "waveform",
                "spambase", "australian", "churn", "vehicle", "balance_scale",
                "kc1", "kc2", "cardiotocography", "wall_robot_navigation", "segment",
                "artificial_characters", "electricity", "gas_drift", "olivetti", "letter"]

# times to run Hyperopt-sklearn, Auto-sklearn, and TPOT
def get_required_time(dataset, n_run):
    if dataset == "gas_drift" or dataset == "olivetti":
        time_all_models = 14000
        time_each_model = 1400
    elif dataset == "letter":
        time_all_models = 13000
        time_each_model = 1300
    elif dataset == "electricity":
        time_all_models = 6300
        time_each_model = 630
    elif dataset == "artificial_characters":
        time_all_models = 1600
        time_each_model = 160
    elif dataset == "digits":
        time_all_models = 1300
        time_each_model = 130
    elif dataset == "eeg_eye_state" or dataset == "wall_robot_navigation":
        time_all_models = 1100
        time_each_model = 110
    elif dataset == "waveform":
        time_all_models = 800
        time_each_model = 80
    else:
        time_all_models = 600
        time_each_model = 60
    time_all_models = int(time_all_models / n_run)
    time_each_model = int(time_each_model / n_run)

    # times in seconds
    return time_all_models, time_each_model

### SMAC/Auto-sklearn ###
if method == "autosklearn":
    import autosklearn.classification
    if dataset == "all":
        for dataset in datasets_all:
            time_all_models, time_each_model = get_required_time(dataset, n_run)
            autosklearn_start_time = timeit.default_timer()
            acc_all = []
            for run in range(n_run):
                X_train, y_train, X_test, y_test = test_functions.auto_ml.gen_train_test_data(dataset, seed=run)
                automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=time_all_models,
                                                                          per_run_time_limit=time_each_model)
                automl.fit(X_train, y_train)
                y_pred = automl.predict(X_test)
                auc = accuracy_score(y_test, y_pred)
                acc_all.append(auc)
                print("run: {}, acc: {}".format(run, round(auc, 4)))
            autosklearn_end_time = timeit.default_timer()
            acc_mean = round(np.mean(acc_all), 4)
            acc_std = round(np.std(acc_all) / np.sqrt(n_run), 4)
            runtime = round(autosklearn_end_time - autosklearn_start_time, 2)
            print("Auto-sklearn acc: {} ({})".format(acc_mean, acc_std))
            print("Auto-sklearn runtime: {}(s)".format(runtime))
            # save result to file
            with open('autosklearn_result_{}.txt'.format(dataset), 'w') as f:
                f.write('dataset: {}, Auto-sklearn acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))
            filename = "autosklearn_{}.file".format(dataset)
            with open(filename, "wb") as f:
                np.save(f, np.array(acc_all))
    else:
        time_all_models, time_each_model = get_required_time(dataset, n_run)
        autosklearn_start_time = timeit.default_timer()
        acc_all = []
        for run in range(n_run):
            X_train, y_train, X_test, y_test = test_functions.auto_ml.gen_train_test_data(dataset, seed=run)
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=time_all_models,
                                                                      per_run_time_limit=time_each_model)
            automl.fit(X_train, y_train)
            y_pred = automl.predict(X_test)
            auc = accuracy_score(y_test, y_pred)
            acc_all.append(auc)
            print("run: {}, acc: {}".format(run, round(auc, 4)))
        autosklearn_end_time = timeit.default_timer()
        acc_mean = round(np.mean(acc_all), 4)
        acc_std = round(np.std(acc_all) / np.sqrt(n_run), 4)
        runtime = round(autosklearn_end_time - autosklearn_start_time, 2)
        print("Auto-sklearn acc: {} ({})".format(acc_mean, acc_std))
        print("Auto-sklearn runtime: {}(s)".format(runtime))
        # save result to file
        with open('autosklearn_result_{}.txt'.format(dataset), 'w') as f:
            f.write('dataset: {}, Auto-sklearn acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))
        filename = "autosklearn_{}.file".format(dataset)
        with open(filename, "wb") as f:
            np.save(f, np.array(acc_all))

### TPE/Hyperopt-sklearn ###
if method == "hpsklearn":
    from hpsklearn import HyperoptEstimator, any_classifier
    if dataset == "all":
        for dataset in datasets_all:
            time_all_models, time_each_model = get_required_time(dataset, n_run)
            hpsklearn_start_time = timeit.default_timer()
            acc_all = []
            for run in range(n_run):
                X_train, y_train, X_test, y_test = test_functions.auto_ml.gen_train_test_data(dataset, seed=run)
                hyperopt = HyperoptEstimator(classifier=any_classifier("clf"), algo=tpe.suggest, max_evals=budget,
                                             preprocessing=[], trial_timeout=time_each_model)
                hyperopt.fit(X_train, y_train)
                y_pred = hyperopt.predict(X_test)
                auc = accuracy_score(y_test, y_pred)
                acc_all.append(auc)
                print("run: {}, acc: {}".format(run, round(auc, 4)))
            hpsklearn_end_time = timeit.default_timer()
            acc_mean = round(np.mean(acc_all), 4)
            acc_std = round(np.std(acc_all) / np.sqrt(n_run), 4)
            runtime = round(hpsklearn_end_time - hpsklearn_start_time, 2)
            print("Hyperopt-sklearn acc: {} ({})".format(acc_mean, acc_std))
            print("Hyperopt-sklearn runtime: {}(s)".format(runtime))
            # save result to file
            with open('hpsklearn_result_{}.txt'.format(dataset), 'w') as f:
                f.write('dataset: {}, Hyperopt-sklearn acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))
            filename = "hpsklearn_{}.file".format(dataset)
            with open(filename, "wb") as f:
                np.save(f, np.array(acc_all))
    else:
        time_all_models, time_each_model = get_required_time(dataset, n_run)
        hpsklearn_start_time = timeit.default_timer()
        acc_all = []
        for run in range(n_run):
            X_train, y_train, X_test, y_test = test_functions.auto_ml.gen_train_test_data(dataset, seed=run)
            hyperopt = HyperoptEstimator(classifier=any_classifier("clf"), algo=tpe.suggest, max_evals=budget,
                                         preprocessing=[], trial_timeout=time_each_model)
            hyperopt.fit(X_train, y_train)
            y_pred = hyperopt.predict(X_test)
            auc = accuracy_score(y_test, y_pred)
            acc_all.append(auc)
            print("run: {}, acc: {}".format(run, round(auc, 4)))
        hpsklearn_end_time = timeit.default_timer()
        acc_mean = round(np.mean(acc_all), 4)
        acc_std = round(np.std(acc_all) / np.sqrt(n_run), 4)
        runtime = round(hpsklearn_end_time - hpsklearn_start_time, 2)
        print("Hyperopt-sklearn acc: {} ({})".format(acc_mean, acc_std))
        print("Hyperopt-sklearn runtime: {}(s)".format(runtime))
        # save result to file
        with open('hpsklearn_result_{}.txt'.format(dataset), 'w') as f:
            f.write('dataset: {}, Hyperopt-sklearn acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))
        filename = "hpsklearn_{}.file".format(dataset)
        with open(filename, "wb") as f:
            np.save(f, np.array(acc_all))

### TPOT ###
if method == "tpot" or method == "all":
    from tpot import TPOTClassifier
    if dataset == "all":
        for dataset in datasets_all:
            time_all_models, time_each_model = get_required_time(dataset, n_run)
            # times returns in seconds but TPOT uses minutes
            time_all_models = int(time_all_models / 60)
            time_each_model = round(time_each_model / 60, 2)
            tpot_start_time = timeit.default_timer()
            acc_all = []
            for run in range(n_run):
                X_train, y_train, X_test, y_test = test_functions.auto_ml.gen_train_test_data(dataset, seed=run)
                tpot = TPOTClassifier(max_time_mins=time_all_models, max_eval_time_mins=time_each_model,
                                      random_state=run, verbosity=1)
                tpot.fit(X_train, y_train)
                y_pred = tpot.predict(X_test)
                auc = accuracy_score(y_test, y_pred)
                acc_all.append(auc)
                print("run: {}, acc: {}".format(run, round(auc, 4)))
            tpot_end_time = timeit.default_timer()
            acc_mean = round(np.mean(acc_all), 4)
            acc_std = round(np.std(acc_all) / np.sqrt(n_run), 4)
            runtime = round(tpot_end_time - tpot_start_time, 2)
            print("TPOT acc: {} ({})".format(acc_mean, acc_std))
            print("TPOT runtime: {}(s)".format(runtime))
            # save result to file
            with open('tpot_result_{}.txt'.format(dataset), 'w') as f:
                f.write('dataset: {}, TPOT acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))
            filename = "tpot_{}.file".format(dataset)
            with open(filename, "wb") as f:
                np.save(f, np.array(acc_all))
    else:
        time_all_models, time_each_model = get_required_time(dataset, n_run)
        # times returns in seconds but TPOT uses minutes
        time_all_models = int(time_all_models / 60)
        time_each_model = round(time_each_model / 60, 2)
        tpot_start_time = timeit.default_timer()
        acc_all = []
        for run in range(9, n_run+2):
            X_train, y_train, X_test, y_test = test_functions.auto_ml.gen_train_test_data(dataset, seed=run)
            tpot = TPOTClassifier(max_time_mins=time_all_models, max_eval_time_mins=time_each_model,
                                  random_state=run, verbosity=1)
            tpot.fit(X_train, y_train)
            y_pred = tpot.predict(X_test)
            auc = accuracy_score(y_test, y_pred)
            acc_all.append(auc)
            print("run: {}, acc: {}".format(run, round(auc, 4)))
        tpot_end_time = timeit.default_timer()
        acc_mean = round(np.mean(acc_all), 4)
        acc_std = round(np.std(acc_all) / np.sqrt(n_run), 4)
        runtime = round(tpot_end_time - tpot_start_time, 2)
        print("TPOT acc: {} ({})".format(acc_mean, acc_std))
        print("TPOT runtime: {}(s)".format(runtime))
        # save result to file
        with open('tpot_result_{}.txt'.format(dataset), 'w') as f:
            f.write('dataset: {}, TPOT acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))
        filename = "tpot_{}.file".format(dataset)
        with open(filename, "wb") as f:
            np.save(f, np.array(acc_all))

# proposed method
### Bandit-BO ###
if method == "banditbo" or method == "all":
    if dataset == "all":
        for dataset in datasets_all:
            banditbo_start_time = timeit.default_timer()
            ts_draw = "set" # func, set
            banditbo = BanditBO_Diff(objfn=f_bo, n_init=n_init, bounds=bounds_mab, acq_type='TS', n_arm=n_arm,
                                     norm_y=True, ts_draw=ts_draw, dataset=dataset, rand_seed=108)
            banditbo.runoptimBatchList(n_run, budget, batch_list)
            with open("banditbo_{}.pickle".format(dataset), "wb") as f:
                pickle.dump(banditbo, f)
            banditbo_end_time = timeit.default_timer()
            print("acc_test: {}".format(np.around(banditbo.acc_tests, 4)))
            acc_mean = round(np.mean(banditbo.acc_tests), 4)
            acc_std = round(np.std(banditbo.acc_tests) / np.sqrt(n_run), 4)
            runtime = round(banditbo_end_time - banditbo_start_time, 2)
            print("Bandit-BO acc: {} ({})".format(acc_mean, acc_std))
            print("Bandit-BO runtime: {}(s)".format(runtime))
            # save result to file
            with open('banditbo_result_{}.txt'.format(dataset), 'w') as f:
                f.write('dataset: {}, Bandit-BO acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))
    else:
        banditbo_start_time = timeit.default_timer()
        ts_draw = "set"  # func, set
        banditbo = BanditBO_Diff(objfn=f_bo, n_init=n_init, bounds=bounds_mab, acq_type='TS', n_arm=n_arm,
                                 norm_y=True, ts_draw=ts_draw, dataset=dataset, rand_seed=108)
        banditbo.runoptimBatchList(n_run, budget, batch_list)
        with open("banditbo_{}.pickle".format(dataset), "wb") as f:
            pickle.dump(banditbo, f)
        banditbo_end_time = timeit.default_timer()
        print("acc_test: {}".format(np.around(banditbo.acc_tests, 4)))
        acc_mean = round(np.mean(banditbo.acc_tests), 4)
        acc_std = round(np.std(banditbo.acc_tests) / np.sqrt(n_run), 4)
        runtime = round(banditbo_end_time - banditbo_start_time, 2)
        print("Bandit-BO acc: {} ({})".format(acc_mean, acc_std))
        print("Bandit-BO runtime: {}(s)".format(runtime))
        # save result to file
        with open('banditbo_result_{}.txt'.format(dataset), 'w') as f:
            f.write('dataset: {}, Bandit-BO acc: {} ({}), runtime: {}'.format(dataset, acc_mean, acc_std, runtime))



