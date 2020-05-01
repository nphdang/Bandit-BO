import numpy as np
import timeit
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
import openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK

datasets_all = ["iris", "digits", "wine", "breast_cancer", "analcatdata_authorship",
                "blood_transfusion", "monks1", "monks2", "steel_plates_fault", "qsar_biodeg",
                "phoneme", "diabetes", "hill_valley", "eeg_eye_state", "waveform",
                "spambase", "australian", "churn", "vehicle", "balance_scale",
                "kc1", "kc2", "cardiotocography", "wall_robot_navigation", "segment",
                "artificial_characters", "electricity", "gas_drift", "olivetti", "letter"]

# cross-validation on training data
cv_train = 5

def gen_train_test_data(dataset="", seed=42):
    print("dataset: {}, seed: {}".format(dataset, seed))
    if dataset == "iris":
        ds = datasets.load_iris()
        X = ds.data
        y = ds.target
    if dataset == "digits":
        ds = datasets.load_digits()
        X = ds.data
        y = ds.target
    if dataset == "wine":
        ds = datasets.load_wine()
        X = ds.data
        y = ds.target
    if dataset == "breast_cancer":
        ds = datasets.load_breast_cancer()
        X = ds.data
        y = ds.target
    if dataset == "olivetti":
        ds = fetch_olivetti_faces()
        # get data point size
        img_rows, img_cols = ds.images.shape[1:]
        dim = img_rows * img_cols
        X = ds.data
        X = X.reshape(-1, dim)
        y = ds.target
    if dataset == "analcatdata_authorship":
        dataset_id = 458
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "Austen"] = "0"
        y_new[y_new == "London"] = "1"
        y_new[y_new == "Milton"] = "2"
        y_new[y_new == "Shakespeare"] = "3"
        y = np.array([int(val) for val in y_new])
    if dataset == "blood_transfusion":
        dataset_id = 1464
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "monks1":
        dataset_id = 333
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "monks2":
        dataset_id = 334
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "steel_plates_fault":
        dataset_id = 1504
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "qsar_biodeg":
        dataset_id = 1494
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "phoneme":
        dataset_id = 1489
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "diabetes":
        dataset_id = 37
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "tested_negative"] = "0"
        y_new[y_new == "tested_positive"] = "1"
        y = np.array([int(val) for val in y_new])
    if dataset == "hill_valley":
        dataset_id = 1479
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "eeg_eye_state":
        dataset_id = 1471
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "kc1":
        dataset_id = 1067
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([str(val) for val in y.values])
        y_new[y_new == "False"] = "0"
        y_new[y_new == "True"] = "1"
        y = np.array([int(val) for val in y_new])
    if dataset == "kc2":
        dataset_id = 1063
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "no"] = "0"
        y_new[y_new == "yes"] = "1"
        y = np.array([int(val) for val in y_new])
    if dataset == "spambase":
        dataset_id = 44
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "electricity":
        dataset_id = 151
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "UP"] = "0"
        y_new[y_new == "DOWN"] = "1"
        y = np.array([int(val) for val in y_new])
    if dataset == "australian":
        dataset_id = 40981
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "letter":
        dataset_id = 6
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        # convert letters to numbers
        y = np.array([ord(val) for val in y.values])
    if dataset == "churn":
        dataset_id = 40701
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "vehicle":
        dataset_id = 54
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "opel"] = "0"
        y_new[y_new == "saab"] = "1"
        y_new[y_new == "bus"] = "2"
        y_new[y_new == "van"] = "3"
        y = np.array([int(val) for val in y_new])
    if dataset == "balance_scale":
        dataset_id = 11
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "L"] = "0"
        y_new[y_new == "B"] = "1"
        y_new[y_new == "R"] = "2"
        y = np.array([int(val) for val in y_new])
    if dataset == "artificial_characters":
        dataset_id = 1459
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "cardiotocography":
        dataset_id = 1466
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "wall_robot_navigation":
        dataset_id = 1497
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "waveform":
        dataset_id = 60
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "gas_drift":
        dataset_id = 1476
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "connect4":
        dataset_id = 40668
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "segment":
        dataset_id = 40984
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "brickface"] = "0"
        y_new[y_new == "sky"] = "1"
        y_new[y_new == "foliage"] = "2"
        y_new[y_new == "cement"] = "3"
        y_new[y_new == "window"] = "4"
        y_new[y_new == "path"] = "5"
        y_new[y_new == "grass"] = "6"
        y = np.array([int(val) for val in y_new])

    # normalize X
    X = MinMaxScaler().fit_transform(X)
    # convert y to integer array
    y = np.array([int(val) for val in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    return X_train, y_train, X_test, y_test

def Adaboost_classify(params, dataset, seed, classify):
    model_name = "Adaboost"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # n_estimators = UniformIntegerHyperparameter(name="n_estimators", lower=50, upper=500, default_value=50, log=False)
    # learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    model = AdaBoostClassifier(n_estimators=int(params["n_estimators"]), learning_rate=np.exp(params["learning_rate"]),
                               random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def GradientBoosting_classify(params, dataset, seed, classify):
    model_name = "GradientBoosting"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
    # subsample = UniformFloatHyperparameter(name="subsample", lower=0.01, upper=1.0, default_value=1.0)
    # max_features = UniformFloatHyperparameter("max_features", 0.1, 1.0 , default_value=1)
    model = GradientBoostingClassifier(learning_rate=np.exp(params["learning_rate"]),
                                       subsample=round(params["subsample"], 4),
                                       max_features=round(params["max_features"], 4), random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def DecisionTree_classify(params, dataset, seed, classify):
    model_name = "DecisionTree"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # max_depth = UniformFloatHyperparameter('max_depth', 0., 2., default_value=0.5)
    model = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def ExtraTrees_classify(params, dataset, seed, classify):
    model_name = "ExtraTrees"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # max_features = UniformFloatHyperparameter("max_features", 0., 1., default_value=0.5)
    max_features = round(float(params["max_features"]), 4)
    max_features = int(train_X.shape[1] ** max_features)
    model = ExtraTreesClassifier(max_features=max_features, random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def RandomForest_classify(params, dataset, seed, classify):
    model_name = "RandomForest"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # n_estimators = UniformIntegerHyperparameter(name="n_estimators", lower=10, upper=50, default_value=50)
    # max_features = UniformFloatHyperparameter("max_features", 0., 1., default_value=0.5)
    max_features = round(float(params["max_features"]), 4)
    max_features = int(train_X.shape[1] ** max_features)
    model = RandomForestClassifier(n_estimators=int(params["n_estimators"]), max_features=max_features, random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def BernoulliNB_classify(params, dataset, seed, classify):
    model_name = "BernoulliNB"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100, default_value=1, log=True)
    model = BernoulliNB(alpha=np.exp(params["alpha"]))
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def MultinomialNB_classify(params, dataset, seed, classify):
    model_name = "MultinomialNB"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100, default_value=1, log=True)
    model = MultinomialNB(alpha=np.exp(params["alpha"]))
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def LDA_classify(params, dataset, seed, classify):
    model_name = "LDA"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # shrinkage = UniformFloatHyperparameter("shrinkage", 0., 1., 0.5)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=round(params["shrinkage"], 4))
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def QDA_classify(params, dataset, seed, classify):
    model_name = "QDA"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # reg_param = UniformFloatHyperparameter('reg_param', 0.0, 1.0, default_value=0.0)
    model = QuadraticDiscriminantAnalysis(reg_param=round(params["reg_param"], 4))
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def LinearSVC_classify(params, dataset, seed, classify):
    model_name = "LinearSVC"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
    model = LinearSVC(C=np.exp(params["C"]), random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def SVC_classify(params, dataset, seed, classify):
    model_name = "SVC"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
    # gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    model = SVC(C=np.exp(params["C"]), gamma=np.exp(params["gamma"]), random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def PassiveAggressive_classify(params, dataset, seed, classify):
    model_name = "PassiveAggressive"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # C = UniformFloatHyperparameter("C", 1e-5, 10, 1.0, log=True)
    model = PassiveAggressiveClassifier(C=np.exp(params["C"]), max_iter=1000, tol=1e-3, random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def SGD_classify(params, dataset, seed, classify):
    model_name = "SGD"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # alpha = UniformFloatHyperparameter("alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
    # l1_ratio = UniformFloatHyperparameter("l1_ratio", 1e-9, 1, log=True, default_value=0.15)
    # eta0 = UniformFloatHyperparameter("eta0", 1e-7, 1e-1, default_value=0.01, log=True)
    model = SGDClassifier(loss="log", alpha=np.exp(params["alpha"]), l1_ratio=round(np.exp(params["l1_ratio"]), 4),
                          eta0=np.exp(params["eta0"]), max_iter=1000, tol=1e-3, random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

def NN_classify(params, dataset, seed, classify):
    model_name = "NeuralNetwork"
    print(model_name, params, dataset, seed)
    np.random.seed(108)
    start_time = timeit.default_timer()
    train_X, train_y, test_X, test_y = gen_train_test_data(dataset, seed)
    # build a classifier based on selected parameters
    # hidden_layer_sizes = UniformIntegerHyperparameter("hidden_layer_sizes", 128, 256, log=True, default_value=128)
    # alpha = UniformFloatHyperparameter("alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
    # learning_rate_init = UniformFloatHyperparameter("learning_rate_init", 1e-4, 1e-1, default_value=0.001, log=True)
    model = MLPClassifier(hidden_layer_sizes=int(np.exp(params["hidden_layer_sizes"])), alpha=np.exp(params["alpha"]),
                          learning_rate_init=np.exp(params["learning_rate_init"]), max_iter=1000, random_state=108)
    if classify == "test":
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # maximize accuracy
        auc = accuracy_score(test_y, pred_y)
    if classify == "cv":
        scores = cross_val_score(model, train_X, train_y, cv=cv_train)
        auc = np.mean(scores)
    # minimize loss
    loss = 1.0 - auc
    end_time = timeit.default_timer()
    print("{}_runtime: {}(s)".format(model_name, round(end_time - start_time, 2)))
    del model

    # dictionary with information for evaluation
    return {'auc': auc, 'loss': loss, 'status': STATUS_OK}

# generate arms from categorical variables
map_arm_model = {0: "Adaboost", 1: "GradientBoosting", 2: "DecisionTree", 3: "ExtraTrees", 4: "RandomForest",
                 5: "BernoulliNB", 6: "MultinomialNB", 7: "LDA", 8: "QDA", 9: "LinearSVC", 10: "SVC",
                 11: "PassiveAggressive", 12: "SGD", 13: "NeuralNetwork"}
print("# of categories: {}".format(len(map_arm_model)))

# count no of function evaluation
cnt_func_eval = 0

# objective function for TPE method
dataset_tpe = ""
seed_tpe = ""
def obj_func_tpe(params, classify="cv"):
    global cnt_func_eval
    cnt_func_eval = cnt_func_eval + 1
    model = params["model"]
    if model not in map_arm_model.values():
        arm = params["model"]  # model in the form of arm index
        model = map_arm_model[arm]
    if model == "Adaboost":
        params = {'n_estimators': params['param']['ada_n_estimators'], 'learning_rate': params['param']['ada_lr_rate']}
        res = Adaboost_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "GradientBoosting":
        params = {'learning_rate': params['param']['gb_lr_rate'], 'subsample': params['param']['gb_subsample'],
                  'max_features': params['param']['gb_max_features']}
        res = GradientBoosting_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "DecisionTree":
        params = {'max_depth': params['param']['dt_max_depth']}
        res = DecisionTree_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "ExtraTrees":
        params = {'max_features': params['param']['et_max_features']}
        res = ExtraTrees_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "RandomForest":
        params = {'n_estimators': params['param']['rf_n_estimators'], 'max_features': params['param']['rf_max_features']}
        res = RandomForest_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "BernoulliNB":
        params = {'alpha': params['param']['ber_alpha']}
        res = BernoulliNB_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "MultinomialNB":
        params = {'alpha': params['param']['mul_alpha']}
        res = MultinomialNB_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "LDA":
        params = {'shrinkage': params['param']['lda_shrinkage']}
        res = LDA_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "QDA":
        params = {'reg_param': params['param']['qda_reg_param']}
        res = QDA_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "LinearSVC":
        params = {'C': params['param']['lin_C']}
        res = LinearSVC_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "SVC":
        params = {'C': params['param']['svm_C'], 'gamma': params['param']['svm_gamma']}
        res = SVC_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "PassiveAggressive":
        params = {'C': params['param']['pa_C']}
        res = PassiveAggressive_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "SGD":
        params = {'alpha': params['param']['sgd_alpha'], 'l1_ratio': params['param']['sgd_l1_ratio'],
                  'eta0': params['param']['sgd_eta0']}
        res = SGD_classify(params, dataset_tpe, seed_tpe, classify)
    if model == "NeuralNetwork":
        params = {'hidden_layer_sizes': params['param']['nn_units'], 'alpha': params['param']['nn_alpha'],
                  'learning_rate_init': params['param']['nn_lr_rate']}
        res = NN_classify(params, dataset_tpe, seed_tpe, classify)

    print("iteration: {}, result: {}".format(cnt_func_eval, res))
    return res

# objective function for Bandit-BO method
def obj_func_bo(arm, X, n_dim, dataset, seed, classify="cv"):
    # n_dim: number of dimensions except the first categorical variable
    X = np.array(X).reshape(-1, n_dim)  # format X to [[]]
    res = []
    for x_val in X:
        model = map_arm_model[arm]
        if model == "Adaboost":
            params = {'n_estimators': x_val[0], 'learning_rate': x_val[1]}
            y_val = Adaboost_classify(params, dataset, seed, classify)["auc"]
        if model == "GradientBoosting":
            params = {'learning_rate': x_val[0], 'subsample': x_val[1], 'max_features': x_val[2]}
            y_val = GradientBoosting_classify(params, dataset, seed, classify)["auc"]
        if model == "DecisionTree":
            params = {'max_depth': x_val[0]}
            y_val = DecisionTree_classify(params, dataset, seed, classify)["auc"]
        if model == "ExtraTrees":
            params = {'max_features': x_val[0]}
            y_val = ExtraTrees_classify(params, dataset, seed, classify)["auc"]
        if model == "RandomForest":
            params = {'n_estimators': x_val[0], 'max_features': x_val[1]}
            y_val = RandomForest_classify(params, dataset, seed, classify)["auc"]
        if model == "BernoulliNB":
            params = {'alpha': x_val[0]}
            y_val = BernoulliNB_classify(params, dataset, seed, classify)["auc"]
        if model == "MultinomialNB":
            params = {'alpha': x_val[0]}
            y_val = MultinomialNB_classify(params, dataset, seed, classify)["auc"]
        if model == "LDA":
            params = {'shrinkage': x_val[0]}
            y_val = LDA_classify(params, dataset, seed, classify)["auc"]
        if model == "QDA":
            params = {'reg_param': x_val[0]}
            y_val = QDA_classify(params, dataset, seed, classify)["auc"]
        if model == "LinearSVC":
            params = {'C': x_val[0]}
            y_val = LinearSVC_classify(params, dataset, seed, classify)["auc"]
        if model == "SVC":
            params = {'C': x_val[0], 'gamma': x_val[1]}
            y_val = SVC_classify(params, dataset, seed, classify)["auc"]
        if model == "PassiveAggressive":
            params = {'C': x_val[0]}
            y_val = PassiveAggressive_classify(params, dataset, seed, classify)["auc"]
        if model == "SGD":
            params = {'alpha': x_val[0], 'l1_ratio': x_val[1], 'eta0': x_val[2]}
            y_val = SGD_classify(params, dataset, seed, classify)["auc"]
        if model == "NeuralNetwork":
            params = {'hidden_layer_sizes': x_val[0], 'alpha': x_val[1], 'learning_rate_init': x_val[2]}
            y_val = NN_classify(params, dataset, seed, classify)["auc"]
        res.append([y_val])
    res = np.array(res)

    return res



