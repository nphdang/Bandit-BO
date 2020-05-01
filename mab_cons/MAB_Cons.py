import numpy as np
import collections
import pickle
import sys
import multiprocessing

class MAB_Cons:
    def __init__(self, objfn, n_init, bounds, acq_type, n_arm, save_result, rand_seed=108):
        self.f = objfn  # function to optimize
        self.bounds = bounds  # function bounds
        self.n_arm = n_arm  # no of arms
        self.acq_type = acq_type  # acquisition function
        self.n_init = n_init  # no of initial points
        self.n_dim = len(self.bounds)  # dimension
        self.rand_seed = rand_seed
        self.n_core = multiprocessing.cpu_count()
        self.nAcq_opt = 100 # 100 no of initializations to optimize acq function
        self.save_result = save_result

        # store the arm recommendations in each batch_size
        self.arm_recommendations = []
        # store the name of the method
        self.method = None
        # store the parameters of the method
        self.params = None

        # set the random number
        np.random.seed(self.rand_seed)

        # store the best input in each iteration in all trials for 1 batch_size
        self.b_best_inps = []
        # store the best func value in each iteration in all trials for 1 batch_size
        self.b_best_vals = []
        # store the frequency of each arm for all trials for 1 batch_size
        self.b_hist_arms = []

    # initialize points for all methods
    def initialize_all_methods(self):
        # initialize Bandit-BO
        if self.method == "BanditBO":
            Xinit_mab = []
            for c in range(self.n_arm):
                Xinit_arm = np.zeros((self.n_init, self.n_dim))
                for i in range(self.n_init):
                    x = np.array([np.random.uniform(self.bounds[d]["domain"][0], self.bounds[d]['domain'][1], 1)[0]
                                  for d in range(0, self.n_dim)])
                    Xinit_arm[i, :] = x
                Xinit_mab.append(Xinit_arm)
            Yinit_mab = []
            for c in range(self.n_arm):
                x = Xinit_mab[c]
                y = self.f(c, x)
                Yinit_mab.append(y)

            return Xinit_mab, Yinit_mab
        # initialize One-hot-Encoding and SMAC
        if self.method == "OnehotEncoding" or self.method == "SMAC":
            n_init_point = self.n_arm * self.n_init
            Xinit_gpy = np.zeros((n_init_point, self.n_dim))
            for c in range(self.n_arm):
                x1 = c
                for i in range(self.n_init):
                    x2 = np.array([np.random.uniform(self.bounds[d]["domain"][0], self.bounds[d]['domain'][1], 1)[0]
                                   for d in range(1, self.n_dim)])
                    x = np.append(x1, x2)
                    Xinit_gpy[i + c * self.n_init, :] = x
            Yinit_gpy = []
            for i in range(n_init_point):
                x = Xinit_gpy[i, :]
                # x[0] is an arm
                y = self.f(int(x[0]), x[1:])
                Yinit_gpy.append(y)

            return Xinit_gpy, Yinit_gpy
        # initialize Merchan-Lobato
        if self.method == "MerchanLobato":
            n_init_point = self.n_arm * self.n_init
            Xinit_ker = np.zeros((n_init_point, self.n_dim))
            for c in range(self.n_arm):
                # map to one-hot encoding
                x1 = np.zeros(self.n_arm)
                x1[c] = 1
                for i in range(self.n_init):
                    x2 = np.array([np.random.uniform(self.bounds[d]["domain"][0], self.bounds[d]['domain'][1], 1)[0]
                                   for d in range(self.n_arm, self.n_dim)])
                    x = np.append(x1, x2)
                    Xinit_ker[i + c * self.n_init, :] = x
            Yinit_ker = []
            for i in range(n_init_point):
                x = Xinit_ker[i, :]
                # map one-hot encoding of categorical variable to arm index
                arm = np.argmax(x[:self.n_arm])
                y = self.f(arm, x[self.n_arm:])
                Yinit_ker.append(y)

            return Xinit_ker, Yinit_ker
        # initialize TPE
        if self.method == "TPE":
            n_init_point = self.n_arm * self.n_init
            # points in BO format
            Xinit_tpe = np.zeros((n_init_point, self.n_dim))
            bounds_keys = list(self.bounds.keys())
            for c in range(self.n_arm):
                x1 = c
                for i in range(self.n_init):
                    x2 = []
                    for d in range(1, self.n_dim):
                        para_str = self.bounds[bounds_keys[d]].__str__()  # convert parameter to string
                        para_list = para_str.split("\n")
                        para_lowerbound = para_list[4]  # get its lower bound
                        para_lowerbound = float(para_lowerbound[(para_lowerbound.index("{") + 1):-1])
                        para_upperbound = para_list[5]  # get its upper bound
                        para_upperbound = float(para_upperbound[(para_upperbound.index("{") + 1):-1])
                        x2.append(np.random.uniform(para_lowerbound, para_upperbound, 1)[0])
                    x = np.append(x1, x2)
                    Xinit_tpe[i + c * self.n_init, :] = x
            Yinit_tpe = []  # store max function values
            Finit_tpe = []  # store min function values
            # convert initial points from BO format to TPE format
            # points in TPE format
            Pinit_tpe = []
            for x in Xinit_tpe:
                point = {bounds_keys[idx]: val for idx, val in enumerate(x)}
                point[bounds_keys[0]] = int(point[bounds_keys[0]])
                Pinit_tpe.append(point)
            for point in Pinit_tpe:
                y = self.f(point)["auc"]
                f = self.f(point)["loss"]
                Yinit_tpe.append(y)
                Finit_tpe.append(f)

            return Xinit_tpe, Pinit_tpe, Yinit_tpe, Finit_tpe

    # for each batch_size, find the max function value of all arms
    def runoptimBatchList(self, trials, budget, batch_list):
        # store batch_list and budget
        # batch_list is a list of batch_size, e.g., [1, 2, 3, 4, 5]
        self.batch_list = batch_list
        self.budget = budget
        n_batch = len(batch_list)
        # a matrix where each col is a batch_size and each ele in a col is the best func value so far for each iteration
        self.mean_bestVals_batch = np.zeros((budget, n_batch))
        self.mean_errVals_batch  = np.zeros((budget, n_batch))
        # a matrix where each col is a batch_size and each ele in a col is the frequency of each arm
        self.mean_histArms_batch = np.zeros((self.n_arm, n_batch))
        
        for batch_idx in range(n_batch):
            # get a batch_size
            b = batch_list[batch_idx]
            print("Running: {}-{} with budget: {} for batch_size: {}".format(self.method, self.params, budget, b))
            self.mean_bestVals_batch[:, batch_idx], self.mean_errVals_batch[:, batch_idx], \
            self.mean_histArms_batch[:, batch_idx] = self.runTrials(trials, budget, b)
            # reset the set of arm recommendations for a new batch_size
            self.arm_recommendations = []
            
    # for each batch_size, run multiple trials
    def runTrials(self, trials, budget, b):
        # a matrix where each col is a trial and each ele in a col is the best input so far for each iteration
        best_inps = np.zeros((budget, trials, self.n_dim))
        # a matrix where each col is a trial and each ele in a col is the best func value so far for each iteration
        best_vals = np.zeros((budget, trials))
        # a matrix where each col is a trial and each ele in a col is the frequency of an arm
        hist_arms = np.zeros((self.n_arm, trials))
        
        for i in range(trials):
            print("Running: {}-{} for trial: {}".format(self.method, self.params, i))
            self.trial_num = i
            initData = None
            initResult = None
            done = False
            attempt_cnt = 0
            while not done and attempt_cnt < 20:
                try:
                    # run for 1 trial with <budget> iterations
                    df = self.runOptim(budget, b, initData, initResult)
                    if self.method == "TPE":
                        best_inps[:, i] = np.array([list(bestx.values()) for bestx in df["best_input"].values])
                    else:
                        best_inps[:, i] = np.array([bestx for bestx in df["best_input"].values])
                    best_vals[:, i] = df['best_value']
                    # compute the arm_recommendation histogram for 1 trial
                    arm_recommendations_1trail = self.arm_recommendations[i * budget:(i + 1) * budget]
                    arm_hist_1trial = collections.Counter(np.array(arm_recommendations_1trail).ravel())
                    print("trial: {}, arm_recommend: {}".format(self.trial_num, arm_recommendations_1trail))
                    # store the frequency of each arm for all trials for 1 batch_size
                    arm_hist_1trial_dict = dict(arm_hist_1trial)
                    arm_hist_1trial_array = np.zeros(self.n_arm)
                    for k in arm_hist_1trial_dict.keys():
                        arm_hist_1trial_array[k] = arm_hist_1trial_dict[k]
                    hist_arms[:, i] = arm_hist_1trial_array
                    done = True
                except Exception as exception:
                    print("got exception: ", exception.__class__.__name__)
                    print("arms pulled till now: ", self.arm_recommendations)
                    if self.method == "TPE":
                        initData, initPoint, initResult, initValue = self.initialize_all_methods()
                    else:
                        initData, initResult = self.initialize_all_methods()
                    attempt_cnt = attempt_cnt + 1
                    print("try again! {} times".format(attempt_cnt))
            if not done:
                sys.exit()
            # store result of each trial
            if self.save_result == True:
                with open("{}_best_inps_b{}_run{}.pickle".format(self.method + "-" + self.params, b, i), "wb") as f:
                    pickle.dump(best_inps, f)
                with open("{}_best_vals_b{}_run{}.pickle".format(self.method + "-" + self.params, b, i), "wb") as f:
                    pickle.dump(best_vals, f)
                with open("{}_hist_arms_b{}_run{}.pickle".format(self.method + "-" + self.params, b, i), "wb") as f:
                    pickle.dump(hist_arms, f)

        print("batch_size: {}, arm_recommend: {}".format(b, self.arm_recommendations))
        # compute mean and standard error for the best func value for each iteration for all trials
        mean_best_vals = np.mean(best_vals, axis=1)
        err_best_vals = np.std(best_vals, axis=1) / np.sqrt(trials)
        # compute mean for the frequency for each arm for all trials
        mean_hist_arms = np.mean(hist_arms, axis=1)
        # store the best input in each iteration in all trials for 1 batch_size
        self.b_best_inps = best_inps
        # store the best func value in each iteration in all trials for 1 batch_size
        self.b_best_vals = best_vals
        # store the frequency of each arm in all trials for 1 batch_size
        self.b_hist_arms = hist_arms

        return mean_best_vals, err_best_vals, mean_hist_arms

             
# =============================================================================
#     Over-ride this!
# =============================================================================
    def runOptim(self, budget, b, initData, initResult):
        print("Over-ride me!")

    # find best function value so far of all arms
    def getBestFuncVal(self, result):
        # result is an array of arrays, each element is an array of y_values for an arm
        # get max function value of each arm
        temp = [np.max(i) for i in result]

        # get max function value of all arms
        return np.max(temp)

    # find worst function value so far of all arms
    def getWorstFuncVal(self, result):
        # result is an array of arrays, each element is an array of y_values for an arm
        # get min function value of each arm
        temp = [np.min(i) for i in result]

        # get min function value of all arms
        return np.min(temp)



