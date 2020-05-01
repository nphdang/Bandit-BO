import numpy as np
import pandas as pd
import copy
import timeit
import sklearn.gaussian_process as gp
from scipy.optimize import minimize
from mab_cons.MAB_Cons import MAB_Cons
from utils import gen_func

# MAB: use TS, compute posterior from GP and maximize function drawn from posterior using optimizer
# BO: use TS
# Don't use Hallucinate
class BanditBO(MAB_Cons):
    def __init__(self, objfn, n_init, bounds, acq_type, n_arm, save_result=False, rand_seed=108,
                 dataset="", norm_y=True, ts_draw="func"):
        super(BanditBO, self).__init__(objfn, n_init, bounds, acq_type, n_arm, save_result, rand_seed)
        self.method = "BanditBO"
        self.params = "(norm_y={}, ts_draw={}, dataset={})".format(norm_y, ts_draw, dataset)
        # normalize y before fitting GP
        self.norm_y = norm_y
        # draw a function from GP posterior as a set or a function (Lobato method)
        self.ts_draw = ts_draw

    # run for 1 trial with <budget> iterations
    # given a batch_size b, find the best func value of all arms within <budget> iterations
    def runOptim(self, budget, b, initData=None, initResult=None):
        if initData is not None and initResult is not None:
            self.data = initData[:]
            self.result = initResult[:]
        else:
            # set a random number
            np.random.seed(self.trial_num)
            self.data, self.result = self.initialize_all_methods()

        # compute besty of initial points
        besty = np.max(self.result)
        besty_idx = np.argmax(self.result)
        besty_row = int(besty_idx / self.n_init)
        besty_col = besty_idx % self.n_init
        assert besty == float(self.result[besty_row][besty_col])
        # compute bestx of initial points
        bestx = self.data[besty_row][besty_col]
        bestarm = besty_row
        print("n_init: {}, bestarm: {}, bestx: {}, besty: {}".format(len(self.result), bestarm,
                                                                     np.around(bestx, 4), round(besty, 4)))

        # keep bestarm, bestx, and besty so far
        bestarm_sofar = bestarm
        bestx_sofar = bestx
        besty_sofar = besty

        n_samples = self.nAcq_opt  # no of initial points to maximize the function drawn from posterior of a given arm
        posteriorY = np.zeros(self.n_arm)  # y values drawn from reward posterior of each action
        posteriorX = np.zeros((self.n_arm, self.n_dim)) # x values drawn from reward posterior of each action
        # bound of data point
        x_bounds = np.zeros((self.n_dim, 2))
        for d in range(self.n_dim):
            x_bounds[d, :] = np.array([self.bounds[d]['domain'][0], self.bounds[d]['domain'][1]])

        # store the result for this trial (becoming one col of matrix best_vals)
        result_list = []
        # store the selected arms of initial points
        arm_list = []
        for b_ele_idx in range(b):
            # store selected arms in all iterations for all trials
            self.arm_recommendations.append(bestarm)
            arm_list.append(bestarm)
        # store the best mse and runtime of initial points
        result_list.append([0, arm_list, bestx, besty])

        if b > 1:
            budget = int(budget / b) + 1
        # store <batch_size> arms selected in the previous iteration
        arm_list_prev = []
        for t in range(1, budget):
            print("iteration: {}".format(t))
            # each iteration, select a batch of arms
            # store suggested points for arms in batch and their y_values
            # note that y_values are not true y_values, they are predictive means
            x_next = np.zeros((b, self.n_dim)) # batch_size x dim of a data point
            y_next = np.zeros(b)
            # store data observed so far
            # array of arrays: <n_arm> arms x no of data points
            tempX = copy.deepcopy(self.data)
            tempY = copy.deepcopy(self.result)
            # store <batch_size> arms selected in this iteration
            arm_list = []
            if t == 1:  # first iteration
                start_time = timeit.default_timer()
                # fit and update GP for all arms
                gp_set = dict()
                for c in range(self.n_arm):
                    # tune both amplitude and length-scale
                    kernel = 1.0 * gp.kernels.RBF(length_scale=1.0)
                    model_gp = gp.GaussianProcessRegressor(kernel=kernel,
                                                           alpha=1e-5,
                                                           n_restarts_optimizer=10,
                                                           normalize_y=False)  # improvement
                    xp_func = np.array(tempX[c])
                    yp_func = np.array([y[0] for y in tempY[c]])
                    model_gp.fit(xp_func, yp_func)
                    gp_set["gp{}".format(c)] = model_gp
                end_time = timeit.default_timer()
                print("time for fitting GPs for all arms: {}(s)".format(round(end_time - start_time, 2)))
            elif t > 1:
                start_time = timeit.default_timer()
                # fit and update GP for only arms selected in the previous iteration
                for c in arm_list_prev:
                    # tune both amplitude and length-scale
                    kernel = 1.0 * gp.kernels.RBF(length_scale=1.0)
                    model_gp = gp.GaussianProcessRegressor(kernel=kernel,
                                                           alpha=1e-5,
                                                           n_restarts_optimizer=10,
                                                           normalize_y=False)  # improvement
                    xp_func = np.array(tempX[c])
                    yp_func = np.array([y[0] for y in tempY[c]])
                    model_gp.fit(xp_func, yp_func)
                    gp_set["gp{}".format(c)] = model_gp
                end_time = timeit.default_timer()
                print("time for fitting GPs for selected arms: {}(s)".format(round(end_time - start_time, 2)))

            for b_ele_idx in range(b):
                start_time = timeit.default_timer()
                # compute samples drawn from reward posterior of each arm
                for c in range(self.n_arm):
                    if self.ts_draw == "func":
                        # draw a sample from y-value posterior of a given arm
                        func = gen_func.sample_gp_with_random_features(gp_set["gp{}".format(c)], nFeatures=1000,
                                                                       use_woodbury_if_faster=True)
                        # minus of the function draw from posterior of an arm
                        def minus_func(x):
                            return -func(x)
                        # explore the parameter space to optimize a function drawn from GP posterior
                        Xtest = np.zeros((n_samples, self.n_dim))
                        for i in range(n_samples):
                            Xtest[i, :] = np.array([np.random.uniform(self.bounds[d]['domain'][0], self.bounds[d]['domain'][1], 1)[0]
                                                    for d in range(self.n_dim)])
                        # use a local optimizer
                        best_xtest = []
                        best_ytest = -np.inf
                        for starting_point in Xtest:
                            # find the minimum of minus the function draw from posterior
                            res = minimize(fun=minus_func, x0=starting_point, bounds=x_bounds, method="L-BFGS-B")
                            # check if success
                            if not res.success:
                                continue
                            # update best_ytest so far
                            if -res.fun > best_ytest:
                                best_xtest = res.x
                                best_ytest = -res.fun
                    elif self.ts_draw == "set":
                        n_samples_rs = n_samples
                        X_rs = np.zeros((n_samples_rs, self.n_dim))
                        for i in range(n_samples_rs):
                            X_rs[i, :] = np.array([np.random.uniform(self.bounds[d]['domain'][0], self.bounds[d]['domain'][1], 1)[0]
                                                   for d in range(self.n_dim)])
                        model_gp = gp_set["gp{}".format(c)]
                        Y_rs = model_gp.sample_y(X_rs, n_samples=1)
                        best_ytest = np.max(Y_rs)
                        best_xtest = X_rs[np.argmax(Y_rs), :]
                    posteriorX[c, :] = best_xtest
                    posteriorY[c] = best_ytest
                # print("posterior_g(x): {}".format(posteriorY))
                # choose the arm which has the highest posterior y-value
                arm = np.argmax(posteriorY)
                end_time = timeit.default_timer()
                print("time for finding the best arm: {}(s)".format(round(end_time - start_time, 2)))
                # store selected arms in all iterations for all trials
                self.arm_recommendations.append(arm)
                # store selected arms in this iteration
                arm_list.append(arm)
                # suggest the next point for the selected arm
                if self.norm_y == True:
                    start_time = timeit.default_timer()
                    # use Thompson Sampling to suggest 1 next data point and its corresponding (predictive) mean
                    # tune both amplitude and length-scale
                    kernel = 1.0 * gp.kernels.RBF(length_scale=1.0)
                    model_gp = gp.GaussianProcessRegressor(kernel=kernel,
                                                           alpha=1e-5,
                                                           n_restarts_optimizer=10,
                                                           normalize_y=True) # improvement
                    xp_func = np.array(tempX[arm])
                    yp_func = np.array([y[0] for y in tempY[arm]])
                    model_gp.fit(xp_func, yp_func)
                    # draw a sample from y-value posterior of a given arm
                    func = gen_func.sample_gp_with_random_features(model_gp, nFeatures=1000,
                                                                   use_woodbury_if_faster=True)
                    # minus of the function draw from posterior of an arm
                    def minus_func(x):
                        return -func(x)
                    # explore the parameter space to optimize a function drawn from GP posterior
                    Xtest = np.zeros((n_samples, self.n_dim))
                    for i in range(n_samples):
                        Xtest[i, :] = np.array([np.random.uniform(self.bounds[d]['domain'][0], self.bounds[d]['domain'][1], 1)[0]
                                                for d in range(self.n_dim)])
                    # use a local optimizer
                    best_xtest = []
                    best_ytest = -np.inf
                    for starting_point in Xtest:
                        # find the minimum of minus the function draw from posterior
                        res = minimize(fun=minus_func, x0=starting_point, bounds=x_bounds, method="L-BFGS-B")
                        # check if success
                        if not res.success:
                            continue
                        # update best_ytest so far
                        if -res.fun > best_ytest:
                            best_ytest = -res.fun
                            best_xtest = res.x
                    # return 1 next data point x_next[b_ele_idx]
                    x_next[b_ele_idx, :] = best_xtest
                    # return predictive mean of x_next[b_ele_idx] (we don't compute f(x_next[b_ele_idx]))
                    y_next[b_ele_idx] = best_ytest
                    end_time = timeit.default_timer()
                    print("time for using TS to suggest the next point: {}(s)".format(round(end_time - start_time, 2)))
                else:
                    # use bestx and besty of the selected arm as the suggested point
                    start_time = timeit.default_timer()
                    # return 1 next data point x_next[b_ele_idx]
                    x_next[b_ele_idx, :] = posteriorX[arm, :]
                    # return predictive mean of x_next[b_ele_idx] (we don't compute f(x_next[b_ele_idx]))
                    y_next[b_ele_idx] = posteriorY[arm]
                    end_time = timeit.default_timer()
                    print("time for using bestx and besty of the selected arm as the next point: {}(s)".format(round(end_time - start_time, 2)))
            # end batch
            # store current selected arms
            arm_list_prev = arm_list
            
            # update the data and result with the true function values
            for idx, arm in enumerate(arm_list):
                y_next = self.f(arm, x_next[idx, :])
                self.data[arm] = np.row_stack((self.data[arm], x_next[idx, :]))
                self.result[arm] = np.row_stack((self.result[arm], y_next))
                y_next = y_next[0][0]
                print("arm_next: {}, x_next: {}, y_next: {}".format(arm, np.around(x_next[idx], 4), round(y_next, 4)))

                # get the best function value for all arms till now
                if y_next > besty_sofar:
                    besty_sofar = y_next
                    bestarm_sofar = arm
                    bestx_sofar = x_next[idx, :]
                besty = self.getBestFuncVal(self.result)
                assert besty == besty_sofar
                print("bestarm: {}, bestx: {}, besty: {}".format(bestarm_sofar, np.around(bestx_sofar, 4), round(besty_sofar, 4)))
                # store the result of this iteration
                result_list.append([t, arm_list, bestx_sofar, besty_sofar])
        if b > 1:
            result_list = result_list[:-1]
        print("Finished ", self.method, " for trial: ", self.trial_num)
        # store the result for all iterations in this trial
        df = pd.DataFrame(result_list, columns=["iter", "arm_list", "best_input", "best_value"])

        return df

