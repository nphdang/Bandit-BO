import numpy as np
import pandas as pd
import copy
import GPyOpt
import GPy
from mab_cons.MAB_Cons import MAB_Cons

# optimize categorical and continuous variables using GpyOpt
class SMAC(MAB_Cons):
    def __init__(self, objfn, n_init, bounds, acq_type, n_arm, save_result=False, rand_seed=108, dataset=""):
        super(SMAC, self).__init__(objfn, n_init, bounds, acq_type, n_arm, save_result, rand_seed)
        self.method = "SMAC"
        self.params = "(dataset={})".format(dataset)

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
        # compute bestx of initial points
        besty_idx = np.argmax(self.result)
        bestx = self.data[besty_idx]
        bestarm = int(bestx[0])
        print("n_init: {}, bestarm: {}, bestx: {}, besty: {}".format(len(self.result), bestarm,
                                                                     np.around(bestx[1:], 4), round(besty, 4)))

        # store the result for this trial (becoming one col of matrix best_vals)
        result_list = []
        # store the select arms of initial points
        arm_list = []
        for b_ele_idx in range(b):
            # store selected arms in all iterations for all trials
            self.arm_recommendations.append(bestarm)
            arm_list.append(bestarm)
        # store the bestx and besty of initial points
        result_list.append([0, arm_list, bestx, besty])
        # reformat y-values
        self.result = np.array(self.result).reshape(-1, 1)

        if b > 1:
            budget = int(budget / b) + 1
        for t in range(1, budget):
            print("iteration: {}".format(t))
            # each iteration, select a batch of arms
            # store suggested points for arms in batch and their y_values
            # note that y_values are not true y_values, they are predictive means
            x_next = np.zeros((b, self.n_dim))  # batch_size x dim of a data point
            y_hall = np.zeros(b)
            # store data observed so far
            # array of arrays: <n_arm> armss x no of data points
            tempX = copy.deepcopy(self.data)
            tempY = copy.deepcopy(self.result)
            # store <batch_size> arms selected in this iteration
            arm_list = []

            for b_ele_idx in range(b):
                my_kernel = GPy.kern.ExpQuad(input_dim=self.n_dim, variance=1.0, lengthscale=1.0)
                smac = GPyOpt.methods.BayesianOptimization(f=None, num_cores=self.n_core,
                                                           domain=self.bounds,
                                                           X=tempX, Y=tempY,
                                                           normalize_Y=True, maximize=True,
                                                           kernel=my_kernel,
                                                           acquisition_type=self.acq_type, acquisition_weight=1.0,
                                                           model_type="RF")
                # return 1 next data point x_next[b_ele_idx]
                x_next[b_ele_idx, :] = smac.suggest_next_locations()
                # convert x_next[b_ele_idx] to one-hot encoding
                x_one_hot = np.zeros(self.n_arm)
                x_one_hot[int(x_next[b_ele_idx, 0])] = 1
                x_one_hot = np.append(x_one_hot, x_next[b_ele_idx, 1:])
                # get predictive mean of x_next[b_ele_idx] (we don't compute f(x_next[b_ele_idx]))
                y_hall[b_ele_idx] = smac.model.model.predict(np.atleast_2d(x_one_hot))[0]
                # get selected arm
                arm = int(x_next[b_ele_idx][0])
                # store selected arms in all iterations for all trials
                self.arm_recommendations.append(arm)
                # store selected arms in this iteration
                arm_list.append(arm)
                # augment data
                tempX = np.vstack((tempX, x_next[b_ele_idx, :]))
                tempY = np.vstack((tempY, y_hall[b_ele_idx]))
            # end batch

            # update the data and result with the true function values
            for idx, arm in enumerate(arm_list):
                y_next = self.f(arm, x_next[idx, 1:])
                # augment data
                self.data = np.vstack((self.data, x_next[idx, :]))
                self.result = np.vstack((self.result, y_next))
                y_next = y_next[0][0]
                print("x_next: {}".format(np.around(x_next[idx, :], 4)))
                print("arm_next: {}, x_next: {}, y_next: {}".format(arm, np.around(x_next[idx, 1:], 4), round(y_next, 4)))

                # get the best function value for all arms till now
                besty = np.max(self.result)
                # compute bestx till now
                besty_idx = np.argmax(self.result)
                bestx = self.data[besty_idx]
                bestarm = int(bestx[0])
                print("bestarm: {}, bestx: {}, besty: {}".format(bestarm, np.around(bestx[1:], 4), round(besty, 4)))
                # store the results of this iteration
                result_list.append([t, arm_list, bestx, besty])
        if b > 1:
            result_list = result_list[:-1]
        print("Finished ", self.method, " for trial: ", self.trial_num)
        # store the result for all iterations in this trial
        df = pd.DataFrame(result_list, columns=["iter", "arm_list", "best_input", "best_value"])

        return df

