import numpy as np
import pandas as pd
import copy
import GPyOpt
from GPy import kern
from mab_cons.MAB_Cons import MAB_Cons

# map categorical values to one-hot encodings
# instead of computing the kernel function k(x1,x2), compute the kernel function k(T(x1),T(x2))
class MerchanLobato(MAB_Cons):
    def __init__(self, objfn, n_init, bounds, acq_type, n_arm, save_result=False, rand_seed=108, dataset=""):
        super(MerchanLobato, self).__init__(objfn, n_init, bounds, acq_type, n_arm, save_result, rand_seed)
        self.method = "MerchanLobato"
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
        bestarm = np.argmax(bestx[:self.n_arm])
        print("n_init: {}, bestarm: {}, bestx: {}, besty: {}".format(len(self.result), bestarm,
                                                                     np.around(bestx[self.n_arm:], 4), round(besty, 4)))

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
            # store <batch_size> arms selected in this iteration
            arm_list = []
            # each iteration, select a batch of arms
            # store suggested points for arms in batch and their y_values
            # note that y_values are not true y_values, they are predictive means
            bo = GPyOpt.methods.BayesianOptimization(f=None, num_cores=self.n_core,
                                                     domain=self.bounds,
                                                     X=self.data, Y=self.result,
                                                     normalize_Y=True, maximize=True,
                                                     acquisition_type=self.acq_type, acquisition_weight=1.0,
                                                     evaluator_type='thompson_sampling', batch_size=b,
                                                     kernel=kern.ExpQuad_Categorical(self.n_dim, categorical_dim=self.n_arm,
                                                                                     variance=1.0, lengthscale=1.0, ARD=False))
            # suggest a batch of arms and their x_next
            x_next = bo.suggest_next_locations()
            # map x_next from real-value vector to arm index and then compute their y_next
            y_next = [self.f(np.argmax(x[:self.n_arm]), x[self.n_arm:]) for x in x_next]
            y_next = np.array(y_next).reshape(-1, 1)
            for b_ele_idx in range(b):
                # get selected arm
                arm = np.argmax(x_next[b_ele_idx][:self.n_arm])
                print("x_next: {}".format(np.around(x_next[b_ele_idx, :], 4)))
                print("arm_next: {}, x_next: {}, y_next: {}".
                      format(arm, np.around(x_next[b_ele_idx][self.n_arm:], 4), round(y_next[b_ele_idx][0], 4)))
                # store selected arms in all iterations for all trials
                self.arm_recommendations.append(arm)
                # store selected arms in this iteration
                arm_list.append(arm)
                # augment data
                self.data = np.vstack((self.data, x_next[b_ele_idx]))
                self.result = np.vstack((self.result, y_next[b_ele_idx]))

                # get the best function value for all arms till now
                besty = np.max(self.result)
                # compute bestx till now
                besty_idx = np.argmax(self.result)
                bestx = self.data[besty_idx]
                bestarm = np.argmax(bestx[:self.n_arm])
                print("bestx: {}".format(np.around(bestx, 4)))
                print("bestarm: {}, bestx: {}, besty: {}".format(bestarm, np.around(bestx[self.n_arm:], 4), round(besty, 4)))
                # store the results of this iteration
                result_list.append([t, arm_list, bestx, besty])
            # end batch
        if b > 1:
            result_list = result_list[:-1]
        print("Finished ", self.method, " for trial: ", self.trial_num)
        # store the result for all iterations in this trial
        df = pd.DataFrame(result_list, columns=["iter", "arm_list", "best_input", "best_value"])

        return df

