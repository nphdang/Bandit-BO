import numpy as np
import pandas as pd
import copy
from hyperopt import tpe, fmin
from functools import partial
from hyperopt.fmin import generate_trials_to_calculate
from mab_cons.MAB_Cons import MAB_Cons

# optimize categorical and continuous variables using TPE in HyperOpt
# initialize points in the same way with BO-based methods
class TPE(MAB_Cons):
    def __init__(self, objfn, n_init, bounds, acq_type, n_arm, save_result=False, rand_seed=108, dataset="", f_type="func"):
        super(TPE, self).__init__(objfn, n_init, bounds, acq_type, n_arm, save_result, rand_seed)
        self.method = "TPE"
        self.params = "(f_type={}, dataset={})".format(f_type, dataset)
        # which type of function to optimize
        # if it is a general function "func", TPE minimizes the function
        # if it is a classifier "class", TPE minimizes the loss
        self.f_type = f_type

    # run for 1 trial with <budget> iterations
    # given a batch_size b, find the best func value of all arms within <budget> iterations
    def runOptim(self, budget, b, initData=None, initResult=None):
        if initData is not None and initResult is not None:
            Xinit = initData[:]
            Yinit = initResult[:]
        else:
            # set a random number
            np.random.seed(self.trial_num)
            # Xinit: points in BO format and Pinit: points in TPE format
            # Yinit: max/auc values and Finit: min/loss values
            Xinit, Pinit, Yinit, Finit = self.initialize_all_methods()

        # get variable names
        bounds_keys = list(self.bounds.keys())
        # compute besty of initial points
        besty = np.max(Yinit)
        # compute bestx of initial points
        besty_idx = np.argmax(Yinit)
        bestp = Pinit[besty_idx]
        # get selected arm for categorical variable (the first parameter)
        first_para_name = bounds_keys[0]
        bestarm = bestp[first_para_name]
        print("n_init: {}, bestarm: {}, bestx: {}, besty: {}".format(len(Finit), bestarm, bestp, round(besty, 4)))

        # store the result for this trial (becoming one col of matrix best_vals)
        result_list = []
        # store the selected arms of initial points
        arm_list = []
        for b_ele_idx in range(b):
            # store selected arms in all iterations for all trials
            self.arm_recommendations.append(bestarm)
            arm_list.append(bestarm)
        # store the bestx and besty of initial points
        result_list.append([0, arm_list, bestp, besty])
        # use initial points for TPE
        tpe_algorithm = partial(tpe.suggest, n_startup_jobs=len(Finit))
        # create trials with initial points
        trials = generate_trials_to_calculate(Pinit, Finit)

        # store best point and best function value so far
        bestx_sofar = []
        besty_sofar = []
        if b > 1:
            budget = int(budget / b) + 1
        for t in range(1, budget):
            print("iteration: {}".format(t))
            # store <batch_size> arms selected in this iteration
            arm_list = []
            # store suggested data points in batch
            x_batch = np.zeros((b, self.n_dim))  # batch_size x dim of a data point
            y_batch = np.zeros(b) # store max function values
            f_batch = np.zeros(b) # store min function values
            # in an iteration, suggest a batch of points
            # only after selecting all points in the batch, we can compute their function values
            for b_ele_idx in range(b):
                # run TPE to suggest the next point which is stored in trials
                best_params = fmin(self.f, self.bounds, tpe_algorithm, len(trials) + 1, trials)
                # get best_x and best_y so far
                bestx_sofar.append(best_params)
                # max/auc of objective function
                best_result = trials.best_trial["result"]["loss"]
                if self.f_type == "func":
                    best_result = -1.0 * best_result
                elif self.f_type == "class":
                    best_result = 1.0 - best_result
                besty_sofar.append(best_result)
                # get selected arm for categorical variable (the first parameter)
                first_para_name = bounds_keys[0]
                arm = int(trials.vals[first_para_name][-1])
                # store selected arms in all iterations for all trials
                self.arm_recommendations.append(arm)
                arm_list.append(arm)
                # get other variables
                x_next = []
                for d in range(1, self.n_dim):
                    para_name = bounds_keys[d]
                    x_next.append(trials.vals[para_name][-1])
                x_batch[b_ele_idx, :] = [arm] + x_next
                # get function value of the next point (indeed, we don't know this function value)
                y_next = trials.results[-1]["loss"]
                f_batch[b_ele_idx] = y_next  # store min function value
                if self.f_type == "func":
                    y_next = -1.0 * y_next
                elif self.f_type == "class":
                    y_next = 1.0 - y_next
                y_batch[b_ele_idx] = y_next  # store max function value
                print("arm_next: {}, x_next: {}, y_next: {}".format(arm, np.around(x_next, 4), round(y_next, 4)))
                if b > 1:
                    # reset trials to suggest the next batch element
                    trials = generate_trials_to_calculate(Pinit, Finit)
            # end batch

            if b > 1:
                # update the data with suggested points in batch
                for ele_idx, x in enumerate(x_batch):
                    point = {bounds_keys[idx]: val for idx, val in enumerate(x)}
                    point[bounds_keys[0]] = int(point[bounds_keys[0]])
                    Pinit.append(point)
                    Finit.append(f_batch[ele_idx])
                # create trails with new batch elements
                trials = generate_trials_to_calculate(Pinit, Finit)

            # instead of computing function values of batch elements,
            # we already have them in y_batch

            for b_ele_idx in range(b):
                # get the best function value till now
                end = (t - 1) * b + (b_ele_idx + 1)
                besty = max(besty_sofar[:end])
                bestx = bestx_sofar[np.argmax(besty_sofar[:end])]
                # get selected arm for categorical variable (the first parameter)
                first_para_name = bounds_keys[0]
                bestarm = bestx[first_para_name]
                print("bestarm: {}, bestx: {}, besty: {}".format(bestarm, bestx, round(besty, 4)))
                # store the results of this iteration
                result_list.append([t, arm_list, bestx, besty])
        if b > 1:
            result_list = result_list[:-1]
        print("Finished ", self.method, " for trial: ", self.trial_num)
        # store the result for all iterations in this trial
        df = pd.DataFrame(result_list, columns=["iter", "arm_list", "best_input", "best_value"])

        return df

