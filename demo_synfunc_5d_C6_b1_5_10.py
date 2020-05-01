import numpy as np
import pickle
import timeit
import datetime
from hyperopt import hp
from mab_cons.OnehotEnconding import OnehotEncoding
from mab_cons.MerchanLobato import MerchanLobato
from mab_cons.SMAC import SMAC
from mab_cons.TPE import TPE
from mab_cons.BanditBO import BanditBO
import test_functions.synfunc_5d

np.random.seed(123)

test_case = "5d_C6"
f_bo = test_functions.synfunc_5d.obj_func_bo
f_tpe = test_functions.synfunc_5d.obj_func_tpe

c_bound_dim = 6
x_bound_dim = 5
x_bound = [-2.0, 2.0]

trials = 10 # 10
n_arm = c_bound_dim
n_dim = x_bound_dim
budget = 120 # 120
batch_list = [1] # [1, 5, 10]
n_init = 2

# BanditBO
bounds_mab = [{"name": "x{}".format(d+1), "type": "continuous", "domain": (x_bound[0], x_bound[1])}
              for d in range(x_bound_dim)]

# OnehotEncoding and SMAC
all_c = tuple(range(c_bound_dim))
bounds_gpy_c = [{'name': 'x1', 'type': 'categorical', 'domain': all_c}]
bounds_gpy_x = [{"name": "x{}".format(d+2), "type": "continuous", "domain": (x_bound[0], x_bound[1])}
                for d in range(x_bound_dim)]
bounds_gpy = list(np.append(bounds_gpy_c, bounds_gpy_x))

# MerchanLobato
bounds_categorical_c = [{"name": "x{}".format(d+1), "type": "continuous", "domain": (0, 1)}
                        for d in range(c_bound_dim)]
bounds_categorical_x = [{"name": "x{}".format(d+c_bound_dim+1), "type": "continuous", "domain": (x_bound[0], x_bound[1])}
                        for d in range(x_bound_dim)]
bounds_categorical = list(np.append(bounds_categorical_c, bounds_categorical_x))

# TPE
all_c = list(range(c_bound_dim))
bounds_tpe_c = {'x1': hp.choice('x1', all_c)}
bounds_tpe_x = {'x{}'.format(d+2): hp.uniform('x{}'.format(d+2), x_bound[0], x_bound[1]) for d in range(x_bound_dim)}
bounds_tpe = {**bounds_tpe_c, **bounds_tpe_x}

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# save models to files
batch_name = "_".join(map(str, batch_list))

# baselines
onehot = OnehotEncoding(objfn=f_bo, n_init=n_init, bounds=bounds_gpy, acq_type='LCB', n_arm=n_arm,
                        dataset=test_case, rand_seed=108)
onehot.runoptimBatchList(trials, budget, batch_list)
with open("{}_onehot_b{}.pickle".format(test_case, batch_name), "wb") as f:
    pickle.dump(onehot, f)

merchanlobato = MerchanLobato(objfn=f_bo, n_init=n_init, bounds=bounds_categorical, acq_type='LCB', n_arm=n_arm,
                              dataset=test_case, rand_seed=108)
merchanlobato.runoptimBatchList(trials, budget, batch_list)
with open("{}_merchanlobato_b{}.pickle".format(test_case, batch_name), "wb") as f:
    pickle.dump(merchanlobato, f)

smac = SMAC(objfn=f_bo, n_init=n_init, bounds=bounds_gpy, acq_type='LCB', n_arm=n_arm,
            dataset=test_case, rand_seed=108)
smac.runoptimBatchList(trials, budget, batch_list)
with open("{}_smac_b{}.pickle".format(test_case, batch_name), "wb") as f:
    pickle.dump(smac, f)

tpe = TPE(objfn=f_tpe, n_init=n_init, bounds=bounds_tpe, acq_type='', n_arm=n_arm,
          f_type="func", dataset=test_case, rand_seed=108)
tpe.runoptimBatchList(trials, budget, batch_list)
with open("{}_tpe_b{}.pickle".format(test_case, batch_name), "wb") as f:
    pickle.dump(tpe, f)

# proposed method
ts_draw = "set" # func, set
banditbo = BanditBO(objfn=f_bo, n_init=n_init, bounds=bounds_mab, acq_type='TS', n_arm=n_arm,
                    norm_y=True, ts_draw=ts_draw, dataset=test_case, rand_seed=108)
banditbo.runoptimBatchList(trials, budget, batch_list)
with open("{}_banditbo_b{}.pickle".format(test_case, batch_name), "wb") as f:
    pickle.dump(banditbo, f)

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))

