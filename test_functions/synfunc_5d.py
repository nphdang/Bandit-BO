import numpy as np
from hyperopt import STATUS_OK

# =============================================================================
# Ackley function
# Bounds : [-2.0, 2.0]
# Global min = 0.0 at x = [0.0, 0.0, 0.0, 0.0, 0.0]
# =============================================================================
def ackley(X, input_dim=5, shifter=0, a=20, b=0.2, c=2*np.pi):
    X = X.reshape(-1, input_dim)
    n = X.shape[0]
    if n == 1:
        sum1 = np.sum((X + shifter) ** 2)
        sum2 = np.sum(np.cos(c * (X + shifter)))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / input_dim))
        term2 = -np.exp(sum2 / input_dim)
        y = term1 + term2 + a + np.exp(1)
    elif n > 1:
        y = np.zeros((n, 1))
        for idx in range(n):
            xx = X[idx, :]
            sum1 = np.sum((xx + shifter) ** 2)
            sum2 = np.sum(np.cos(c * (xx + shifter)))
            term1 = -a * np.exp(-b * np.sqrt(sum1 / input_dim))
            term2 = -np.exp(sum2 / input_dim)
            y[idx] = term1 + term2 + a + np.exp(1)

    return y.reshape(-1, 1) + shifter # Min=0, Max=20

# set dimension for the function
n_dim = 5
# find maximum value
sign = -1

# objective function for BO-based methods
def obj_func_bo(arm, X):
    # maximize function    
    auc = sign*ackley(X, input_dim=n_dim, shifter=arm)

    return auc

# objective function for TPE method
def obj_func_tpe(params):
    arm = params["x1"]
    X = [params["x2"], params["x3"], params["x4"], params["x5"], params["x6"]]
    X = np.array(X).reshape(-1, len(X)) # format X to [[]]
    # minimize function
    loss = ackley(X, input_dim=n_dim, shifter=arm)
    # maximize function
    auc = sign*loss

    return {"auc": auc, "loss": loss, "status": STATUS_OK}

if __name__ == '__main__':
    from scipy.optimize import minimize
    # explore the parameter space
    nStartPoints = 5000
    bounds = np.array([[-2, 2]]*n_dim)
    Xtest = np.zeros((nStartPoints, n_dim))
    for i in range(nStartPoints):
        Xtest[i, :] = np.array([np.random.uniform(bounds[d][0], bounds[d][1], 1)[0] for d in range(n_dim)])

    # find minimum
    best_xtest = []
    best_ytest = np.inf
    for starting_point in Xtest:
        res = minimize(fun=ackley, x0=starting_point, bounds=bounds, method="L-BFGS-B")
        # check if success
        if not res.success:
            continue
        # update best_ytest so far
        if res.fun[0] < best_ytest:
            best_ytest = res.fun[0]
            best_xtest = res.x
    # min_x: [-0.  0.  0. -0. -0.], min_y: [0.]
    print("min_x: {}, min_y: {}".format(np.around(best_xtest, 2), np.around(best_ytest, 2)))

    # find maximum
    best_xtest = []
    best_ytest = -np.inf
    def minus_func(X):
        return -ackley(X)
    for starting_point in Xtest:
        res = minimize(fun=minus_func, x0=starting_point, bounds=bounds, method="L-BFGS-B")
        # check if success
        if not res.success:
            continue
        # update best_ytest so far
        if -res.fun[0] > best_ytest:
            best_ytest = -res.fun[0]
            best_xtest = res.x
    # max_x: [-1.61  2.    1.61  1.61  1.61], max_y: [7.81]
    print("max_x: {}, max_y: {}".format(np.around(best_xtest, 2), np.around(best_ytest, 2)))

