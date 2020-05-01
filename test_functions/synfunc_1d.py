import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from hyperopt import STATUS_OK

# =============================================================================
# Two-Gaussian function
# Bounds : [-2.0, 10.0]
# Global max = 1.4 at x = 2.0
# =============================================================================
def twogaussian(x, shifter=0.0):
    y = np.exp(-(x-(shifter*0.1)-2)**2)+np.exp(-(x-(shifter*0.1)-6)**2/10)+1/((x+(shifter*0.1))**2+1)

    return y.reshape(-1, 1) + shifter # Min=0.2, Max=1.4

# find minimum value
sign = -1

# objective function for BO-based methods
def obj_func_bo(arm, x):
    # maximize function
    auc = twogaussian(x, shifter=arm*0.5)

    return auc

# objective function for TPE method
def obj_func_tpe(params):
    arm = params["x1"]
    x = params["x2"]
    # maximize function
    auc = twogaussian(x, shifter=arm*0.5)
    # minimize function
    loss = sign*auc

    return {"auc": auc, "loss": loss, "status": STATUS_OK}

if __name__ == '__main__':
    x = np.linspace(-2, 10, 1000)
    # case 1
    y0 = twogaussian(x, 0*0.5)
    y1 = twogaussian(x, 1*0.5)
    y2 = twogaussian(x, 2*0.5)
    y3 = twogaussian(x, 3*0.5)
    y4 = twogaussian(x, 4*0.5)
    y5 = twogaussian(x, 5*0.5)
    plt.plot(x, y0, label="func 0")
    plt.plot(x, y1, label="func 1")
    plt.plot(x, y2, label="func 2")
    plt.plot(x, y3, label="func 3")
    plt.plot(x, y4, label="func 4")
    plt.plot(x, y5, label="func 5")
    plt.title("best func: {}, best y-value: {}".format(5, round(np.max(y5), 2)))
    plt.legend(loc=1)
    plt.show()

    from scipy.optimize import minimize
    n_dim = 1
    # explore the parameter space
    nStartPoints = 1000
    bounds = np.array([[-2.0, 10]]*n_dim)
    Xtest = np.zeros((nStartPoints, n_dim))
    for i in range(nStartPoints):
        Xtest[i, :] = np.array([np.random.uniform(bounds[d][0], bounds[d][1], 1)[0] for d in range(n_dim)])

    # find maximum
    best_xtest = []
    best_ytest = -np.inf
    def minus_func(x):
        return -twogaussian(x)
    for starting_point in Xtest:
        res = minimize(fun=minus_func, x0=starting_point, bounds=bounds, method="L-BFGS-B")
        # check if success
        if not res.success:
            continue
        # update best_ytest so far
        if -res.fun[0] > best_ytest:
            best_ytest = -res.fun[0]
            best_xtest = res.x
    # max_x: [2.], max_y: [1.4]
    print("max_x: {}, max_y: {}".format(np.around(best_xtest, 2), np.around(best_ytest, 2)))

    # find minimum
    best_xtest = []
    best_ytest = np.inf
    for starting_point in Xtest:
        res = minimize(fun=twogaussian, x0=starting_point, bounds=bounds, method="L-BFGS-B")
        # check if success
        if not res.success:
            continue
        # update best_ytest so far
        if res.fun[0] < best_ytest:
            best_ytest = res.fun[0]
            best_xtest = res.x
    # min_x: [-2.], min_y: [0.2]
    print("min_x: {}, min_y: {}".format(np.around(best_xtest, 2), np.around(best_ytest, 2)))




