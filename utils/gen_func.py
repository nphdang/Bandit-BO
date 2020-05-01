import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import numpy.linalg as npla

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

# nFeatures is m in the paper, i.e., m-dimensional vector f
def sample_gp_with_random_features(gau_pro, nFeatures, use_woodbury_if_faster=True):
    d = gau_pro.X_train_.shape[1] # dimension of a data point x
    # N_data is n in the paper
    N_data = gau_pro.X_train_.shape[0] # number of observed data points
    # noise value: nu2 is sigma2 in the paper
    nu2 = gau_pro.get_params()["alpha"]
    # sigma2 is alpha in the paper
    sigma2 = 1.0

    if "kernel__k2" in gau_pro.get_params().keys():
        # get kernel name
        ker_name = str(gau_pro.get_params()["kernel__k2"])
        ker_name = ker_name[:ker_name.index("(")]
        # get kernel length scale
        ker_ls = gau_pro.get_params()["kernel__k2__length_scale"]
    else:
        # get kernel name
        ker_name = str(gau_pro.get_params()["kernel"])
        ker_name = ker_name[:ker_name.index("(")]
        # get kernel length scale
        ker_ls = gau_pro.get_params()["kernel__length_scale"]
    # print("kernel: {}, ls: {}".format(ker_name, ker_ls))

    # draw the random features W
    if ker_name == "RBF":
        W = npr.randn(nFeatures, d) / ker_ls
    elif ker_name == "Matern":
        m = 5.0 / 2.0
        W = npr.randn(nFeatures, d) / ker_ls / np.sqrt(npr.gamma(shape=m, scale=1.0 / m, size=(nFeatures, 1)))
    else:
        raise Exception("this random feature sampling is for RBF or Matern kernels and you are using {}".format(ker_name))
    # draw b
    b = npr.uniform(low=0, high=2 * np.pi, size=nFeatures)[:, None]

    randomness = npr.randn(nFeatures) # Gaussian dist. with mean 0 and variance 1

    # W has size nFeatures by d
    # tDesignMatrix has size Nfeatures by Ndata
    # woodbury has size Ndata by Ndata
    # z is a vector of length nFeatures

    if N_data > 0:
        gp_inputs = gau_pro.X_train_
        # tDesignMatrix is matrix theta_transpose in the paper
        tDesignMatrix = np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, gp_inputs.T) + b)

        if use_woodbury_if_faster and N_data < nFeatures:
            # have a cost of N^2d instead of d^3 by doing this woodbury thing (ref. to Appendix B.2 in [23])

            # obtain the posterior on the coefficients
            # woodbury is matrix A in the paper
            woodbury = np.dot(tDesignMatrix.T, tDesignMatrix) + nu2 * np.eye(N_data)
            chol_woodbury = spla.cholesky(woodbury)
            z = np.dot(tDesignMatrix, gau_pro.y_train_ / nu2)
            m = z - np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), np.dot(tDesignMatrix.T, z)))

            D, U = npla.eigh(woodbury)
            # sort the eigenvalues
            idx = D.argsort()[::-1]  # in decreasing order instead of increasing
            D = D[idx]
            U = U[:, idx]
            R = 1.0 / (np.sqrt(D) * (np.sqrt(D) + np.sqrt(nu2)))

            # sample from the posterior of the coefficients
            theta = randomness - \
                    np.dot(tDesignMatrix, np.dot(U, (R * np.dot(U.T, np.dot(tDesignMatrix.T, randomness))))) + m
        else:
            # all you are doing here is sampling from the posterior of the linear model that approximates the GP
            approx_Kxx = np.dot(tDesignMatrix, tDesignMatrix.T)
            chol_Sigma_inverse = spla.cholesky(approx_Kxx + nu2 * np.eye(nFeatures))
            Sigma = chol2inv(chol_Sigma_inverse)
            m = spla.cho_solve((chol_Sigma_inverse, False), np.dot(tDesignMatrix, gau_pro.y_train_))
            theta = m + np.dot(randomness, spla.cholesky(Sigma * nu2, lower=False)).T
    else:
        # sample from the prior -- same for Matern
        theta = npr.randn(nFeatures)

    # return function f^(i)(x) in the paper
    def wrapper(x):
        if x.ndim == 1:
            x = x[None]
        # result is f^(i)(x) in the paper
        result = np.dot(theta.T, np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b))
        if result.size == 1:
            result = float(result)  # if the answer is just a number, take it out of the numpy array wrapper

        return result

    return wrapper





