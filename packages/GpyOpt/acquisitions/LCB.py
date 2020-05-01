# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionLCB(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(AcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  

    def _compute_exploration_weight1(self):
        n_dim = self.model.input_dim
        n_iter = len(self.model.model.X) - n_dim
        lengthscale = self.model.model.param_array[1]
        thetan_2 = self.model.model.param_array[0]
        x_bounds = np.zeros((n_dim, 2))
        for d in range(n_dim):
            x_bounds[d, :] = np.array([self.space.space[d].domain[0], self.space.space[d].domain[1]])
        radius = np.abs(np.max(x_bounds[:, 1] - x_bounds[:, 0]))
        b = np.sqrt(2) * np.sqrt(thetan_2) / lengthscale
        a = 1
        nu = 0.2
        sigma = 0.1
        tau_n = (4 * n_dim + 4) * np.log(n_iter) + 2 * np.log(2 * np.pi ** 2 / (3 * sigma)) + \
                2 * n_dim * np.log(n_dim * b * radius * np.sqrt(np.log(4 * n_dim * a / sigma)))
        b_n = np.sqrt(np.abs(nu * tau_n))
        return b_n

    def _compute_exploration_weight2(self):
        if self.space.has_types["categorical"] == True: # One-hot-Encoding and SMAC
            n_arms = self.space.model_input_dims[0]
            try: # One-hot-Encoding
                # dim of One-hot-Encoding is larger than the dim defined in bound in categorical setting
                # (extra variables for categorical variables)
                n_dim = self.model.input_dim - n_arms
                # n_iter = int(len(self.model.model.X) / n_arms) - n_dim
                # n_iter = no of function evaluations
                n_iter = len(self.model.model.X)
            except: # SMAC
                # dim of SMAC is larger than the dim defined in bound in categorical setting
                # (extra variables for categorical variables)
                n_dim = self.space.model_dimensionality - n_arms
                # n_iter = int(len(self.model.X) / n_arms) - n_dim
                # n_iter = no of function evaluations
                n_iter = len(self.model.X)
        else: # Merchan-Lobato and TS-UCB (or One-hot-Encoding, Merchan-Lobato, SMAC, TS-UCB in discrete and continuous cases)
            try:
                # dim of Merchan-Lobato is the same as bound in both categorical and discrete settings
                # dim of One-hot-Encoding and SMAC are the same as bound in discrete setting (as they just round continuous)
                n_dim = self.model.input_dim
                # in setting: n_iter = n_dim + 1
                # n_iter = len(self.model.model.X) - n_dim
                # n_iter = no of function evaluations
                n_iter = len(self.model.model.X)
            except: # SMAC
                # dim of SMAC is the same as bound in discrete setting
                n_dim = self.space.model_dimensionality
                # in setting: n_iter = n_dim + 1
                # n_iter = len(self.model.X) - n_dim
                # n_iter = no of function evaluations
                n_iter = len(self.model.X)
        try: # One-hot-Encoding, Merchan-Lobato, and TS-UCB
            lengthscale = self.model.model.param_array[1]
            thetan_2 = self.model.model.param_array[0]
        except: # SMAC
            lengthscale = 1.0
            thetan_2 = 1.0
        x_bounds = np.zeros((n_dim, 2))
        for d in range(n_dim):
            x_bounds[d, :] = np.array([self.space.space[d].domain[0], self.space.space[d].domain[1]])
        a = 1
        b = np.sqrt(2) * np.sqrt(thetan_2) / lengthscale
        radius = np.abs(np.max(x_bounds[:, 1] - x_bounds[:, 0]))
        delta = 0.1
        tau_n = 2 * np.log(2 * (n_iter ** 2) * (np.pi ** 2) / (3 * delta)) + \
                2 * n_dim * np.log((n_iter ** 2) * n_dim * b * radius * np.sqrt(np.log(4 * n_dim * a / delta)))
        nu = self.exploration_weight
        b_n = np.sqrt(np.abs(nu * tau_n))
        return b_n

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        """
        m, s = self.model.predict(x)   
        # f_acqu = -m + self.exploration_weight * s
        exploration_weight = self._compute_exploration_weight2()
        f_acqu = m + exploration_weight * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        # f_acqu = -m + self.exploration_weight * s
        exploration_weight = self._compute_exploration_weight2()
        f_acqu = m + exploration_weight * s
        # df_acqu = -dmdx + self.exploration_weight * dsdx
        df_acqu = dmdx + exploration_weight * dsdx
        return f_acqu, df_acqu

