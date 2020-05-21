# Bandit-BO: Bayesian Optimization for Categorical and Category-Specific Continuous Inputs
This is the implementation of the Bandit-BO method in the paper "Bayesian Optimization for Categorical and Category-Specific Continuous Inputs", AAAI 2020: https://aaai.org/Papers/AAAI/2020GB/AAAI-NguyenD.4977.pdf

# Introduction
Many real-world functions are defined over both categorical and category-specific continuous variables and thus cannot be optimized by traditional Bayesian optimization (BO) methods. For example, hyper-parameter tunning for a neural network involves both "activation" (categorical variable) and learning rate (continuous variable). Similarly, in automated machine learning where the goal is to find the best machine learning model along with its optimal hyper-parameters, we can view each model (e.g. decision tree) as a categorical variable and its hyper-parameters (e.g. depth) as continuous variables.

To optimize such functions, we propose a new method that formulates the problem as a multi-armed bandit problem, wherein each category corresponds to an arm with its reward distribution centered around the optimum of the objective function in continuous variables. Our goal is to identify the best arm and the maximizer of the corresponding continuous function simultaneously. Our algorithm uses a Thompson sampling scheme that helps connecting both multi-arm bandit and BO in a unified framework.

## Bandit-BO Demonstration
![main_idea](https://github.com/nphdang/Bandit-BO/blob/master/main_idea.jpg)

# Installation
1. Gpy 1.9.5 (to run Merchan-Lobato method)
2. GpyOpt 1.2.5 (to run One-hot-Encoding and SMAC methods)
3. Hyperopt 0.1.1 (to run TPE)
## Note 
Since the baselines don't support batch optimization, you need to copy the files in folder "packages" to replace the corresponding files in Gpy, GpyOpt, and Hyperopt after installing them.

# How to run
- Run "python demo_synfunc_1d_C6_b1_5_10.py" to optimize 2d synthetic function (1 categorical + 1 continuous) with the number of arms C=6 (fixed) and the batch size=[1, 5, 10]. Change variable "batch_list" to run with different batch sizes.
- Run "python demo_synfunc_1d_C25_b5.py" to optimize 2d synthetic function (1 categorical + 1 continuous) with the number of arms C=25 and the batch size=5 (fixed). Change variable "c_bound_dim" to run with different numbers of arms.
- Run "python plot_results.py" to plot the results. Change four variables test_case = "1d_C6", n_arm = 6, budget = 40, and batch_list = [1] to plot the corresponding result.

# Automated Machine Learning
- Install three automated machine learning packages
1. Hyperopt-sklearn: https://github.com/hyperopt/hyperopt-sklearn
2. Auto-sklearn: https://github.com/automl/auto-sklearn 
3. Tree-Based Pipeline Optimization Tool (TPOT): https://github.com/EpistasisLab/tpot
- Run "python demo_auto_ml.py --dataset iris --method all" to test the performance of all methods on the dataset "iris".

# Reference
Dang Nguyen, Sunil Gupta, Santu Rana, Alistair Shilton, Svetha Venkatesh (2020). Bayesian Optimization for Categorical and Category-Specific Continuous Inputs. AAAI 2020, New York, USA
