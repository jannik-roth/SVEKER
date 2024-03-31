import abc
from typing import Literal, Mapping
from typing_extensions import override
import numpy as np
from numpy.random import RandomState
from .kernels import TanimotoKernel, RBFKernel, PolynomialKernel, SigmoidKernel
from sklearn.svm import SVR, SVC
from scipy.special import expit
from functools import cached_property

class ExplainingSVM(abc.ABC):
    '''
    Base class for SVMs with exact Shapley values
    '''
    def __init__(self, kernel_type = 'tanimoto', degree = 3, gamma = 1.0, coef0 = 0.0, no_player_value = 0.0) -> None:
        '''
        Initializes a SVM with exact Shapley values

        :param kernel_type: Kernel to be used, must be in ``['tanimoto', 'rbf', 'poly', 'sigmoid']``
        :type kernel_type: str
        :param gamma: Parameter :math:`\\gamma`, only for Polynomial, RBF, Sigmoid kernel.
        :type gamma: float
        :param degree: Parameter :math:`d`, only for Polynomial kernel, must be non-negative.
        :type degree: float
        :param coef0: Parameter :math:`r`, only for Polynomial, Sigmoid kernel.
        :type coef0: float
        '''
        super().__init__()
        self.no_player_value = no_player_value
        self.kernel_type = kernel_type

        match kernel_type.lower():
            case 'tanimoto':
                self._kernel = TanimotoKernel(no_player_value=no_player_value)
            case 'rbf':
                self.gamma = gamma
                self._kernel = RBFKernel(no_player_value=no_player_value, gamma=gamma)
            case 'poly':
                self.gamma = gamma
                self.degree = degree
                self.coef0 = coef0
                self._kernel = PolynomialKernel(no_player_value=no_player_value, gamma=gamma, degree=degree, coef0=coef0)
            case 'sigmoid':
                self.gamma = gamma
                self.coef0 = coef0
                self._kernel = SigmoidKernel(no_player_value=no_player_value, gamma=gamma, coef0=coef0)
            case _:
                NotImplementedError(f'The kernel {kernel_type} is not implemented!')

    def _vec_shapley_values(self, vector: np.ndarray):
        '''
        Calculate the Shapley values for a single(!) vector

        :param vector: Vector to be explained using Shapley values
        :type vector: np.ndarray
        :return: Shapley values for each feature in vector
        :rtype: np.ndarray
        '''
        # repeat the vector to explain
        rep_vector = np.tile(vector, (self._explicit_sup_vecs.shape[0], 1))
        # get the intersection of the vector with the support vectors
        inter = np.multiply(vector, self._explicit_sup_vecs)
         
        only_vector = rep_vector - inter
        only_sup_vec = self._explicit_sup_vecs - inter

        n_shared = inter.sum(axis=1)
        n_only_v = only_vector.sum(axis=1)
        n_only_sv = only_sup_vec.sum(axis=1)

        # combine i and d in matrix, transpose for convenience
        comb = np.vstack([n_shared, n_only_v + n_only_sv]).T

        weight_inter = np.array([self._kernel.phi_fplus(*vec) for vec in comb]).reshape(-1, 1)
        weight_diff = np.array([self._kernel.phi_fminus(*vec) for vec in comb]).reshape(-1, 1)

        # add contribution
        feat_cont = inter * weight_inter + (only_vector + only_sup_vec) * weight_diff

        # multiply with y_n w_n
        shapley_values = feat_cont * self.dual_coef_.reshape(-1, 1)

        return shapley_values.sum(axis=0)
    
    def shapley_values(self, x: np.ndarray):
        '''
        Calculate the Shapley values for multiple query vectors.

        :param vector: Vector(s) to be explained using Shapley values
        :type vector: np.ndarray
        :return: Shapley values for each feature in vector(s)
        :rtype: np.ndarray
        '''
        return np.vstack([self._vec_shapley_values(x[i, :]) for i in range(x.shape[0])])
    
class ExplainingSVR(SVR, ExplainingSVM):
    '''
    Class for SVR with exact Shapley values. Inherits from ``ExplainingSVM`` and from `sklearn.SVR <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_.

    For documentation of the ``sklearn`` methods, please take a look at the documentation there. Here, only the newly implemented methods will be discussed.
    '''
    def __init__(self, *, 
                 kernel_type: Literal['rbf', 'tanimoto', 'poly', 'simgoid'] = 'rbf',
                 degree: int = 3,
                 gamma: float = 1.0,
                 coef0: float = 0,
                 tol: float = 0.001,
                 C: float = 1, 
                 epsilon: float = 0.1, 
                 shrinking: bool = True, 
                 cache_size: float = 200,
                 verbose: bool = False, 
                 max_iter: int = -1,
                 no_player_value = 0.0) -> None:
        '''
        Initializes SVC with exact Shapley values.

        :param C: Regularization parameter
        :type C: float
        :param kernel_type: Kernel to be used, must be in ``['tanimoto', 'rbf', 'poly', 'sigmoid']``
        :type kernel_type: str
        :param gamma: Parameter :math:`\\gamma`, only for Polynomial, RBF, Sigmoid kernel.
        :type gamma: float
        :param degree: Parameter :math:`d`, only for Polynomial kernel, must be non-negative.
        :type degree: float
        :param coef0: Parameter :math:`r`, only for Polynomial, Sigmoid kernel.
        :type coef0: float
        :param epsilon:
        :type epsilon: float
        :param shrinking: Wheter to use shrinking heuristic
        :type shrinking: bool
        :param cache_size: Kernel chache size
        :type cache_size: float
        :param class_weight: Set the parameter :math:`C` of class :mahtL`i` to ``class_weight[i]*C`` for SVC, if not given, all classes will have the weight of unity.
        :type class_weight: dict or 'balanced'
        :param verbose: Enables verbose output
        :type verbose: bool
        :param max_iter: Maximum number of iterations, if -1 no limit
        :type max_iter: int
        :param no_player_value: Defines the no player value
        :type no_player_value: float
        '''
        ExplainingSVM.__init__(self, 
                               kernel_type=kernel_type, 
                               degree=degree, 
                               gamma=gamma, 
                               coef0=coef0, 
                               no_player_value=no_player_value)
        SVR.__init__(self,
                     kernel=self._kernel.get_kernel_function(), 
                     degree=degree, 
                     gamma=gamma, 
                     coef0=coef0, 
                     tol=tol, 
                     C=C, 
                     epsilon=epsilon, 
                     shrinking=shrinking, 
                     cache_size=cache_size, 
                     verbose=verbose, 
                     max_iter=max_iter)
        
    def set_params(self, **params):
        '''
        Sets the parameters of the class. Necessary in order to work with sklearn Pipelines and grid searches

        :param params: parameter to change with corresponding value
        :return: Returns Class with changed parameters.
        '''
        for param, value in params.items():
            setattr(self, param, value)
            if param in ['kernel_type', 'gamma', 'degree', 'coef0', 'no_player_value']:
                # change in kernel, need to initialize from scratch, mainly to be safe
                ExplainingSVM.__init__(self,
                                       kernel_type = params.get('kernel_type', self.get_params()['kernel_type']),
                                       degree = params.get('degree', self.get_params()['degree']),
                                       gamma = params.get('gamma', self.get_params()['gamma']),
                                       coef0 = params.get('coef0', self.get_params()['coef0']),
                                       no_player_value= params.get('no_player_value', self.get_params()['no_player_value']))
                self.kernel = self._kernel.get_kernel_function()
        
        return self
        
    def fit(self, X, y, sample_weight=None):
        '''
        Fits the classifier and saves the explicit support vectors.

        :param X: Training vectors
        :type X: array_like
        :param y: Target values
        :type y: array_like
        :param sample_weights: Per-sample weight
        :type sample_weights: array_like or None
        '''
        super().fit(X, y, sample_weight=sample_weight)
        self._explicit_sup_vecs = self._BaseLibSVM__Xfit[self.support_]
        return self
    
    @cached_property
    def expected_value(self):
        '''
        Expected value
        
        :return: Expected value
        :rtype: float
        '''
        return self.intercept_
    
    def shapley_values(self, x: np.ndarray):
        '''
        Calculate the Shapley values for multiple query vectors.

        :param vector: Vector(s) to be explained using Shapley values
        :type vector: np.ndarray
        :return: Shapley values for each feature in vector(s)
        :rtype: np.ndarray
        '''
        return super().shapley_values(x)
    
class ExplainingSVC(SVC, ExplainingSVM):
    '''
    Class for SVC with exact Shapley values. Inherits from ``ExplainingSVM`` and from `sklearn.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.

    For documentation of the ``sklearn`` methods, please take a look at the documentation there. Here, only the newly implemented methods will be discussed.
    '''
    def __init__(self, *, 
                 C: float = 1, 
                 kernel_type: Literal['rbf', 'tanimoto', 'poly', 'sigmoid'] = 'tanimoto', 
                 degree: int = 3, 
                 gamma: float = 1.0, 
                 coef0: float = 0, 
                 shrinking: bool = True, 
                 probability: bool = True, 
                 tol: float = 0.001, 
                 cache_size: float = 200, 
                 class_weight: Mapping | str | None = None, 
                 verbose: bool = False, 
                 max_iter: int = -1, 
                 decision_function_shape: Literal['ovo', 'ovr'] = "ovr", 
                 break_ties: bool = False, 
                 random_state: int | RandomState | None = None,
                 no_player_value: float = 0.0) -> None:
        '''
        Initializes SVC with exact Shapley values.

        :param C: Regularization parameter
        :type C: float
        :param kernel_type: Kernel to be used, must be in ``['tanimoto', 'rbf', 'poly', 'sigmoid']``
        :type kernel_type: str
        :param gamma: Parameter :math:`\\gamma`, only for Polynomial, RBF, Sigmoid kernel.
        :type gamma: float
        :param degree: Parameter :math:`d`, only for Polynomial kernel, must be non-negative.
        :type degree: float
        :param coef0: Parameter :math:`r`, only for Polynomial, Sigmoid kernel.
        :type coef0: float
        :param shrinking: Wheter to use shrinking heuristic
        :type shrinking: bool
        :param probability: Enables probability estimates thorugh Platt scaling
        :type probability: bool
        :param tol: Tolerance for stopping criteria
        :type tol: float
        :param cache_size: Kernel chache size
        :type cache_size: float
        :param class_weight: Set the parameter :math:`C` of class :mahtL`i` to ``class_weight[i]*C`` for SVC, if not given, all classes will have the weight of unity.
        :type class_weight: dict or 'balanced'
        :param verbose: Enables verbose output
        :type verbose: bool
        :param max_iter: Maximum number of iterations, if -1 no limit
        :type max_iter: int
        :param decision_function_shape: one-v-rest (``ovr``) or one-v-one (``ovo``)
        :type decision_function: str
        :param break_ties: If true, ``decision_function_shape='ovr'``, and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned.
        :type break_ties: bool
        :param random_state: Controls pseudo random numbers
        :type random_state: int, RandomState instnace or None
        :param no_player_value: Defines the no player value
        :type no_player_value: float
        '''
        ExplainingSVM.__init__(self,
                               kernel_type=kernel_type,
                               degree=degree,
                               gamma=gamma,
                               coef0=coef0,
                               no_player_value=no_player_value)
        SVC.__init__(self,
                     C=C, 
                     kernel=self._kernel.get_kernel_function(), 
                     degree=degree, 
                     gamma=gamma, 
                     coef0=coef0, 
                     shrinking=shrinking, 
                     probability=probability, 
                     tol=tol, 
                     cache_size=cache_size, 
                     class_weight=class_weight, 
                     verbose=verbose, 
                     max_iter=max_iter, 
                     decision_function_shape=decision_function_shape, 
                     break_ties=break_ties, 
                     random_state=random_state)
        
    def set_params(self, **params):
        '''
        Sets the parameters of the class. Necessary in order to work with sklearn Pipelines and grid searches

        :param params: parameter to change with corresponding value
        :return: Returns Class with changed parameters.
        '''
        for param, value in params.items():
            setattr(self, param, value)
            if param in ['kernel_type', 'gamma', 'degree', 'coef0', 'no_player_value']:
                # change in kernel, need to initialize from scratch, mainly to be safe
                ExplainingSVM.__init__(self,
                                       kernel_type = params.get('kernel_type', self.get_params()['kernel_type']),
                                       degree = params.get('degree', self.get_params()['degree']),
                                       gamma = params.get('gamma', self.get_params()['gamma']),
                                       coef0 = params.get('coef0', self.get_params()['coef0']),
                                       no_player_value= params.get('no_player_value', self.get_params()['no_player_value']))
                self.kernel = self._kernel.get_kernel_function()
        
        return self
    
    def fit(self, X, y, sample_weight=None):
        '''
        Fits the classifier and saves the explicit support vectors.

        :param X: Training vectors
        :type X: array_like
        :param y: Target values
        :type y: array_like
        :param sample_weights: Per-sample weight
        :type param_weights: array_like or None
        '''
        super().fit(X, y, sample_weight=sample_weight)
        self._explicit_sup_vecs = self._BaseLibSVM__Xfit[self.support_]
        return self
    
    @cached_property
    def expected_value(self):
        '''
        Expected value
        
        :return: Expected value
        :rtype: float
        '''
        return self.intercept_
    
    @cached_property
    def expected_value_logit_space(self):
        '''
        Expected value in logit space. Calcualted by scaling the the expected value using Platt scaling parameters.

        :return: Expected value in logit space
        :rtype: float
        '''
        return -self.intercept_ * self.probA_ + self.probB_
    
    def platt(self, dist):
        '''
        Platt scaling

        :math:`\\frac{1}{1-\\exp{}(-Ax+B)}`

        where :math:`A` and :math:`B` are the parameters from Platt scaling and :math:`x` is the distance from the hyperplane.

        :param dist: Distance from the hyperplane
        :type dist: float
        :return: Platt-scaled distance
        :rtype: float
        '''
        # using a minus sing in front of probB_
        # otherwise, getting incorrect probabilities
        return expit(-(dist*self.probA_ - self.probB_))

    # reimplementation because of numerical inaccuracies between
    # predict_proba(x_query) and platt(decision_function(x_query))
    @override
    def predict_proba(self, X):
        '''
        Reimplementation of ``predict_proba`` to fix numerical inconsistencies

        :param X: Input vectors
        :type X: array_like
        :return: Predicted probabilities
        :rtype: array-like
        '''
        kf = self._kernel.get_kernel_function()
        dist = (kf(X, self._explicit_sup_vecs) * self.dual_coef_).sum(axis=1)
        dist += self.intercept_
        proba1 = self.platt(dist)
        return np.vstack([1.0-proba1, proba1]).T
    
    def shapley_values(self, x: np.ndarray, logit_space=False):
        '''
        Calculates the Shapley values for the input vectors

        :param X: Input vectors
        :type X: array-like
        :param logit_space: Whether to scale the shapley values to logit space using Platt scaling parameters
        :type logit_space: bool
        :return: Shapley values for the input vectors
        :rtype: array-like
        '''
        if logit_space:
            return - super().shapley_values(x) * self.probA_
        else:
            return super().shapley_values(x)