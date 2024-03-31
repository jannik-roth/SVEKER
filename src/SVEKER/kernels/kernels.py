from ..utils import m_coeff, c_fminus, c_fplus, factorial, binom
import abc
from typing_extensions import override
from typing import Callable
import numpy as np
from functools import cache

class BaseKernel(abc.ABC):
    '''
    Base Class for different kernels to be used.

    Defines the necessary functions for the different kernels to be used.
    '''
    def __init__(self, no_player_value) -> None:
        '''
        Initializes the Base Kernel.

        :param no_player_value: Kernel value associated to the empty coalition :math:`\\nu(\\varnothing)` 
        '''
        super().__init__()
        self.no_player_value = no_player_value
    @abc.abstractmethod
    def delta_nu_fplus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the kernel upon addition of a feature to the intersection, :math:`\\Delta\\nu_{f^+}`.
        Kernel-specific, needs to be implemented for the specific kernel.

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elemtnes in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^+}`
        :rtype: float
        '''
        pass

    @abc.abstractmethod
    def delta_nu_fminus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the kernel upon addition of a feature to the symmetric difference, :math:`\\Delta\\nu_{f^-}`.
        Kernel-specific, needs to be implemented for the specific kernel.

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elemtnes in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^-}`
        :rtype: float
        '''
        pass

    @abc.abstractmethod
    def get_kernel_function(self) -> Callable:
        '''
        Returns the Kernel function to be used with `sklearn.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ or `sklearn.SVR <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_. See ``kernel = 'precomputed'`` in the `sklearn` documentaion for details.
        '''
        pass

    @cache
    def phi_fplus(self, i: int, d: int) -> float:
        '''
        Naïve implementation of Shapley value calculation for adding a feature to the intersection.

        :math:`\\phi_{f^+} (I, D) = \\sum_{N_i = 0}^{I-1} \\sum_{N_d = 0}^{D} \\Delta\\nu_{f^+}(N_i, N_d) \, C_{f^+} (N_i, N_d, I, D) \, m_\\mathrm{coeff}(N_i, N_d, I, D)`

        This summation is valid for all kernels but it can be simplified using the properties of the :math:`\\Delta\\nu_{f^+}(N_i, N_d)` term.

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the intersection, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        res = 0.0
        for n_i in range(i):
            for n_d in range(d+1):
                res += self.delta_nu_fplus(n_i, n_d) * c_fplus(n_i, n_d, i, d) * m_coeff(n_i, n_d, i, d)
        return res

    @cache
    def phi_fminus(self, i: int, d: int) -> float:        
        '''
        Naïve implementation of Shapley value calculation for adding a feature to the symmetric difference.

        :math:`\\phi_{f^-} (I, D) = \\sum_{N_i = 0}^{I} \\sum_{N_d = 0}^{D-1} \\Delta\\nu_{f^-}(N_i, N_d) \, C_{f^-} (N_i, N_d, I, D) \, m_\\mathrm{coeff}(N_i, N_d, I, D)`

        This summation is valid for all kernels but it can be simplified using the properties of the :math:`\\Delta\\nu_{f^-}(N_i, N_d)` term.

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the intersection, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        res = 0.0
        for n_i in range(i+1):
            for n_d in range(d):
                res += self.delta_nu_fminus(n_i, n_d) * c_fminus(n_i, n_d, i, d) * m_coeff(n_i, n_d, i, d)
        return res
    
class TanimotoKernel(BaseKernel):    
    '''
    Implements the Tanimoto Kernel

    :math:`K(I, D) = \\frac{I}{I+D}`
    '''
    def __init__(self, no_player_value = 0.0) -> None:
        '''
        Initializes the Tanimoto Kernel.
        '''
        super().__init__(no_player_value=no_player_value)
    
    @cache
    @override
    def delta_nu_fplus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the Tanimoto Kernel upon adding a feature to the intersection.

        .. math::

            \\Delta\\nu_{f^+} (N_i, N_d) = \\begin{cases}
            1 - \\nu(\\varnothing) & \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
            \\frac{1}{N_d + 1} & \\text{if } N_i = 0 \\text{ and } N_d \\neq 0 \\\\
            \\frac{N_i + 1}{N_i + N_d + 1} - \\frac{N_i}{N_i + N_d} & \\text{else}
            \\end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elements in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^+}`
        :rtype: float
        '''
        if n_i == 0 and n_d == 0:
            return 1. - self.no_player_value
        elif n_i == 0:
            return 1. / (n_d + 1.)
        else:
            frac1 = (n_i + 1) / (n_i + n_d + 1)
            frac2 = n_i / (n_i + n_d)
            return frac1 - frac2
    
    @cache
    @override
    def delta_nu_fminus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the Tanimoto Kernel upon adding a feature to the symmetric difference.

        .. math::

                \\Delta\\nu_{f^-} (N_i, N_d) = \\begin{cases}
                -\\nu(\\varnothing) & \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
                0 & \\text{if } N_i = 0 \\text{ and } N_d \\neq 0 \\\\
                \\frac{N_i}{N_i + N_d + 1} - \\frac{N_i}{N_i + N_d} & \\text{else}
                \\end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elements in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^-}`
        :rtype: float
        '''
        if n_i == 0 and n_d == 0:
            return -self.no_player_value
        elif n_i == 0:
            return 0
        else:
            frac1 = n_i / (n_i + n_d + 1)
            frac2 = n_i / (n_i + n_d)
            return frac1 - frac2
    
    @override
    def get_kernel_function(self) -> Callable:
        def tanimoto_kernel(matrix_a: np.ndarray, matrix_b: np.ndarray):
            intersection = matrix_a.dot(matrix_b.transpose())
            norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
            norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
            union = np.add.outer(norm_1, norm_2.T) - intersection
            # to avoid divison by zero problem
            # if union == 0 then also intersection == 0
            union[union == 0] = 1
            return intersection / union
        return tanimoto_kernel
    
class RBFKernel(BaseKernel):    
    '''
    Implements the RBF Kernel

    :math:`K(I, D) = \\exp{}\\left\\{ -\\gamma D\\right\\}`
    '''
    def __init__(self, no_player_value = 0.0, gamma = 1.0) -> None:
        '''
        Initializes the RBF Kernel.

        :param gamma: Parameter :math:`\\gamma` for the RBF kernel.
        :type gamma: float
        '''
        super().__init__(no_player_value=no_player_value)
        self.gamma = gamma

    @cache
    @override
    def delta_nu_fplus(self, n_i: int, n_d: int) -> float:      
        '''
        Calculates the change in the RBF Kernel upon adding a feature to the intersection.

        .. math::

            \\Delta\\nu_{f^+} (N_i, N_d) = 
            \\begin{cases}
            1 - \\nu(\\varnothing) & \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
            0 & \\text{else}
            \end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elements in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^+}`
        :rtype: float
        '''
        if n_i == n_d and n_i == 0:
            return 1. - self.no_player_value
        else:
            return 0.
    
    @cache
    @override
    def delta_nu_fminus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the RBF Kernel upon adding a feature to the symmetric difference.

        .. math::

            \\Delta\\nu_{f^-} (N_i, N_d) = 
            \\begin{cases}
            \\exp{}\\left\\{-\gamma (N_d + 1)\\right\\} - \\nu(\\varnothing) & \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
            \\exp{}\\left\\{-\\gamma (N_d + 1)\\right\\} - \\exp{}\\left\\{-\\gamma N_d\\right\\} & \\text{else}
            \\end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elemtnes in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^-}`
        :rtype: float
        '''
        if (n_d == 0) and (n_i == 0):
            return np.exp(-self.gamma*(n_d + 1)) - self.no_player_value
        else:
            return np.exp(-self.gamma*(n_d + 1)) - np.exp(-self.gamma*n_d)
    
    @cache
    @override
    def phi_fplus(self, i: int, d: int) -> float:        
        '''
        Simplified summation for the Shapley value calculation for adding a feature to the intersection. Only valid for RBF Kernel.

        :math:`\\phi_{f^+} (I, D) = \\frac{1-\\nu(\\varnothing)}{I + D}`

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the intersection, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        return (1.-self.no_player_value)/(i+d)
    
    @cache
    @override
    def phi_fminus(self, i: int, d: int) -> float:        
        '''
        Simplified summation for the Shapley value calculation for adding a feature to the symmetric difference. Only valid for RBF Kernel.

        .. math::
                \\begin{split}
                \\phi_{f^-} (I, D) =& \\frac{\\exp{}\\left\\{-\\gamma\\right\\} - \\nu(\\varnothing)}{I+D} \\\\
                       & + \\left( \\exp{}\\left\\{ -\\gamma\\right\\} -1\\right) \\sum_{N_i=1}^I \\binom{I}{N_i} \\, m_\\mathrm{coeff}(N_i, 0, I, D) \\\\
                       & + \\sum_{N_i = 0}^{I} \\sum_{N_d = 1}^{D-1} \\Delta\\nu_{f^-}(N_i, N_d) \\, C_{f^-} (N_i, N_d, I, D) \\, m_\\mathrm{coeff}(N_i, N_d, I, D)  
                \\end{split}

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the symmetric difference, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        sum1 = (np.exp(-self.gamma) - self.no_player_value)/ (i + d)
        
        sum2 = 0.0
        for n_i in range(1, i + 1):
            sum2 += binom(i, n_i) * m_coeff(n_i, 0, i, d)
        sum2 *= (np.exp(-self.gamma) - 1.)

        sum3 = 0.0
        for n_i in range(i + 1):
            for n_d in range(1, d):
                sum3 += self.delta_nu_fminus(n_i, n_d) * c_fminus(n_i, n_d, i, d) * m_coeff(n_i, n_d, i, d)
        return sum1 + sum2 + sum3
    
    @override
    def get_kernel_function(self) -> Callable:
        def rbf_kernel(matrix_a: np.ndarray, matrix_b: np.ndarray):
            norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
            norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
            distance_squared = np.add.outer(norm_1, norm_2.T) - 2 * matrix_a.dot(matrix_b.transpose())
            return np.exp(-self.gamma * distance_squared)
        return rbf_kernel
    
class PolynomialKernel(BaseKernel):    
    '''
    Implements the Polynomial kernel

    :math:`K(I, D) = (\\gamma I + r)^d`
    '''
    def __init__(self, no_player_value = 0.0, gamma = 1.0, degree = 3, coef0=0.0) -> None:
        '''
        Initializes the Polynomial Kernel.

        :param gamma: Parameter :math:`\\gamma` for the Polynomial kernel.
        :type gamma: float
        :param degree: Parameter :math:`d` for the Polynomial kernel, must be non-negative.
        :type degree: float
        :param coef0: Parameter :math:`r` for the Polynomial kernel.
        :type coef0: float
        '''
        super().__init__(no_player_value=no_player_value)
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    @cache
    @override
    def delta_nu_fplus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the Polynomial kernel upon adding a feature to the intersection.

        .. math::

            \\Delta\\nu_{f^+} (N_i, N_d) = \\begin{cases}
            (\\gamma (N_i + 1) + r)^d - \\nu(\\varnothing) & \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
            (\\gamma (N_i + 1) + r)^d - (\gamma N_i + r)^d & \\text{else}
            \\end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elements in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^+}`
        :rtype: float
        '''
        if n_i == 0 and n_d == 0:
            return (self.gamma * (n_i + 1) + self.coef0) ** self.degree - self.no_player_value
        else:
            tmp1 = (self.gamma * (n_i + 1) + self.coef0) ** self.degree
            tmp2 = (self.gamma * (n_i) + self.coef0) ** self.degree
            return tmp1 - tmp2
    
    @cache
    @override
    def delta_nu_fminus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the Polynomial kernel upon adding a feature to the symmetric difference.

        .. math::

            \\Delta\\nu_{f^-} (N_i, N_d) = \\begin{cases}
            r^d - \\nu(\\varnothing)& \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
            0 & \\text{else}
            \\end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elements in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^-}`
        :rtype: float
        '''
        if n_i == 0 and n_d == 0:
            return self.coef0 ** self.degree - self.no_player_value
        else:
            return 0
    
    @cache
    @override
    def phi_fminus(self, i: int, d: int) -> float:
        '''
        Simplified summation for the Shapley value calculation for adding a feature to the symmetric difference. Only valid for Polynomial kernel.

        :math:`\\phi_{f^-} (I, D) = \\frac{r^d - \\nu(\\varnothing)}{I + D}`

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the symmetric difference, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        return (self.coef0 ** self.degree - self.no_player_value) /(i + d)

    @cache
    @override
    def phi_fplus(self, i: int, d: int) -> float:
        '''
        Simplified summation for the Shapley value calculation for adding a feature to the intersection. Only valid for Polynomial kernel.

        .. math::
                \\begin{split}
                \\phi_{f^+} (I, D) =& \\frac{(\\gamma + r)^d - \\nu(\\varnothing)}{I + D} \\\\
                & + \\Delta\\nu_{f^+}(0, 1) \\sum_{N_d=1}^D \\binom{D}{N_d} \\, m_\\mathrm{coeff}(0, N_d, I, D) \\\\
                & + \\sum_{N_i = 1}^{I-1} \\Delta\\nu_{f^+}(N_i, 0) \\sum_{N_d = 0}^{D}  \\, C_{f^+} (N_i, N_d, I, D) \\, m_\\mathrm{coeff}(N_i, N_d, I, D)  
                \\end{split}

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the intersection, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        sum1 = ((self.gamma + self.coef0) ** self.degree - self.no_player_value) / (i + d)

        sum2 = 0.0
        for n_d in range(1, d + 1):
            sum2 += binom(d, n_d) * m_coeff(0, n_d, i, d)
        sum2 *= self.delta_nu_fplus(0, 1)

        sum3 = 0.0
        for n_i in range(1, i):
            nu_fac = self.delta_nu_fplus(n_i, 0)
            for n_d in range(0, d + 1):
                sum3 += nu_fac * c_fplus(n_i, n_d, i, d) * m_coeff(n_i, n_d, i, d)
        
        return sum1 + sum2 + sum3

    @override
    def get_kernel_function(self) -> Callable:
        def polynomial_kernel(matrix_a : np.ndarray, matrix_b: np.ndarray):
            dot = np.dot(matrix_a, matrix_b.T)
            return (self.gamma * dot + self.coef0) ** self.degree
        return polynomial_kernel
    
class SigmoidKernel(BaseKernel):
    '''
    Implements the Sigmoid kernel

    :math:`K(I, D) = \\tanh(\\gamma I + r)`
    '''
    def __init__(self, no_player_value = 0.0, gamma = 1.0, coef0 = 0.0) -> None:
        '''
        Initializes the Sigmoid Kernel.

        :param gamma: Parameter :math:`\\gamma` for the Sigmoid kernel.
        :type gamma: float
        :param coef0: Parameter :math:`r` for the Sigmoid kernel.
        :type coef0: float
        '''
        super().__init__(no_player_value=no_player_value)
        self.gamma = gamma
        self.coef0 = coef0
    
    @cache
    @override
    def delta_nu_fplus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the Sigmoid kernel upon adding a feature to the intersection.

        .. math::

            \\Delta\\nu_{f^+} (N_i, N_d) = \\begin{cases}
            \\tanh{}(\\gamma (N_i + 1) + r) - \\nu(\\varnothing)& \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
            \\tanh{}(\\gamma (N_i + 1) + r) - \\tanh{}(\gamma N_i + r) & \\text{else}
            \\end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elements in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^+}`
        :rtype: float
        '''
        if n_i == 0 and n_d == 0:
            return np.tanh(self.gamma * (n_i + 1) + self.coef0) - self.no_player_value
        else:
            tmp1 = np.tanh(self.gamma * (n_i + 1) + self.coef0)
            tmp2 = np.tanh(self.gamma * (n_i) + self.coef0)
            return tmp1 - tmp2    
    
    @cache
    @override
    def delta_nu_fminus(self, n_i: int, n_d: int) -> float:
        '''
        Calculates the change in the Sigmoid kernel upon adding a feature to the symmetric difference.

        .. math::

            \\Delta\\nu_{f^-} (N_i, N_d) = \\begin{cases}
            \\tanh{}(r) - \\nu(\\varnothing) & \\text{if } N_i = 0 \\text{ and } N_d = 0 \\\\
            0 & \\text{else}
            \\end{cases}

        :param n_i: Number of elements in the intersection, :math:`N_i`
        :type n_i: int
        :param n_d: Number of elemtnes in the symmetric difference, :math:`N_d`
        :type n_d: int
        :return: Change in kernel, :math:`\\Delta\\nu_{f^-}`
        :rtype: float
        '''
        if n_i == 0 and n_d == 0:
            return np.tanh(self.coef0) - self.no_player_value
        else:
            return 0
    
    @cache
    @override
    def phi_fminus(self, i: int, d: int) -> float:
        '''
        Simplified summation for the Shapley value calculation for adding a feature to the symmetric difference. Only valid for Sigmoid kernel.

        :math:`\\phi_{f^-} (I, D) = \\frac{\\tanh{}(r) - \\nu(\\varnothing)}{I + D}`

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the symmetric difference, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        return (np.tanh(self.coef0) - self.no_player_value) /(i + d)

    @cache
    @override
    def phi_fplus(self, i: int, d: int) -> float:
        '''
        Simplified summation for the Shapley value calculation for adding a feature to the intersection. Only valid for Sigmoid kernel.

        .. math::
                \\begin{split}
                \\phi_{f^+} (I, D) = & \\frac{\\tanh{}(\\gamma + r) - \\nu(\\varnothing)}{I + D} \\\\
                & + \\Delta\\nu_{f^+}(0, 1) \\sum_{N_d=1}^D \\binom{D}{N_d} \\, m_\\mathrm{coeff}(0, N_d, I, D) \\\\
                & + \\sum_{N_i = 1}^{I-1} \\Delta\\nu_{f^+}(N_i, 0) \\sum_{N_d = 0}^{D}  \\, C_{f^+} (N_i, N_d, I, D) \\, m_\\mathrm{coeff}(N_i, N_d, I, D)
                \\end{split}

        :param i: Size of intersection, :math:`I`
        :type i: int
        :param d: Size of the symmetric difference, :math:`D`
        :type d: int
        :return: Shapley value for adding a feature to the intersection, :math:`\\phi_{f^+}`
        :rtype: float
        '''
        sum1 = (np.tanh(self.gamma + self.coef0) - self.no_player_value) / (i + d)

        sum2 = 0.0
        for n_d in range(1, d + 1):
            sum2 += binom(d, n_d) * m_coeff(0, n_d, i, d)
        sum2 *= self.delta_nu_fplus(0, 1)

        sum3 = 0.0
        for n_i in range(1, i):
            nu_fac = self.delta_nu_fplus(n_i, 0)
            for n_d in range(0, d + 1):
                sum3 += nu_fac * c_fplus(n_i, n_d, i, d) * m_coeff(n_i, n_d, i, d)
        
        return sum1 + sum2 + sum3
    
    @override
    def get_kernel_function(self) -> Callable:
        def sigmoid_kernel(matrix_a: np.ndarray, matrix_b: np.ndarray):
            dot = np.dot(matrix_a, matrix_b.T)
            return np.tanh(self.gamma*dot + self.coef0)
        return sigmoid_kernel