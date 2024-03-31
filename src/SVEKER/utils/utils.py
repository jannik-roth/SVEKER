#from functools import cache
import scipy.special as sps
import math
from functools import cache

@cache
def binom(x: int,
          y: int):
    return sps.binom(x, y)

@cache
def factorial(x: int):
    return math.factorial(x) if x>=0 else 0

@cache
def c_fplus(n_i: int, n_d: int, i: int, d: int):
    '''
    Calculates the :math:`C_{f^+}` factor.

    :math:`C_{f^+} = \\binom{I-1}{N_i} \\binom{D}{N_d}`

    Number of unique combinations of :math:`N_i` elements in a set of :math:`I-1` (the assessed feature is not part of the coalition) and `N_d` elements in a set of :math:`D`.

    :param n_i: Number of elements in the intersection, :math:`N_i`
    :type n_i: int
    :param n_d: Number of elemtnes in the symmetric difference, :math:`N_d`
    :type n_d: int
    :param i: Size of the intersection, :math:`I`
    :type i: int
    :param d: Size of the symmetric difference, :math:`D`
    :type d: int
    :return: :math:`C_{f^+}` factor
    :rtype: float
    '''
    return binom(i-1, n_i) * binom(d, n_d)

@cache
def c_fminus(n_i: int, n_d: int, i: int, d:int):
    '''
    Calculates the :math:`C_{f^-}` factor

    :math:`C_{f^-} = \\binom{I}{N_i} \\binom{D-1}{N_d}`    
    
    Number of unique combinations of :math:`N_i` elements in a set of :math:`I`  and `N_d` elements in a set of :math:`D-1` (the assessed feature is not part of the coalition).

    :param n_i: Number of elements in the intersection, :math:`N_i`
    :type n_i: int
    :param n_d: Number of elemtnes in the symmetric difference, :math:`N_d`
    :type n_d: int
    :param i: Size of the intersection, :math:`I`
    :type i: int
    :param d: Size of the symmetric difference, :math:`D`
    :type d: int
    :return: :math:`C_{f^-}` factor
    :rtype: float

    '''
    return binom(i, n_i) * binom(d-1, n_d)

@cache
def m_coeff(n_i: int, n_d: int, i: int, d:int):
    '''
    Calculates the multinomial coefficient :math:`m_\\mathrm{coeff}`

    :math:`m_\\mathrm{coeff} = \\frac{(N_i+N_d)!(I+D-N_i-N_d-1)!}{(I+D)!}`

    Number of permutations of the coalition :math:`(N_i+N_d)` multiplied by the number of features not contained in the coalition :math:`(I+D-N_i-N_d-1)` divided number of all possible coalitions :math:`(I+D)`
    
    :param n_i: Number of elements in the intersection, :math:`N_i`
    :type n_i: int
    :param n_d: Number of elemtnes in the symmetric difference, :math:`N_d`
    :type n_d: int
    :param i: Size of the intersection, :math:`I`
    :type i: int
    :param d: Size of the symmetric difference, :math:`D`
    :type d: int
    :return: multinomial coefficient :math:`m_\\mathrm{coeff}`
    :rtype: float
    '''
    fac1 = factorial(n_i + n_d)
    fac2 = factorial(i + d - n_i - n_d - 1)
    fac3 = factorial(i + d)
    return fac1 * fac2 / fac3