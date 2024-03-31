import unittest
from SVEKER.utils.utils import binom, factorial, c_fplus, c_fminus, m_coeff
from SVEKER.kernels.kernels import TanimotoKernel, RBFKernel, PolynomialKernel, SigmoidKernel
import math
import numpy as np

class TestUtils(unittest.TestCase):

    # binom
    def test_binom_valid(self):
        self.assertEqual(binom(4, 2), 6)
        self.assertEqual(binom(2, 2), 1)
        self.assertEqual(binom(10, 5), 252)
    
    def test_binom_invalid(self):
        self.assertEqual(binom(2, 3), 0)
        self.assertEqual(binom(0, 3), 0)
        self.assertEqual(binom(3, 0), 1)
    
    def test_binom_negative(self):
        self.assertTrue(math.isnan(binom(-3, 2)))

    # factorial
    def test_factorial_valid(self):
        self.assertEqual(factorial(6), 720)
        self.assertEqual(factorial(15), 1307674368000)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(0), 1)

    def test_factorial_invalid(self):
        # assume negative factorials hava a value of zero
        self.assertEqual(factorial(-4), 0)

    # c_fplus
    def test_c_fplus_valid(self):
        self.assertEqual(c_fplus(1, 2, 3, 4), 12)
        self.assertEqual(c_fplus(1, 2, 6, 4), 30)
        self.assertEqual(c_fplus(17, 3, 20, 45), 2426490)

    def test_c_fplus_invalid(self):
        self.assertEqual(c_fplus(3, 4, 1, 2), 0)
        self.assertEqual(c_fplus(-1, 2, 3, 4), 0)

    # c_fminus
    def test_c_fminus_valid(self):
        self.assertEqual(c_fminus(1, 2, 3, 4), 9)
        self.assertEqual(c_fminus(1, 2, 6, 4), 18)
        self.assertEqual(c_fminus(17, 3, 20, 45), 15098160)

    def test_c_fminus_invalid(self):
        self.assertEqual(c_fminus(3, 4, 1, 2), 0)
        self.assertEqual(c_fminus(-1, 2, 3, 4), 0)

    # m_coeff
    def test_m_coeff_valid(self):
        self.assertAlmostEqual(m_coeff(1, 2, 3, 4), 0.007142857142857143)
        self.assertAlmostEqual(m_coeff(3, 4, 7, 8), 1.9425019425019425e-05)
        self.assertAlmostEqual(m_coeff(7, 10, 12, 18), 6.423110660465666e-10)

    def test_m_coeff_invalid(self):
        self.assertAlmostEqual(m_coeff(1, 2, -3, 4), 0)

    # test vectors simple (see paper)
    def test_example_vectors_paper(self):
    # see SVERAD paper
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        # Table 1
        intersec_res = np.array([[(1, 0.25), (2, 1./12), (1, 1./12)],
                                 [(1, 1./12), (2, 1./12), (1, 0.25)]])
        for i in range(inter):
            for d in range(diff+1):
                self.assertEqual(c_fplus(i, d, inter, diff), intersec_res[i, d, 0])
                self.assertAlmostEqual(m_coeff(i, d, inter, diff), intersec_res[i, d, 1])

        # Table 2
        diff_res = np.array([[(1, 0.25), (1, 1./12)],
                             [(2, 1./12), (2, 1./12)],
                             [(1, 1./12), (1, 0.25)]])
        for i in range(inter+1):
            for d in range(diff):
                self.assertEqual(c_fminus(i, d, inter, diff), diff_res[i, d, 0])
                self.assertAlmostEqual(m_coeff(i, d, inter, diff), diff_res[i, d, 1])

class TestTanimotoKernel(unittest.TestCase):

    def test_tanimoto_kernel_vectors_paper(self):
        k = TanimotoKernel()
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        res_plus = np.array([[1, 0.5, 1./3.],
                             [0, 1./6, 1./6]])
        for i in range(inter):
            for d in range(diff+1):
                self.assertAlmostEqual(k.delta_nu_fplus(i,d), res_plus[i, d])
                                 
        res_minus = np.array([[0, 0],
                              [-0.5, -1./6],
                              [-1./3, -1./6]])
        for i in range(inter+1):
            for d in range(diff):
                self.assertAlmostEqual(k.delta_nu_fminus(i,d), res_minus[i, d])

    def test_tanimoto_kernel_vectors_paper_2(self):
        k = TanimotoKernel()
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        pl = k.phi_fplus(inter, diff)
        pm = k.phi_fminus(inter, diff)

        self.assertAlmostEqual(pl, 0.4305555555555555)
        self.assertAlmostEqual(pm, -0.1805555555555555)

        res = pl*inter + pm*diff
        self.assertAlmostEqual(res, inter/(inter + diff))

class TestRBFKernel(unittest.TestCase):
    
    def test_rbf_kernel_vectors_paper(self):
        k = RBFKernel(gamma=0.5)
        
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        res_plus = np.array([[1, 0, 0],
                             [0, 0, 0]])
        for i in range(inter):
            for d in range(diff+1):
                self.assertAlmostEqual(k.delta_nu_fplus(i,d), res_plus[i, d])
                            

        emh = np.exp(-0.5)
        emu = np.exp(-1.0)
        res_minus = np.array([[emh, emu - emh],
                              [emh - 1, emu - emh],
                              [emh - 1, emu - emh]])
        for i in range(inter+1):
            for d in range(diff):
                self.assertAlmostEqual(k.delta_nu_fminus(i,d), res_minus[i, d])

    def test_rbf_kernel_vectors_paper_2(self):
        k = RBFKernel(gamma=0.5)
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        pl = k.phi_fplus(inter, diff)
        pm = k.phi_fminus(inter, diff)

        self.assertAlmostEqual(pl, 0.25)
        self.assertAlmostEqual(pm, -0.06606027941427882)

        res = pl*inter + pm*diff
        res_num = np.exp(-0.5 * (np.sum((v1-v2)**2)))
        self.assertAlmostEqual(res, res_num)

class TestPolynomialKernel(unittest.TestCase):
    
    def test_polynomial_kernel_vectors_paper(self):
        k = PolynomialKernel(gamma=0.5, degree=3, coef0=1.5)
        
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        res_plus = np.array([[8, 4.625, 4.625],
                             [7.625, 7.625, 7.625]])
        for i in range(inter):
            for d in range(diff+1):
                self.assertAlmostEqual(k.delta_nu_fplus(i,d), res_plus[i, d])
                            

        res_minus = np.array([[3.375, 0],
                              [0, 0],
                              [0, 0]])
        for i in range(inter+1):
            for d in range(diff):
                self.assertAlmostEqual(k.delta_nu_fminus(i,d), res_minus[i, d])

    def test_rbf_kernel_vectors_paper_2(self):
        k = PolynomialKernel(gamma=0.5, degree=3, coef0=1.5)
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        pl = k.phi_fplus(inter, diff)
        pm = k.phi_fminus(inter, diff)

        self.assertAlmostEqual(pl, 6.96875)
        self.assertAlmostEqual(pm, 0.84375)

        res = pl*inter + pm*diff
        res_num = (k.gamma * np.dot(v1, v2.T) + k.coef0) ** k.degree
        self.assertAlmostEqual(res, res_num)

class TestSigmoidKernel(unittest.TestCase):
    
    def test_sigmoid_kernel_vectors_paper(self):
        k = SigmoidKernel(gamma=0.5, coef0=1.5)
        
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        res_plus = np.array([[0.9640275800758169, 0.058879326430950396, 0.058879326430950396],
                             [0.02258671807561341, 0.02258671807561341, 0.02258671807561341]])
        for i in range(inter):
            for d in range(diff+1):
                self.assertAlmostEqual(k.delta_nu_fplus(i,d), res_plus[i, d])
                            

        res_minus = np.array([[0.9051482536448665, 0],
                              [0, 0],
                              [0, 0]])
        for i in range(inter+1):
            for d in range(diff):
                self.assertAlmostEqual(k.delta_nu_fminus(i,d), res_minus[i, d])

    def test_sigmoid_kernel_vectors_paper_2(self):
        k = SigmoidKernel(gamma=0.5, coef0=1.5)
        v1 = np.array([1, 0, 0, 1, 0])
        v2 = np.array([1, 0, 1, 1, 1])
        inter = np.sum(np.dot(v1, v2))
        diff = np.sum(v1 != v2)

        pl = k.phi_fplus(inter, diff)
        pm = k.phi_fminus(inter, diff)

        self.assertAlmostEqual(pl, 0.26702008566449853)
        self.assertAlmostEqual(pm, 0.22628706341121663)

        res = pl*inter + pm*diff
        res_num = np.tanh(k.gamma * np.dot(v1, v2.T) + k.coef0)
        self.assertAlmostEqual(res, res_num)
if __name__ == '__main__':
    unittest.main(verbosity=2)