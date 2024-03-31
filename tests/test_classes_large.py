import unittest
import SVEKER
import numpy as np

class TestLargeClassifier(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # Generating some random data
        n_dim = 1024
        n_tr = 200
        n_qu = 1
        np.random.seed(52)

        self.x_train = np.random.randint(0,2,size=(n_tr, n_dim))
        self.y_train = np.random.randint(0, 2, size=(n_tr,))

        self.x_query = np.random.randint(0,2,size=(n_qu,n_dim))
        self.y_query = np.random.randint(0,2,size=(n_qu,))

        self.tanimoto = SVEKER.ExplainingSVC(kernel_type = 'tanimoto')
        self.tanimoto.fit(self.x_train, self.y_train)

        self.rbf = SVEKER.ExplainingSVC(kernel_type = 'rbf', gamma = 1.5)
        self.rbf.fit(self.x_train, self.y_train)

        self.poly = SVEKER.ExplainingSVC(kernel_type = 'poly', gamma = 0.5, degree = 3, coef0 = 1.5)
        self.poly.fit(self.x_train, self.y_train)
        
        self.sig = SVEKER.ExplainingSVC(kernel_type = 'sigmoid', gamma = 0.5, coef0 = 1.5)
        self.sig.fit(self.x_train, self.y_train)

    # Tanimoto Kernel
    def test_large_classifier_tanimoto_dec_svs(self):
        df = self.tanimoto.decision_function(self.x_query)
        svs = self.tanimoto.shapley_values(self.x_query).sum(axis=1) + self.tanimoto.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_large_classifier_tanimoto_dec_proba(self):
        df = self.tanimoto.decision_function(self.x_query)
        proba_platt = self.tanimoto.platt(df)
        proba = self.tanimoto.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_large_classifier_tanimoto_logits(self):
        svs_ls = self.tanimoto.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.tanimoto.expected_value_logit_space
        proba = self.tanimoto.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))
    
    # RBF Kernel
    def test_large_classifier_rbf_dec_svs(self):
        df = self.rbf.decision_function(self.x_query)
        svs = self.rbf.shapley_values(self.x_query).sum(axis=1) + self.rbf.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_large_classifier_rbf_dec_proba(self):
        df = self.rbf.decision_function(self.x_query)
        proba_platt = self.rbf.platt(df)
        proba = self.rbf.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_large_classifier_rbf_logits(self):
        svs_ls = self.rbf.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.rbf.expected_value_logit_space
        proba = self.rbf.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))
    
    # Polynomial Kernel
    def test_large_classifier_poly_dec_svs(self):
        df = self.poly.decision_function(self.x_query)
        svs = self.poly.shapley_values(self.x_query).sum(axis=1) + self.poly.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_large_classifier_poly_dec_proba(self):
        df = self.poly.decision_function(self.x_query)
        proba_platt = self.poly.platt(df)
        proba = self.poly.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_large_classifier_poly_logits(self):
        svs_ls = self.poly.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.poly.expected_value_logit_space
        proba = self.poly.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))
    
    # Sigmoid Kernel
    def test_large_classifier_sig_dec_svs(self):
        df = self.sig.decision_function(self.x_query)
        svs = self.sig.shapley_values(self.x_query).sum(axis=1) + self.sig.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_large_classifier_sig_dec_proba(self):
        df = self.sig.decision_function(self.x_query)
        proba_platt = self.sig.platt(df)
        proba = self.sig.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_large_classifier_sig_logits(self):
        svs_ls = self.sig.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.sig.expected_value_logit_space
        proba = self.sig.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))



class TestLargeRegressor(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        # Generating some random data
        n_dim = 1024
        n_tr = 200
        n_qu = 1
        np.random.seed(52)

        self.x_train = np.random.randint(0,2,size=(n_tr, n_dim))
        self.y_train = np.random.randn(n_tr)

        self.x_query = np.random.randint(0,2,size=(n_qu,n_dim))
        self.y_query = np.random.randn(n_qu)

        self.tanimoto = SVEKER.ExplainingSVR(kernel_type = 'tanimoto')
        self.rbf = SVEKER.ExplainingSVR(kernel_type = 'rbf', gamma = 1.5)
        self.poly = SVEKER.ExplainingSVR(kernel_type = 'poly', gamma = 0.5, degree = 3, coef0 = 1.5)
        self.sig = SVEKER.ExplainingSVR(kernel_type = 'sigmoid', gamma = 0.5, coef0 = 1.5)

    # Tanimoto Kernel
    def test_large_regressor_tanimoto_svs(self):
        self.tanimoto.fit(self.x_train, self.y_train)
        df = self.tanimoto.predict(self.x_query)
        svs = self.tanimoto.shapley_values(self.x_query).sum(axis=1) + self.tanimoto.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    # RBF Kernel
    def test_large_regressor_rbf_svs(self):
        self.rbf.fit(self.x_train, self.y_train)
        df = self.rbf.predict(self.x_query)
        svs = self.rbf.shapley_values(self.x_query).sum(axis=1) + self.rbf.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    # Polynomial Kernel
    def test_large_regressor_poly_svs(self):
        self.poly.fit(self.x_train, self.y_train)
        df = self.poly.predict(self.x_query)
        svs = self.poly.shapley_values(self.x_query).sum(axis=1) + self.poly.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    # Sigmoid Kernel
    def test_large_regressor_sig_svs(self):
        self.sig.fit(self.x_train, self.y_train)
        df = self.sig.predict(self.x_query)
        svs = self.sig.shapley_values(self.x_query).sum(axis=1) + self.sig.expected_value
        self.assertTrue(np.allclose(df, svs))

if __name__ == '__main__':
    # takes roughly 10 mins to run
    unittest.main(verbosity=2)