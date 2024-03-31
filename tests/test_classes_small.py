import unittest
import SVEKER
import numpy as np

class TestSmallClassifier(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        # Generating some random data
        n_dim = 10
        n_tr = 50
        n_qu = 5
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
    def test_small_classifier_tanimoto_dec_svs(self):
        df = self.tanimoto.decision_function(self.x_query)
        svs = self.tanimoto.shapley_values(self.x_query).sum(axis=1) + self.tanimoto.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_small_classifier_tanimoto_dec_proba(self):
        df = self.tanimoto.decision_function(self.x_query)
        proba_platt = self.tanimoto.platt(df)
        proba = self.tanimoto.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_small_classifier_tanimoto_logits(self):
        svs_ls = self.tanimoto.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.tanimoto.expected_value_logit_space
        proba = self.tanimoto.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))
    
    # RBF Kernel
    def test_small_classifier_rbf_dec_svs(self):
        df = self.rbf.decision_function(self.x_query)
        svs = self.rbf.shapley_values(self.x_query).sum(axis=1) + self.rbf.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_small_classifier_rbf_dec_proba(self):
        df = self.rbf.decision_function(self.x_query)
        proba_platt = self.rbf.platt(df)
        proba = self.rbf.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_small_classifier_rbf_logits(self):
        svs_ls = self.rbf.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.rbf.expected_value_logit_space
        proba = self.rbf.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))
    
    # Polynomial Kernel
    def test_small_classifier_poly_dec_svs(self):
        df = self.poly.decision_function(self.x_query)
        svs = self.poly.shapley_values(self.x_query).sum(axis=1) + self.poly.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_small_classifier_poly_dec_proba(self):
        df = self.poly.decision_function(self.x_query)
        proba_platt = self.poly.platt(df)
        proba = self.poly.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_small_classifier_poly_logits(self):
        svs_ls = self.poly.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.poly.expected_value_logit_space
        proba = self.poly.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))
    
    # Sigmoid Kernel
    def test_small_classifier_sig_dec_svs(self):
        df = self.sig.decision_function(self.x_query)
        svs = self.sig.shapley_values(self.x_query).sum(axis=1) + self.sig.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    def test_small_classifier_sig_dec_proba(self):
        df = self.sig.decision_function(self.x_query)
        proba_platt = self.sig.platt(df)
        proba = self.sig.predict_proba(self.x_query)
        self.assertTrue(np.allclose(proba[:,1], proba_platt))
    
    def test_small_classifier_sig_logits(self):
        svs_ls = self.sig.shapley_values(self.x_query, logit_space=True).sum(axis=1) + self.sig.expected_value_logit_space
        proba = self.sig.predict_proba(self.x_query)[:,1]
        self.assertTrue(np.allclose(svs_ls, np.log(proba/(1.0-proba))))



class TestSmallRegressor(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        # Generating some random data
        n_dim = 10
        n_tr = 50
        n_qu = 5
        np.random.seed(52)

        self.x_train = np.random.randint(0,2,size=(n_tr, n_dim))
        self.y_train = np.random.randn(n_tr)

        self.x_query = np.random.randint(0,2,size=(n_qu,n_dim))
        self.y_query = np.random.randn(n_qu)

        self.tanimoto = SVEKER.ExplainingSVR(kernel_type = 'tanimoto')
        self.tanimoto.fit(self.x_train, self.y_train)

        self.rbf = SVEKER.ExplainingSVR(kernel_type = 'rbf', gamma = 1.5)
        self.rbf.fit(self.x_train, self.y_train)
        
        self.poly = SVEKER.ExplainingSVR(kernel_type = 'poly', gamma = 0.5, degree = 3, coef0 = 1.5)
        self.poly.fit(self.x_train, self.y_train)
        
        self.sig = SVEKER.ExplainingSVR(kernel_type = 'sigmoid', gamma = 0.5, coef0 = 1.5)
        self.sig.fit(self.x_train, self.y_train)

    # Tanimoto Kernel
    def test_small_regressor_tanimoto_svs(self):
        df = self.tanimoto.predict(self.x_query)
        svs = self.tanimoto.shapley_values(self.x_query).sum(axis=1) + self.tanimoto.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    # RBF Kernel
    def test_small_regressor_rbf_svs(self):
        df = self.rbf.predict(self.x_query)
        svs = self.rbf.shapley_values(self.x_query).sum(axis=1) + self.rbf.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    # Polynomial Kernel
    def test_small_regressor_poly_svs(self):
        df = self.poly.predict(self.x_query)
        svs = self.poly.shapley_values(self.x_query).sum(axis=1) + self.poly.expected_value
        self.assertTrue(np.allclose(df, svs))
    
    # Sigmoid Kernel
    def test_small_regressor_sig_svs(self):
        df = self.sig.predict(self.x_query)
        svs = self.sig.shapley_values(self.x_query).sum(axis=1) + self.sig.expected_value
        self.assertTrue(np.allclose(df, svs))

class TestNoPlayerValueRegressor(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        # Generating some random data
        n_dim = 10
        n_tr = 50
        n_qu = 5
        np.random.seed(52)

        self.x_train = np.random.randint(0,2,size=(n_tr, n_dim))
        self.y_train = np.random.randn(n_tr)

        self.x_query = np.random.randint(0,2,size=(n_qu,n_dim))
        self.y_query = np.random.randn(n_qu)

        npv1 = 0.0
        npv2 = 10.0

        self.tan1 = SVEKER.ExplainingSVR(kernel_type = 'tanimoto', no_player_value=npv1)
        self.tan1.fit(self.x_train, self.y_train)
        self.tan2 = SVEKER.ExplainingSVR(kernel_type = 'tanimoto', no_player_value=npv2)
        self.tan2.fit(self.x_train, self.y_train)
        
        self.rbf1 = SVEKER.ExplainingSVR(kernel_type = 'rbf', no_player_value=npv1)
        self.rbf1.fit(self.x_train, self.y_train)
        self.rbf2 = SVEKER.ExplainingSVR(kernel_type = 'rbf', no_player_value=npv2)
        self.rbf2.fit(self.x_train, self.y_train)

        self.poly1 = SVEKER.ExplainingSVR(kernel_type = 'poly', no_player_value=npv1)
        self.poly1.fit(self.x_train, self.y_train)
        self.poly2 = SVEKER.ExplainingSVR(kernel_type = 'poly', no_player_value=npv2)
        self.poly2.fit(self.x_train, self.y_train)
        
        self.sig1 = SVEKER.ExplainingSVR(kernel_type = 'sigmoid', no_player_value=npv1)
        self.sig1.fit(self.x_train, self.y_train)
        self.sig2 = SVEKER.ExplainingSVR(kernel_type = 'sigmoid', no_player_value=npv2)
        self.sig2.fit(self.x_train, self.y_train)


    def test_no_player_value_regressor_tanimoto(self):
        df1 = self.tan1.predict(self.x_query)
        df2 = self.tan2.predict(self.x_query)
        svs1 = self.tan1.shapley_values(self.x_query).sum(axis=1) + self.tan1.expected_value
        svs2 = self.tan2.shapley_values(self.x_query).sum(axis=1) + self.tan2.expected_value
        
        self.assertTrue(np.allclose(df1, df2))
        self.assertTrue(np.allclose(svs1, svs2))
        self.assertTrue(np.allclose(df1, svs1))
    
    def test_no_player_value_regressor_rbf(self):
        df1 = self.rbf1.predict(self.x_query)
        df2 = self.rbf2.predict(self.x_query)
        svs1 = self.rbf1.shapley_values(self.x_query).sum(axis=1) + self.rbf1.expected_value
        svs2 = self.rbf2.shapley_values(self.x_query).sum(axis=1) + self.rbf2.expected_value
        
        self.assertTrue(np.allclose(df1, df2))
        self.assertTrue(np.allclose(svs1, svs2))
        self.assertTrue(np.allclose(df1, svs1))
    
    def test_no_player_value_regressor_sig(self):
        df1 = self.sig1.predict(self.x_query)
        df2 = self.sig2.predict(self.x_query)
        svs1 = self.sig1.shapley_values(self.x_query).sum(axis=1) + self.sig1.expected_value
        svs2 = self.sig2.shapley_values(self.x_query).sum(axis=1) + self.sig2.expected_value
        
        self.assertTrue(np.allclose(df1, df2))
        self.assertTrue(np.allclose(svs1, svs2))
        self.assertTrue(np.allclose(df1, svs1))
    
    def test_no_player_value_regressor_poly(self):
        df1 = self.poly1.predict(self.x_query)
        df2 = self.poly2.predict(self.x_query)
        svs1 = self.poly1.shapley_values(self.x_query).sum(axis=1) + self.poly1.expected_value
        svs2 = self.poly2.shapley_values(self.x_query).sum(axis=1) + self.poly2.expected_value
        
        self.assertTrue(np.allclose(df1, df2))
        self.assertTrue(np.allclose(svs1, svs2))
        self.assertTrue(np.allclose(df1, svs1))

if __name__ == '__main__':
    unittest.main(verbosity=2)