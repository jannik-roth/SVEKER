Shapley Value-expressed Kernels (SVEKER)
================================================

This package implements exact Shapley Values for binary input data for Support Vector Machines.

Documentation
-------------

For a nicely rendered version of the Documentation please visit `https://jannik-roth.github.io/SVEKER/ <https://jannik-roth.github.io/SVEKER/>`_

Example Usage
-------------

SVEKER is built on top of `scikit-learn`'s implementaion of `SVR <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_ and `SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ and can be used as a drop-in replacement. Below are some exampler for the regression and classification use.

Regression
..........

First, we import the necessary packages and create some dummy data

.. code-block::

    import SVEKER
    import numpy as np
    from sklearn.model_selection import train_test_split

    n_feat = 50
    n_samples = 500

    x = np.random.randint(low=0, high=2, size=(n_samples, n_feat))
    y = np.random.rand(n_samples)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

Now we can buil our model and fit it

.. code-block::

    model = SVEKER.ExplainingSVR(kernel_type='tanimoto')
    model.fit(x_train, y_train)

We can obtain the predictions as usual via ``predict``

.. code-block::

    pred = model.predict(x_test)

To obtain the Shapley values we simply use ``shapley_values``

.. code-block::

    shapley_values = model.shapley_values(x_test)

To verify that everything works fine we add the Shapley values pre prediciotn and add the expected value to it

.. code-block::

    np.allclose(shapley_values.sum(axis=-1) + model.expected_value, pred)

which returns ``True``.

Classification
..............

Same as above, we import the necessary packages and create some dummy data

.. code-block::

    import SVEKER
    import numpy as np
    from sklearn.model_selection import train_test_split

    n_feat = 50
    n_samples = 500

    x = np.random.randint(low=0, high=2, size=(n_samples, n_feat))
    y = np.random.rand(n_samples)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

Next, we fit the model

.. code-block::
    
    model = SVEKER.ExplainingSVC(kernel_type='tanimoto')
    model.fit(x_train, y_train)

The Shapley values on default will be calculated in the decision function space

.. code-block::

    df = model.decision_function(x_test)
    svs = model.shapley_values(x_test).sum(axis=-1) + model.expected_value
    np.allclose(df, svs)

which returns ``True``. Alternatively, you can also specifiy the logit output space

.. code-block::

    proba = model.predict_proba(x_test)[:,1]
    logits = model.shapley_values(x_test, logit_space=True).sum(axis=-1) + model.expected_value_logit_space
    np.allclose(logits, np.log(proba/(1.-proba)))

which returns ``True``.

Citations
---------

For the original SVETA (Tanimoto kernel) implementation:

- Feldmann, C., & Bajorath, J. (2022). `Calculation of exact Shapley values for support vector machines with Tanimoto kernel enables model interpretation` iScience, 25(9), `https://doi.org/10.1016/j.isci.2022.105023 <https://doi.org/10.1016/j.isci.2022.105023>`_

For the original SVERAD (rbf kernel) implementation:

- Mastropietro, A., Feldmann, C., & Bajorath, J. (2023). `Calculation of exact Shapley values for explaining support vector machine models using the radial basis function kernel` Scientific Reports, 13(1), 19561, `https://doi.org/10.1038/s41598-023-46930-2 <https://doi.org/10.1038/s41598-023-46930-2>`_