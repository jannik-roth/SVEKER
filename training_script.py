import SVEKER
from STAR_protocol_utils.dataloader import MorganFPDataLoader
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import pandas as pd
import dill
import os
import yaml

def main():

    # load parameters
    with open('parameters.yaml') as params:
        args = yaml.load(params, Loader=yaml.FullLoader)
    
    TID = args['training']['TID']
    DATA_FOLDER = args['training']['DATA_FOLDER']
    N_SPLITS = args['training']['N_SPLITS']
    TRAIN_SIZE = args['training']['TRAIN_SIZE']
    RANDOM_STATE = args['training']['RANDOM_STATE']
    KERNELS = args['training']['KERNELS']

    SCORING_GRIDSEARCH = args['training']['SCORING_GRIDSEARCH']
    N_CV_GRIDSEARCH = args['training']['N_CV_GRIDSEARCH']

    PATH_TO_MODELS_FOLDER = args['training']['PATH_TO_MODELS_FOLDER']
    PATH_TO_RESULTS_FOLDER = args['training']['PATH_TO_RESULTS_FOLDER']

    FILE_NAME_PERFORMANCE_RESULTS = args['training']['FILE_NAME_PERFORMANCE_RESULTS']
    FILE_NAME_SHAPLEY_VALUES = args['training']['FILE_NAME_SHAPLEY_VALUES']

    CALC_SHAPLEY = args['training']['CALC_SHAPLEY']

    path_to_performance_results = os.path.join(PATH_TO_RESULTS_FOLDER, FILE_NAME_PERFORMANCE_RESULTS)
    path_to_shapley_values = os.path.join(PATH_TO_RESULTS_FOLDER, FILE_NAME_SHAPLEY_VALUES)

    # create folders, ignore if they already exist
    for path in [PATH_TO_MODELS_FOLDER, PATH_TO_RESULTS_FOLDER]:
        os.makedirs(path, exist_ok=True)


    # search space
    param_space = {'tanimoto': {'C': [1, 10, 100]},
                   'rbf':      {'C': [1, 10, 100],
                                'gamma': [0.001, 0.01, 0.1]},
                   'sigmoid':  {'C': [100, 1_000, 10_000],
                                'gamma': [0.001, 0.01, 0.1],
                                'coef0': [-3, -2.5, -2, -1.5]},
                   'poly':     {'C': [1e-4, 1e-3, 1e-2],
                                'gamma': [1, 10, 100],
                                'coef0': [-5.0, -4.5, -4.0, -3.5],
                                'degree': [2, 3]}}
    

    # start with training

    # obtain Data
    dataloader = MorganFPDataLoader(chembl_tid=TID,
                                    folder=DATA_FOLDER)
    x, y, (smiles, ) = dataloader.get_data()

    # initialiize splitter
    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS,
                                      train_size=TRAIN_SIZE,
                                      random_state=RANDOM_STATE)
    
    # initialize results
    performance_results = pd.DataFrame()
    shapley_results = pd.DataFrame()

    for i, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        print(f'On split: {i}')
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        _, smiles_test = smiles[train_idx], smiles[test_idx]

        for kernel in KERNELS:
            print(f'  -> on kernel: {kernel}')
            model = SVEKER.ExplainingSVC(kernel_type=kernel)
            gridsearch = GridSearchCV(estimator=model,
                                      param_grid=param_space[kernel],
                                      scoring=SCORING_GRIDSEARCH,
                                      cv=N_CV_GRIDSEARCH,
                                      refit=True)
            gridsearch.fit(x_train, y_train)

            y_predict =gridsearch.predict(x_test)

            # performance
            results = {'kernel': kernel,
                       'split': i,
                       'y_test': [y_test],
                       'y_predict': [y_predict]} | gridsearch.best_params_
            performance_results = pd.concat((performance_results, pd.DataFrame.from_dict(results)))


            if CALC_SHAPLEY:
                expected_values = gridsearch.best_estimator_.expected_value
                shapley_values = gridsearch.best_estimator_.shapley_values(x_test)
                decision_function = gridsearch.best_estimator_.decision_function(x_test)


                res_shapley = {'kernel': kernel,
                               'split': i,
                               'smiles': smiles_test,
                               'fp': [fp for fp in x_test],
                               'y_train': y_test,
                               'y_predict': y_predict,
                               'shapley_values': [sv for sv in shapley_values],
                               'expected_val': expected_values[0],
                               'decision_function': decision_function}
                shapley_results = pd.concat((shapley_results, pd.DataFrame.from_dict(res_shapley)))
                shapley_results.to_pickle(path_to_shapley_values)


            # saving
            with open(os.path.join(PATH_TO_MODELS_FOLDER, f'tid_{TID}_split_{i}_kernel_{kernel}.dill'), 'wb') as f:
                dill.dump(gridsearch.best_estimator_, f)
            performance_results.to_pickle(path_to_performance_results)

if __name__ == '__main__':
    main()
    print('Finished training')    