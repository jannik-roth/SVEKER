import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import os
import rdkit.Chem as Chem
import yaml_

from STAR_protocol_utils.mappings import get_ecfp4_bit_info, shap_to_atom_weight, get_atom_wise_weight_map

def get_scoring_function(name):
    # see https://stackoverflow.com/a/75284184
    # and here: https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/metrics/_scorer.py#L376
    scorer = metrics.get_scorer(name)
    func = lambda y_true, y_pred: scorer._sign * scorer._score_func(y_true=y_true,
                                                                    y_pred=y_pred,
                                                                    **scorer._kwargs)
    return func

def save_performance_plots(perf_df,
                           metric_names,
                           path_to_performance_figure_folder,
                           metric_name_nice):
    performance_df = perf_df.copy()
    for metric_name in metric_names:
        score_func = get_scoring_function(metric_name)
        performance_df[metric_name] = performance_df.apply(lambda x: score_func(x.y_test, x.y_predict), axis=1) # CHANGE LATER
    
        fig, axs = plt.subplots(figsize=(8,7))
        sns.boxplot(performance_df, x='kernel', y=metric_name, hue='kernel', legend=False)
        axs.set_xlabel('Kernel')
        axs.set_ylabel('')
        axs.set_title(metric_name_nice.get(metric_name, metric_name))

        fig.tight_layout()
        fig.savefig(os.path.join(path_to_performance_figure_folder, f'{metric_name}.png'), dpi=300)
        plt.close()

def save_correlation_df(shapley_df, path_to_correlation_results, correlation='pearson'):
    from scipy.stats import pearsonr, spearmanr
    if correlation.lower() == 'pearson':
        corr = pearsonr
        corr_name = 'pearson'
    elif correlation.lower() == 'spearman':
        corr = spearmanr
        corr_name = 'spearman'
    else:
        raise KeyError(f'I do NOT know the correlation: {corr}')

    rows_list = []

    for split in shapley_df.split.unique():
        df_split = shapley_df.query('split == @split')
        for smiles in df_split.smiles.unique():
            df_smiles = df_split.query('smiles == @smiles')
            for kernel_1 in df_smiles.kernel.unique():
                for kernel_2 in df_smiles.kernel.unique():
                    sv1 = df_smiles.query('kernel == @kernel_1').iloc[0].shapley_values
                    sv2 = df_smiles.query('kernel == @kernel_2').iloc[0].shapley_values
                    fp = df_smiles.fp.iloc[0]
                    correl = corr(sv1, sv2).statistic
                    correl_pres = corr(sv1[fp == 1], sv2[fp == 1]).statistic
                    correl_abs = corr(sv1[fp == 0], sv2[fp == 0]).statistic
                    results = {'split': split,
                               'smiles': smiles,
                               'kernel_1': kernel_1,
                               'kernel_2': kernel_2,
                               f'{corr_name}_all': correl,
                               f'{corr_name}_pres': correl_pres,
                               f'{corr_name}_abs': correl_abs}
                    rows_list.append(results)

    corr_results = pd.DataFrame(rows_list)
    corr_results.to_pickle(path_to_correlation_results) 

def save_correlation_plots(corr_results, path_to_correlation_figure_folder, types=['all'], correlation='pearson'):
    if correlation.lower() == 'pearson':
        corr_nice = "Pearson's $\\rho$"
    elif correlation.lower() == 'spearman':
        corr_nice = "Spearman's $\\rho$"
    else:
        raise KeyError(f"I do NOT know the correlation: {correlation}")
    
    for type in types:
        fig, axs = plt.subplots(figsize=(8,7))
        sns.boxplot(corr_results,
                    x='kernel_1',
                    y=f'{correlation}_{type}',
                    hue='kernel_2',
                    ax=axs)
        axs.set_ylabel(corr_nice)
        axs.set_xlabel("Kernel 1")
        sns.move_legend(axs, 'center left', bbox_to_anchor=(1.0, 0.5), title='Kernel 2')
    
        fig.tight_layout()
        fig.savefig(os.path.join(path_to_correlation_figure_folder, f'{correlation}_{type}.png'), dpi=300)
        plt.close()

def draw_mol(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)

    # Start by getting a PNG of the molecular drawing:
    d2d = Draw.MolDraw2DCairo(500, 350)
    # make the background transparent
    d2d.drawOptions().setBackgroundColour((1,1,1,0))
    d2d.drawOptions().useBWAtomPalette()
    d2d.DrawMolecule(mol)
    img = Draw._drawerToImage(d2d)
    return img

def create_scatter(df):

    svs = np.stack(df.shapley_values)
    kernels = np.array(df.kernel)
    fp = df.iloc[0].fp

    rename_dict={'TAN': 'SV$_{\mathrm{TAN}}$', 'RBF': 'SV$_{\mathrm{RBF}}$', 'POL': 'SV$_{\mathrm{POL}}$', 'SIG': 'SV$_{\mathrm{SIG}}$'}
    features_palette = {'Absent': sns.color_palette('muted')[0],
                        'Present': sns.color_palette('muted')[1]}

    df = pd.DataFrame(np.vstack((svs, fp)).T)
    df = df.rename(columns={i: kernel for i, kernel in enumerate(kernels)} | {len(kernels): 'bit'})
    df['Features'] = df.bit.apply(lambda x: 'Absent' if x==0.0 else 'Present')
    df.drop(columns='bit', inplace=True)

    g = sns.PairGrid(df.rename(columns=rename_dict),
                        hue='Features',
                        corner=True,
                        diag_sharey=False,
                        palette=features_palette)
    g.map_diag(sns.kdeplot, common_norm=False)
    g.map_lower(sns.scatterplot)
    g.add_legend()

    return g

def save_scatter_plots(df_shapley,
                        path_to_scatter_figure_folder,
                        num_cpds_active_per_split=1,
                        num_cpds_inactive_per_split=1,
                        save_mols=True):

    for split in df_shapley.split.unique():
        kernels = df_shapley.kernel.unique()
        df_correct_split = df_shapley.query('split == @split').query('y_train == y_predict')
        df_correct_all_kernel_split = df_correct_split.groupby('smiles').filter(lambda x: len(x) == len(kernels))

        for act in range(num_cpds_active_per_split):
            smiles = df_correct_all_kernel_split.query('y_train == 1').iloc[act].smiles
            tmp_df = df_correct_all_kernel_split.query('smiles == @smiles')
            g = create_scatter(tmp_df)
            g.tight_layout()
            g.savefig(os.path.join(path_to_scatter_figure_folder, f'split_{split}_scatter_{smiles}_active_{act}.png'), dpi=300)
            plt.close()
            if save_mols:
                mol_img = draw_mol(smiles)
                mol_img.save(os.path.join(path_to_scatter_figure_folder, f'split_{split}_mol_{smiles}_active_{act}.png'))


        for inact in range(num_cpds_inactive_per_split):
            smiles = df_correct_all_kernel_split.query('y_train == 0').iloc[act].smiles
            tmp_df = df_correct_all_kernel_split.query('smiles == @smiles')
            g = create_scatter(tmp_df)
            g.tight_layout()
            g.savefig(os.path.join(path_to_scatter_figure_folder, f'split_{split}_scatter_{smiles}_inactive_{inact}.png'), dpi=300)
            plt.close()

            if save_mols:
                mol_img = draw_mol(smiles)
                mol_img.save(os.path.join(path_to_scatter_figure_folder, f'split_{split}_mol_{smiles}_active_{inact}.png'))

def create_mappings(smiles, mol_df, path_to_mappings_figure_folder, activity_descr, split, id):
    mol = Chem.MolFromSmiles(smiles)

    for kernel in mol_df.kernel.unique():
        sv = mol_df.query('kernel == @kernel').shapley_values.iloc[0]
        bit_info = get_ecfp4_bit_info(smiles, n_bits=sv.shape[0])
        shap_mapping = shap_to_atom_weight(mol, bit_info, sv)
        png = get_atom_wise_weight_map(mol, shap_mapping, mol_size=(500,360), return_png=True)
        png.save(os.path.join(path_to_mappings_figure_folder, f'split_{split}_mol_mappings_{smiles}_{kernel}_{activity_descr}_{id}.png'))

def save_mappings(df_shapley, num_cpds_mappings_active_per_split, num_cpds_mappings_inactive_per_split, path_to_mappings_figure_folder):

    for split in df_shapley.split.unique():
        kernels = df_shapley.kernel.unique()
        df_correct_split = df_shapley.query('split == @split').query('y_train == y_predict')
        df_correct_all_kernel_split = df_correct_split.groupby('smiles').filter(lambda x: len(x) == len(kernels))

        # first the active compounds
        for act in range(num_cpds_mappings_active_per_split):
            smiles = df_correct_all_kernel_split.query('y_train == 1').iloc[act].smiles
            tmp_df = df_correct_all_kernel_split.query('smiles == @smiles')
            create_mappings(smiles=smiles,
                            mol_df=tmp_df,
                            path_to_mappings_figure_folder=path_to_mappings_figure_folder,
                            activity_descr='active',
                            split=split,
                            id=act)

        for inact in range(num_cpds_mappings_inactive_per_split):
            smiles = df_correct_all_kernel_split.query('y_train == 0').iloc[inact].smiles
            tmp_df = df_correct_all_kernel_split.query('smiles == @smiles')
            create_mappings(smiles=smiles,
                            mol_df=tmp_df,
                            path_to_mappings_figure_folder=path_to_mappings_figure_folder,
                            activity_descr='inactive',
                            split=split,
                            id=inact)


def main():

    with open('parameters.yaml') as params:
        args = yaml.load(params, Loader=yaml.FullLoader)

    PATH_TO_PERFORMANCE_RESULTS = args['analysis']['PATH_TO_PERFORMANCE_RESULTS']
    PATH_TO_SHAPLEY_RESULTS =  args['analysis']['PATH_TO_SHAPLEY_RESULTS']
    PATH_TO_CORRELATION_RESULTS = args['analysis']['PATH_TO_CORRELATION_RESULTS']
    CORRELATION_STATISTIC = args['analysis']['CORRELATION_STATISTIC']
    PATH_TO_PERFORMANCE_FIGURE_FOLDER = args['analysis']['PATH_TO_PERFORMANCE_FIGURE_FOLDER']
    PATH_TO_CORRELATION_FIGURE_FOLDER = args['analysis']['PATH_TO_CORRELATION_FIGURE_FOLDER']
    PATH_TO_SCATTER_FIGURE_FOLDER = args['analysis']['PATH_TO_SCATTER_FIGURE_FOLDER']
    PATH_TO_MAPPINGS_FIGURE_FOLDER = args['analysis']['PATH_TO_MAPPINGS_FIGURE_FOLDER']

    METRIC_NAMES = args['analysis']['METRIC_NAMES']
    SAVE_PERFORMANCE_PLOTS = args['analysis']['SAVE_PERFORMANCE_PLOTS']
    CALCULATE_CORRELATION_STATISTICS = args['analysis']['CALCULATE_CORRELATION_STATISTICS']
    SAVE_CORRELATION_FIGURES = args['analysis']['SAVE_CORRELATION_FIGURES']
    SAVE_SCATTER_PLOTS = args['analysis']['SAVE_SCATTER_PLOTS']
    NUM_CPDS_SCATTER_ACTIVE_PER_SPLIT = args['analysis']['NUM_CPDS_SCATTER_ACTIVE_PER_SPLIT']
    NUM_CPDS_SCATTER_INACTIVE_PER_SPLIT = args['analysis']['NUM_CPDS_SCATTER_INACTIVE_PER_SPLIT']
    SAVE_SCATTER_MOLS = args['analysis']['SAVE_SCATTER_MOLS']
    NUM_CPDS_MAPPINGS_ACTIVE_PER_SPLIT = args['analysis']['NUM_CPDS_MAPPINGS_ACTIVE_PER_SPLIT']
    NUM_CPDS_MAPPINGS_INACTIVE_PER_SPLIT = args['analysis']['NUM_CPDS_MAPPINGS_INACTIVE_PER_SPLIT']
    SAVE_MAPPINGS = args['analysis']['SAVE_MAPPINGS']
        
    metric_name_nice = {'accuracy': 'Accuracy',
                        'matthews_corrcoef': "Matthew's Correlation Coefficient $\phi$",
                        'f1': 'F1 Score'}
    kernel_name_nice = {'tanimoto': 'TAN',
                        'rbf': 'RBF',
                        'sigmoid': 'SIG',
                        'poly': 'POL'}

    sns.set_context('talk', font_scale=1)
    sns.set_palette('Set2')

    # create folders, ignore if they do exist
    for path in [PATH_TO_PERFORMANCE_FIGURE_FOLDER, PATH_TO_CORRELATION_FIGURE_FOLDER, PATH_TO_SCATTER_FIGURE_FOLDER, PATH_TO_MAPPINGS_FIGURE_FOLDER]:
        os.makedirs(path, exist_ok=True)

    if SAVE_PERFORMANCE_PLOTS:
        print('Creating and Saving Performance Plots')

        performance_df = pd.read_pickle(PATH_TO_PERFORMANCE_RESULTS)
        performance_df.replace({'kernel': kernel_name_nice}, inplace=True)

        save_performance_plots(perf_df=performance_df,
                               metric_names=METRIC_NAMES,
                               path_to_performance_figure_folder=PATH_TO_PERFORMANCE_FIGURE_FOLDER,
                               metric_name_nice=metric_name_nice)

    if CALCULATE_CORRELATION_STATISTICS:
        print('Calculating Correlation Statistics')

        shapley_df = pd.read_pickle(PATH_TO_SHAPLEY_RESULTS)
        shapley_df.replace({'kernel': kernel_name_nice}, inplace=True)

        save_correlation_df(shapley_df=shapley_df,
                            path_to_correlation_results=PATH_TO_CORRELATION_RESULTS,
                            correlation=CORRELATION_STATISTIC)
        

    if SAVE_CORRELATION_FIGURES:
        print('Creating and Saving Correlation Figures')

        corr_results = pd.read_pickle(PATH_TO_CORRELATION_RESULTS)

        save_correlation_plots(corr_results=corr_results,
                               path_to_correlation_figure_folder=PATH_TO_CORRELATION_FIGURE_FOLDER,
                               types=['all', 'pres', 'abs'],
                               correlation=CORRELATION_STATISTIC)
    if SAVE_SCATTER_PLOTS:
        print('Creating and Saving Scatter Figures')
        
        shapley_df = pd.read_pickle(PATH_TO_SHAPLEY_RESULTS)
        shapley_df.replace({'kernel': kernel_name_nice}, inplace=True)

        save_scatter_plots(df_shapley=shapley_df,
                           path_to_scatter_figure_folder=PATH_TO_SCATTER_FIGURE_FOLDER,
                           num_cpds_active_per_split=NUM_CPDS_SCATTER_ACTIVE_PER_SPLIT,
                           num_cpds_inactive_per_split=NUM_CPDS_SCATTER_INACTIVE_PER_SPLIT,
                           save_mols=SAVE_SCATTER_MOLS)

    if SAVE_MAPPINGS:
        print('Creating and Saving Mappings')

        shapley_df = pd.read_pickle(PATH_TO_SHAPLEY_RESULTS)
        shapley_df.replace({'kernel': kernel_name_nice}, inplace=True)
        
        save_mappings(df_shapley=shapley_df,
                      num_cpds_mappings_active_per_split=NUM_CPDS_MAPPINGS_ACTIVE_PER_SPLIT,
                      num_cpds_mappings_inactive_per_split=NUM_CPDS_MAPPINGS_INACTIVE_PER_SPLIT,
                      path_to_mappings_figure_folder=PATH_TO_MAPPINGS_FIGURE_FOLDER)
        
if __name__ == '__main__':
    main()
    print('Finished analysis.')