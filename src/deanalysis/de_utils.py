import os

os.environ['R_HOME'] = "/ahg/regevdata/users/kjag/.conda/envs/deanalysis/lib/R"

import scanpy as sc
import pandas as pd
import numpy as np
import scipy
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from IPython.display import display
from collections import Counter
from matplotlib import pyplot as plt
import random
import time

# R integration
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri, r, Formula
from rpy2.robjects.vectors import StrVector, FloatVector, ListVector
import rpy2.robjects as ro

lme4 = importr('lme4')
lmer = importr('lmerTest') # overloads lmer function from lme4 package
base = importr('base')
stats = importr('stats')

COLORS=["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744"]

def fit_lme(formula, df, family='gaussian', random_effect=True, **fit_kwargs):
    f = Formula(formula)

    with localconverter(ro.default_converter + pandas2ri.converter):
        if family == 'gaussian':
            if random_effect:
                control = lme4.lmerControl(**{'calc.derivs': True,
                                              'check.rankX': 'silent.drop.cols', 
                                              'check.conv.singular': r('lme4::.makeCC')(action = "ignore",  tol = 1e-4)})
                fit = lmer.lmer(f, df, control=control, **fit_kwargs)
            else:
                fit = stats.lm(f, df, **fit_kwargs)
        elif family in ('binomial', 'poisson'):
            if random_effect:
                fit = lme4.glmer(f, df, nAGQ=0, family=family, **fit_kwargs)
            else:
                fit = stats.glm(f, df, nAGQ=0, family=family, **fit_kwargs)
        else:
            if random_effect:
                control = lme4.glmerControl(**{'optimizer': 'nloptwrap', 
                                   'calc.derivs': True,
                                   'check.rankX': 'silent.drop.cols',
                                   'check.conv.singular': r('lme4::.makeCC')(action = "ignore",  tol = 1e-4)})
                fit = r('lme4::glmer.nb')(f, df, **{'nb.control': control}, **fit_kwargs)
            else:
                fit = r('MASS::glm.nb')(f, df, **fit_kwargs)
        
        anova_df = stats.anova(fit)
    
    coef_df = r['as.data.frame'](stats.coef(base.summary(fit)))
    coef_df = pandas2ri.rpy2py(coef_df)
    
    return coef_df, anova_df


def fit_lme_python(formula, df, family='gaussian', random_effect=True, **fit_kwargs):

    if family == 'gaussian':
        if random_effect:
            pass
        else:
            df = sm.add_constant(df)
            model = sm.GLM.from_formula(formula, df, family=sm.families.Gaussian())
            res = model.fit()
            res = res.summary2().tables[1]
    elif family in ('binomial', 'poisson'):
        if random_effect:
            df = sm.add_constant(df)
            df['pid'] = df['pid'].astype('category')
            model = PoissonBayesMixedGLM.from_formula(formula, {"a": '0 + C(pid)'}, df)
            res = model.fit_map()
            res = res.summary().tables[0]
        else:
            # 'gene ~ treatment_status + percent_mito + total_counts + PCA0 + PCA1 + PCA2',
            df = sm.add_constant(df)
            #df['total_counts'] = np.log(df['total_counts'])
            #df['pid'] = df['pid'].astype('category')
            #formula = 'gene ~ de_analysis + percent_mito + total_counts + PCA0 + PCA1 + PCA2'
            #formula = formula
            model = sm.GLM.from_formula(formula, df, family=sm.families.Poisson())
            res = model.fit()
            res = res.summary2().tables[1]
    else:
        if random_effect:
            pass
            """
            control = lme4.glmerControl(**{'optimizer': 'nloptwrap', 
                               'calc.derivs': True,
                               'check.rankX': 'silent.drop.cols',
                               'check.conv.singular': r('lme4::.makeCC')(action = "ignore",  tol = 1e-4)})
            fit = r('lme4::glmer.nb')(f, df, **{'nb.control': control}, **fit_kwargs)
            """
        else:
            pass

    return res

def _fit(formula, gene, adata, obs_features, use_raw, family, random_effect):
    
    gene_vec = adata.obs_vector(gene) if not use_raw else adata.obs_vector(gene, layer='counts')

    covariates = adata.obs[obs_features].copy()
    covariates.loc[:, 'gene'] = gene_vec
    
    try:
        coefs, anova = fit_lme(formula, covariates, family=family, random_effect=random_effect)
    except Exception as e:
        print("Error for", gene)
        print(e)
        coefs, anova = None, None

    return coefs, anova


def fit_lme_adata(adata, formula, obs_features, family='gaussian', random_effect=False, use_raw=False, n_jobs=6):

    adata = adata.copy()
    
    if n_jobs == 1:
        para_result = [_fit(formula, 
                            gene, 
                            adata, 
                            obs_features, 
                            use_raw, 
                            family,
                            random_effect) for gene in adata.var_names]
    else:    
        para_result = Parallel(n_jobs=n_jobs)(delayed(_fit)(formula, 
                                                            gene,
                                                            adata,
                                                            obs_features, 
                                                            use_raw, 
                                                            family,
                                                            random_effect) for gene in adata.var_names)
      
    coef_df = {k:v[0] for k, v in zip(adata.var_names, para_result) if v[0] is not None}
    coef_df = pd.concat([df.assign(gene=gene) for gene, df in coef_df.items()], axis=0)
    coef_df = coef_df.reset_index().rename(columns={'index': 'fixed_effect'})
    
    return coef_df

def compute_pvalues(coef):
    if 'z' in coef.columns:
        coef['zscore'] = coef['z']
    else:
        coef['zscore'] = coef['Post. Mean']/coef['Post. SD']
    coef['Pr(>|z|)'] = scipy.stats.norm.sf(abs(coef['zscore']))
    sig, pval, _, _ = multipletests(coef['Pr(>|z|)'], method='fdr_bh', alpha=0.1)
    coef['significant'] = sig
    coef['pval_adj'] = pval
    coef['neglog_pval_adj'] = -np.log10(coef.pval_adj+1e-300)
    return coef

def fit_DE_model(adata, label, celltype, n_jobs=20):
    res = []
    
    print (celltype)
    celltypeadata = adata

    print(f'*** Dataset: {label:>5} Cell type: {celltype:>5} # genes: {celltypeadata.shape[1]:>5} # cells: {celltypeadata.shape[0]:>5} ***')
    celltypeadata.obs['pid'] = celltypeadata.obs['pid'].astype('category')
    celltypeadata.obs['de_analysis'] = celltypeadata.obs['de_analysis'].astype('category')
    
    treatment_col = 'None'
    if 'de_analysis' in celltypeadata.obs.columns:
        print(set(celltypeadata.obs.de_analysis))
        source_df = pd.get_dummies(celltypeadata.obs.de_analysis)
        for col in source_df.columns:
            celltypeadata.obs[col] = source_df[col].astype('int')
            treatment_col = source_df.columns[source_df.columns!='Untreated'][0]

    coef = fit_lme_adata(celltypeadata, 
                                    'gene ~ (1|pid) + offset(log(total_counts)) + PCA0 + PCA1 + PCA2 + ' + treatment_col,
                                    #'gene ~ offset(log(cell_counts)) + ' + treatment_col,
                                    ['pid', 'total_counts', 'PCA0', 'PCA1', 'PCA2', treatment_col],
                                    #['cell_counts', treatment_col],
                                    family='poisson', #'gaussian'
                                    random_effect = True,#False,
                                    use_raw=True, #False,
                                    n_jobs=1 #min(n_jobs, celltypeadata.shape[1])
                        )
    
    #coef = compute_pvalues(coef)
    #res.append(coef)
    
    sig, pval, _, _ = multipletests(coef['Pr(>|z|)'], method='fdr_bh', alpha=0.1)
    #sig, pval, _, _ = multipletests(coef['Pr(>|t|)'], method='fdr_bh', alpha=0.1)
    coef['significant'] = sig
    coef['pval_adj'] = pval
    coef['neglog_pval_adj'] = -np.log10(coef.pval_adj+1e-300)
    res.append(coef.assign(celltype=label))
            
    if not res:
        return None

    res = pd.concat(res, axis=0)
    #res = res[res.fixed_effect == 'genotype'].sort_values('pval_adj').reset_index(drop=True).assign(dataset=label)
    res = res[res.fixed_effect.str.startswith(treatment_col)]
    return res.sort_values('pval_adj').reset_index(drop=True).assign(dataset=label)