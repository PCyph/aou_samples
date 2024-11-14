# ALL CODE below is from Demo - Polygenic_Risk_Score_Genetic_Ancestry_Calibration

import os
import hail as hl
hl.init(default_reference='GRCh38', idempotent=True)

from bokeh.io import show, output_notebook
from bokeh.layouts import gridplot
output_notebook()

import pandas as pd
import ast

cdr_storage_path = os.environ.get("CDR_STORAGE_PATH")
project = os.environ.get("GOOGLE_PROJECT")
mt_path = os.environ.get("WGS_HAIL_STORAGE_PATH")
auxiliary_path = f'{cdr_storage_path}/wgs/vcf/aux'
sample_qc_path = f'{auxiliary_path}/qc'

mt_wgs = hl.read_matrix_table(mt_path)

relatedness = f'{auxiliary_path}/relatedness'
related_samples_path = f'{relatedness}/relatedness_flagged_samples.tsv'
related_remove = hl.import_table(related_samples_path,
                                 types={"sample_id.s":"tstr"},
                                key="sample_id.s")

mt_wgs = mt_wgs.anti_join_cols(related_remove)

flagged_samples_path = f'{sample_qc_path}/flagged_samples.tsv'
flagged_samples = hl.import_table(flagged_samples_path, key='s')

mt_wgs = mt_wgs.anti_join_cols(flagged_samples)


ancestry_pred_path = f'{auxiliary_path}/ancestry/ancestry_preds.tsv'
ancestry_pred = hl.import_table(ancestry_pred_path, key='research_id')

mt_wgs = mt_wgs.annotate_cols(ancestry_pred = ancestry_pred[mt_wgs.s].ancestry_pred)
samples_pd = mt_wgs.cols().to_pandas()

# see counts by ancestry
samples_pd.groupby("ancestry_pred")["ancestry_pred"].count()

# downsample ancestry group with more than 2000 samples to the minimum of that set (2286), 
# and combine with ancestries which were not downsampled
ancestries_to_downsample = samples_pd.groupby("ancestry_pred")["ancestry_pred"].count().loc[lambda x : x > 2000]
downsample_to = ancestries_to_downsample.min() # 2283
anestries_to_downsample_list = ancestries_to_downsample.index.tolist()
downsampled_ancestries = samples_pd.query("ancestry_pred in @anestries_to_downsample_list").\
    groupby("ancestry_pred").sample(n=downsample_to)
non_downsampled_ancestries = samples_pd.query("ancestry_pred not in @anestries_to_downsample_list")

selected_samples_pd = pd.concat([downsampled_ancestries, non_downsampled_ancestries]).set_index("s")
# check that ancestry counts look good
selected_samples_pd.groupby("ancestry_pred")["ancestry_pred"].count()

# split the downsampled_cohort randomly in half (for each ancestry) between training and testing
training_cohort_pd = selected_samples_pd.groupby("ancestry_pred").sample(frac=0.5)
test_cohort_pd = selected_samples_pd.drop(training_cohort_pd.index)
# check counts
print(f'training: {training_cohort_pd.groupby("ancestry_pred")["ancestry_pred"].count()}')
print(f'test: {test_cohort_pd.groupby("ancestry_pred")["ancestry_pred"].count()}')

training_cohort = hl.Table.from_pandas(training_cohort_pd.reset_index(), key='s')
test_cohort = hl.Table.from_pandas(test_cohort_pd.reset_index(), key='s')
full_cohort = hl.Table.from_pandas(selected_samples_pd.reset_index(), key='s')

ancestry_pred_training = ancestry_pred.semi_join(training_cohort)
ancestry_pred_training_pd = ancestry_pred_training.to_pandas()

ancestry_pred_test = ancestry_pred.semi_join(test_cohort)
ancestry_pred_test_pd = ancestry_pred_test.to_pandas()

ancestry_pred_training_pd[[f'pc_{i}' for i in range(1,17)]] = pd.DataFrame([ast.literal_eval(x) for x in ancestry_pred_training_pd.pca_features.tolist()],
                                                                    index = ancestry_pred_training_pd.index)

ancestry_pred_test_pd[[f'pc_{i}' for i in range(1,17)]] = pd.DataFrame([ast.literal_eval(x) for x in ancestry_pred_test_pd.pca_features.tolist()],
                                                                    index = ancestry_pred_test_pd.index)

import seaborn as sns
# check out pca scatter plot, make sure looks reasonable
sns.scatterplot(data = ancestry_pred_training_pd, x='pc_1', y='pc_2', hue='ancestry_pred', alpha=0.05)

sns.scatterplot(data = ancestry_pred_test_pd, x='pc_1', y='pc_2', hue='ancestry_pred', alpha=0.05)

bucket = os.getenv("WORKSPACE_BUCKET")
training_pcs=ancestry_pred_training_pd[['research_id', 'ancestry_pred'] + [f'pc_{i}' for i in range(1,17)]]
test_pcs=ancestry_pred_test_pd[['research_id', 'ancestry_pred'] + [f'pc_{i}' for i in range(1,17)]]
training_pcs.to_csv(f'{bucket}/pcs/train_set.pcs.csv', index=False)
test_pcs.to_csv(f'{bucket}/pcs/test_set.pcs.csv', index=False)

mt_wgs = mt_wgs.semi_join_cols(full_cohort)
mt_wgs.count()

weights_files_base = ["asthma_PRS_weights.txt",
                "atrial_fibrilation_PRS_weights.txt",
                "bmi_PRS_weights.txt",
                "breast_cancer_PRS_weights.txt",
                "ckd_PRS_weights.txt",
                "coronary_heart_disease_PRS_weights.txt",
                "hypercholesterolemia_PRS_weights.txt",
                "prostate_cancer_PRS_weights.txt",
                "t1d_PRS_weights.txt",
                "t2d_PRS_weights.txt"]

weights_hg38= f'{bucket}/weights/hg38'

def load_weights_and_add_condition_name(path):
    weights_ht = hl.import_table(path, types={"id":"tstr",
                                                  "effect_allele":"tstr",
                                                  "weight":"tfloat",
                                                  "chr":"tstr",
                                                  "pos":"tint32",
                                                  "a1":"tstr",
                                                  "a2":"tstr"},
                             delimiter=",",
                             key="id")
    condition = os.path.basename(path).replace("_PRS_weights.txt","")
    weights_ht = weights_ht.annotate_globals(condition=condition)
    return weights_ht

weights_tables = [load_weights_and_add_condition_name(f'{weights_hg38}/{w}') for w in weights_files_base]

import functools
all_weights = functools.reduce(lambda x,y: x.union(y), weights_tables)
all_weights = all_weights.cache()


weights_intervals_pd = all_weights.select("chr","pos").to_pandas().drop("id", axis=1)


weights_intervals_pd["end"] = weights_intervals_pd["pos"]
weights_intervals_pd = weights_intervals_pd.drop_duplicates().sort_values(["chr","pos"])
weights_intervals_pd.to_csv(f'{bucket}/weights/hg38/weights_intervals.txt',
                                             header=False,
                                             index=False,
                                             sep="\t")
weights_sites = hl.import_locus_intervals(f'{bucket}/weights/hg38/weights_intervals.txt')

weights_sites = weights_sites.cache()


mt_wgs = mt_wgs.filter_rows(hl.is_defined(weights_sites[mt_wgs.locus]))
mt_wgs = mt_wgs.select_entries(mt_wgs.GT)
mt_wgs = mt_wgs.select_rows()
mt_wgs.write(f'{bucket}/cohorts/test_and_train.mt')

import os
import hail as hl
hl.init(default_reference='GRCh38', idempotent=True)

from bokeh.io import show, output_notebook
from bokeh.layouts import gridplot
import pandas as pd
output_notebook()

bucket = os.getenv("WORKSPACE_BUCKET")
mt_path = f'{bucket}/cohorts/test_and_train.mt'
mt_cohort = hl.read_matrix_table(mt_path)
mt_cohort = mt_cohort.repartition(1000)
mt_cohort.write(f'{bucket}/cohort/test_and_train_1000_partitions.mt')

import os
import hail as hl
hl.init(default_reference='GRCh38', idempotent=True)

from bokeh.io import show, output_notebook
from bokeh.layouts import gridplot
import pandas as pd
output_notebook()

bucket = os.getenv("WORKSPACE_BUCKET")
mt_path = f'{bucket}/cohort/test_and_train_1000_partitions.mt'
mt_cohort = hl.read_matrix_table(mt_path)

weights_files_base = ["asthma_PRS_weights.txt",
                "atrial_fibrilation_PRS_weights.txt",
                "bmi_PRS_weights.txt",
                "breast_cancer_PRS_weights.txt",
                "ckd_PRS_weights.txt",
                "coronary_heart_disease_PRS_weights.txt",
                "hypercholesterolemia_PRS_weights.txt",
                "prostate_cancer_PRS_weights.txt",
                "t1d_PRS_weights.txt",
                "t2d_PRS_weights.txt"]

weights_hg38= f'{bucket}/weights/hg38'

def load_weights(path):
    weights_ht = hl.import_table(path, types={"id":"tstr",
                                                  "effect_allele":"tstr",
                                                  "weight":"tfloat",
                                                  "chr":"tstr",
                                                  "pos":"tint32",
                                                  "a1":"tstr",
                                                  "a2":"tstr"},
                             delimiter=",")
    condition = os.path.basename(path).replace("_PRS_weights.txt","")
    weights_ht = weights_ht.annotate_globals(condition=condition)
    weights_ht = weights_ht.annotate(locus=hl.locus(weights_ht.chr, weights_ht.pos))
    weights_ht = weights_ht.key_by(weights_ht.locus)
    return weights_ht

for weight_file in weights_files_base:
    weights = load_weights(os.path.join(weights_hg38, weight_file))
    mt_cohort_scored = mt_cohort.annotate_rows(effect_allele = weights[mt_cohort.locus].effect_allele, 
                            weight=weights[mt_cohort.locus].weight)
    mt_cohort_scored = mt_cohort_scored.filter_rows(hl.is_defined(mt_cohort_scored.effect_allele))
    mt_cohort_scored = mt_cohort_scored.annotate_rows(effect_allele_index = 
                                                      mt_cohort_scored.alleles.index(
                                                          mt_cohort_scored.effect_allele)
                                                     )
    mt_cohort_scored = mt_cohort_scored.annotate_cols(score=
                                              hl.agg.sum(
                                                  hl.if_else(
                                                      hl.is_defined(mt_cohort_scored.effect_allele_index),
                                                      mt_cohort_scored.GT.
                                                      one_hot_alleles(mt_cohort_scored.alleles.length())[mt_cohort_scored.effect_allele_index],
                                                      0)
                                                  * mt_cohort_scored.weight
                                              )
                                             )
    
    score_pd = mt_cohort_scored.cols().to_pandas()
    out_path = os.path.join(bucket, "cohorts", f'{weight_file.replace("_PRS_weights.txt","")}_train_and_test_scores.csv')
    score_pd.to_csv(out_path, index = False)



import numpy as np
import pandas as pd
import os
import seaborn as sns


bucket = os.getenv("WORKSPACE_BUCKET")
train_pcs_df = pd.read_csv(os.path.join(bucket, "pcs", "train_set.pcs.csv"), index_col="research_id")
test_pcs_df = pd.read_csv(os.path.join(bucket, "pcs", "test_set.pcs.csv"), index_col="research_id")

# function to extract pcs (+ 1 vec) and scores as numpy arrays
def extract_pcs_and_scores(df, pc_indices):
    pcs_np = df[[f'pc_{i}' for i in pc_indices]].to_numpy()
    n_rows = pcs_np.shape[0]
    pcs_np = np.append(np.repeat(1, n_rows).reshape(n_rows, 1), pcs_np, axis = 1)
    scores = df['score'].to_numpy()
    return pcs_np, scores

def extract_standardized_pcs_and_scores(df, pc_indices):
    pcs_np = df[[f'pc_{i}' for i in pc_indices]].to_numpy()
    pcs_np = (pcs_np - np.mean(pcs_np, axis = 0))/np.std(pcs_np, axis = 0)
    n_rows = pcs_np.shape[0]
    pcs_np = np.append(np.repeat(1, n_rows).reshape(n_rows, 1), pcs_np, axis = 1)
    scores = df['score'].to_numpy()
    return pcs_np, scores

# function to predict mean
def f_mu(pcs, theta):
    return np.dot(pcs, theta)

# function to predict variance
def f_var(pcs, theta):
    return np.exp(np.dot(pcs, theta))

# objective function
# returns loss, gradient
def obj(pcs, scores, theta):
    split_index = pcs.shape[1]
    
    theta_mu = theta[:split_index]
    theta_var = theta[split_index:]
    mu = f_mu(pcs, theta_mu)
    var = f_var(pcs, theta_var)
    # loss is - log of normal distribution (with constant log sqrt(2pi) removed)
    loss = np.sum(np.log(np.sqrt(var)) + (1/2)*(scores-mu)**2/var)
    
    #gradient of loss
    mu_coeff = -(scores-mu)/var
    sig_coeff = 1/2 - (1/2)*(scores-mu)**2/var
    grad = np.append(np.sum(pcs * mu_coeff[:,np.newaxis], axis=0), np.sum(pcs * sig_coeff[:, np.newaxis], axis=0))
    
    return loss, grad

def obj_L1(pcs, scores, theta, l):
    unreg_loss, unreg_grad = obj(pcs, scores, theta)
    loss = unreg_loss + l*np.sum(np.abs(theta))
    grad = unreg_grad + l*np.sign(theta)
    return loss, grad

def adjust_scores(pcs, scores, theta):
    split_index = pcs.shape[1]
    
    theta_mu = theta[:split_index]
    theta_var = theta[split_index:]
    mu = f_mu(pcs, theta_mu)
    var = f_var(pcs, theta_var)
    
    adjusted_scores = (scores - mu)/np.sqrt(var)
    return adjusted_scores
from scipy.optimize import minimize

conditions = ["asthma",
                "atrial_fibrilation",
                "bmi",
                "breast_cancer",
                "ckd",
                "coronary_heart_disease",
                "hypercholesterolemia",
                "prostate_cancer",
                "t2d"]

train_adjusted_scores_dfs = []
test_adjusted_scores_dfs = []
for condition in conditions:
    # load scores
    scores = pd.read_csv(os.path.join(bucket, "cohorts", f'{condition}_train_and_test_scores.csv'), index_col="s")
    train_scores_and_pcs = scores.join(train_pcs_df.drop("ancestry_pred", axis=1), how="inner")
    test_scores_and_pcs = scores.join(test_pcs_df.drop("ancestry_pred", axis=1), how="inner")
    
    n_pcs = 4
    train_pcs, train_scores = extract_pcs_and_scores(train_scores_and_pcs, range(1, n_pcs + 1))
    x0 = np.append(np.append(train_scores.mean(), np.repeat(0, n_pcs)), 
                   np.append(np.log(train_scores.var()), np.repeat(0, n_pcs))
                  )
    res = minimize(lambda theta : obj(train_pcs, train_scores, theta), x0, method="BFGS", jac=True)
    test_pcs, test_scores = extract_pcs_and_scores(test_scores_and_pcs, range(1, n_pcs + 1))
    
    adjusted_train_scores = adjust_scores(train_pcs, train_scores, res.x)
    adjusted_test_scores = adjust_scores(test_pcs, test_scores, res.x)
    
    df_train_adjusted = train_scores_and_pcs.copy()[["ancestry_pred", "score"]]
    df_train_adjusted["adjusted_score"] = adjusted_train_scores
    df_train_adjusted["condition"] = condition
    
    df_test_adjusted = test_scores_and_pcs.copy()[["ancestry_pred", "score"]]
    df_test_adjusted["adjusted_score"] = adjusted_test_scores
    df_test_adjusted["condition"] = condition
    
    train_adjusted_scores_dfs.append(df_train_adjusted)
    test_adjusted_scores_dfs.append(df_test_adjusted)
    
train_adjusted_scores = pd.concat(train_adjusted_scores_dfs)
test_adjusted_scores = pd.concat(test_adjusted_scores_dfs)


g = sns.FacetGrid(train_adjusted_scores, col='condition', hue = "ancestry_pred", 
                  col_wrap=3, sharex=False, sharey=False)
g.map_dataframe(sns.kdeplot, x='score', common_norm=False)
g.add_legend()
g.set_titles(col_template="{col_name}")


from scipy.stats import norm
import matplotlib.pyplot as plt



def map_pdf(x, **kwargs):
    x_pdf = np.linspace(-5, 5, 100)
    y_pdf = norm.pdf(x_pdf)
    plt.plot(x_pdf, y_pdf, c='k', lw=1, ls="dashed")
    
g = sns.FacetGrid(train_adjusted_scores, col='condition', hue = "ancestry_pred", 
                  col_wrap=3)
g.map_dataframe(sns.kdeplot, x='adjusted_score', common_norm=False)
g.map(map_pdf, 'adjusted_score')
g.add_legend()
g.set_titles(col_template="{col_name}")

def map_pdf(x, **kwargs):
    x_pdf = np.linspace(-5, 5, 100)
    y_pdf = norm.pdf(x_pdf)
    plt.plot(x_pdf, y_pdf, c='k', lw=1, ls="dashed")
    
g = sns.FacetGrid(test_adjusted_scores, col='condition', hue = "ancestry_pred", 
                  col_wrap=3)
g.map_dataframe(sns.kdeplot, x='adjusted_score', common_norm=False)
g.map(map_pdf, 'adjusted_score')
g.add_legend()
g.set_titles(col_template="{col_name}")

import statsmodels.api as sm

g = sns.FacetGrid(test_adjusted_scores, col='condition', 
                  col_wrap=3, hue='ancestry_pred')


def qqplot_new(x, ax=None, **kwargs):
    kwargs['data']=kwargs.pop('data')[x]
    kwargs['markeredgecolor'] = kwargs.pop('color')
    if ax is None:
        ax = plt.gca()
    sm.qqplot(ax=ax, marker='.', **kwargs, line='45')

g.map_dataframe(qqplot_new, "adjusted_score")
g.add_legend()

from statsmodels.stats.proportion import proportion_confint 

def get_high_fracs_and_ci(scores, threshold, by):
    counts = scores.groupby(by=by).size().to_frame()
    counts = counts.rename({0 : 'total'}, axis = 1)
    high_counts = scores.query('adjusted_score > @norm.ppf(@threshold)').groupby(by=by).size().to_frame()
    high_counts = high_counts.rename({0 : 'high'}, axis = 1)

    high_fracs = counts.join(high_counts)
    high_fracs = high_fracs.reset_index()

    high_fracs['frac_high'] = high_fracs['high']/high_fracs['total']
    ci_low, ci_high = proportion_confint(high_fracs['high'], high_fracs['total'], method='beta')
    high_fracs['ci_high'] = ci_high
    high_fracs['ci_low'] = ci_low
    return high_fracs
    
test_high_fracs = get_high_fracs_and_ci(test_adjusted_scores, 0.95, by=['condition', 'ancestry_pred'])
test_high_fracs['cohort'] = 'test'
train_high_fracs = get_high_fracs_and_ci(train_adjusted_scores, 0.95, by=['condition', 'ancestry_pred'])
train_high_fracs['cohort'] = 'train'

high_fracs = pd.concat([test_high_fracs, train_high_fracs])

g = sns.FacetGrid(high_fracs, col='condition', 
                  col_wrap=3, hue='cohort')

def err_plot(x, y, y_ci_high, y_ci_low, **kwargs):
    x_d = kwargs['data'][x]
    y_d = kwargs['data'][y]
    y_ci_high_err = kwargs['data'][y_ci_high] - y_d
    y_ci_low_err = y_d - kwargs['data'][y_ci_low]
    kwargs.pop('data')
    plt.errorbar(x_d, y_d, yerr=(y_ci_low_err, y_ci_high_err), fmt='.', **kwargs)

g.map_dataframe(err_plot, x = 'ancestry_pred', y='frac_high', 
                y_ci_high = 'ci_high', y_ci_low = 'ci_low')
g.refline(y=0.05)
g.add_legend()

scores = pd.read_csv(os.path.join(bucket, "cohorts", 'asthma_train_and_test_scores.csv'), index_col="s")
train_scores_and_pcs = scores.join(train_pcs_df.drop("ancestry_pred", axis=1), how="inner")
test_scores_and_pcs = scores.join(test_pcs_df.drop("ancestry_pred", axis=1), how="inner")

n_pcs_a = []
train_loss_a = []
test_loss_a = []

for n_pcs in range(1,17):
    train_pcs, train_scores = extract_pcs_and_scores(train_scores_and_pcs, range(1, n_pcs + 1))
    x0 = np.append(np.append(train_scores.mean(), np.repeat(0, n_pcs)), 
                   np.append(np.log(train_scores.var()), np.repeat(0, n_pcs))
                  )
    res = minimize(lambda theta : obj(train_pcs, train_scores, theta), x0, method="BFGS", jac=True)
    
    test_pcs, test_scores = extract_pcs_and_scores(test_scores_and_pcs, range(1, n_pcs + 1))
    
    test_loss = obj(test_pcs, test_scores, res.x)[0]
    
    train_loss_a.append(res.fun)
    test_loss_a.append(test_loss)
    n_pcs_a.append(n_pcs)
    
data = pd.DataFrame({'n_pcs': n_pcs_a + n_pcs_a,
                    'loss' : train_loss_a + test_loss_a,
                    'cohort' : ['train' for _ in range(len(n_pcs_a))] + ['test' for _ in range(len(n_pcs_a))]
                    })

sns.pointplot(data, x='n_pcs', y='loss', hue='cohort')

chosen_pcs = [1,2,3,6,10]
train_pcs, train_scores = extract_pcs_and_scores(train_scores_and_pcs, chosen_pcs)
x0 = np.append(np.append(train_scores.mean(), np.repeat(0, len(chosen_pcs))), 
               np.append(np.log(train_scores.var()), np.repeat(0, len(chosen_pcs)))
              )
res = minimize(lambda theta : obj(train_pcs, train_scores, theta), x0, method="BFGS", jac=True)

test_pcs, test_scores = extract_pcs_and_scores(test_scores_and_pcs, chosen_pcs)

test_loss = obj(test_pcs, test_scores, res.x)[0]

print(f'All 16 pcs, train_loss={train_loss_a[-1]}, test_loss={test_loss_a[-1]}')
print(f'First 5 pcs, train_loss={train_loss_a[4]}, test_loss={test_loss_a[4]}')
print(f'pcs {chosen_pcs}, train_loss={res.fun}, test_loss={test_loss}')

loss_dfs = []

for condition in conditions:
    # load scores
    scores = pd.read_csv(os.path.join(bucket, "cohorts", f'{condition}_train_and_test_scores.csv'), index_col="s")
    train_scores_and_pcs = scores.join(train_pcs_df.drop("ancestry_pred", axis=1), how="inner")
    test_scores_and_pcs = scores.join(test_pcs_df.drop("ancestry_pred", axis=1), how="inner")
    

    train_loss_a = []
    test_loss_a = []
    for n_pcs in range(1,17):
        train_pcs, train_scores = extract_pcs_and_scores(train_scores_and_pcs, range(1, n_pcs + 1))
        x0 = np.append(np.append(train_scores.mean(), np.repeat(0, n_pcs)), 
                       np.append(np.log(train_scores.var()), np.repeat(0, n_pcs))
                      )
        res = minimize(lambda theta : obj(train_pcs, train_scores, theta), x0, method="BFGS", jac=True)
        test_pcs, test_scores = extract_pcs_and_scores(test_scores_and_pcs, range(1, n_pcs + 1))
        
        test_loss = obj(test_pcs, test_scores, res.x)[0]
        train_loss_a.append(res.fun)
        test_loss_a.append(test_loss)
        
    loss_df = pd.DataFrame({'n_pcs': np.tile([i for i in range(1,17)], 2),
                    'loss' : train_loss_a + test_loss_a,
                    'cohort' : ['train' for _ in range(len(n_pcs_a))] + ['test' for _ in range(len(n_pcs_a))],
                    'condition' : np.repeat(condition, 32)
                    }) 
    loss_dfs.append(loss_df)
    
all_loss_df = pd.concat(loss_dfs)
g = sns.FacetGrid(all_loss_df, col='condition', 
                  col_wrap=3, hue='cohort', sharey=False)
g.map_dataframe(sns.lineplot, x='n_pcs', y='loss')
g.add_legend()

from scipy.optimize import minimize_scalar
scores = pd.read_csv(os.path.join(bucket, "cohorts", 'asthma_train_and_test_scores.csv'), index_col="s")
train_scores_and_pcs = scores.join(train_pcs_df.drop("ancestry_pred", axis=1), how="inner")
test_scores_and_pcs = scores.join(test_pcs_df.drop("ancestry_pred", axis=1), how="inner")

train_pcs, train_scores = extract_standardized_pcs_and_scores(train_scores_and_pcs, range(1,17))

def obj_cross_val(pcs, scores, l):
    ret = 0
    n_per_batch = int(len(pcs)/5)
    for i in range(5):
        i_min = i*n_per_batch
        i_max = min(len(pcs)-1, (i+1)*n_per_batch)
        sub_pcs_train = np.delete(pcs, range(i_min,i_max + 1), axis = 0)
        sub_scores_train = np.delete(scores, range(i_min,i_max + 1), axis = 0)
        sub_pcs_val = pcs[i_min:i_max]
        sub_scores_val = scores[i_min:i_max]
        x0 = np.append(np.append(sub_scores_train.mean(), np.repeat(0, 16)), 
               np.append(np.log(sub_scores_train.var()), np.repeat(0, 16))
              )
        res_L1 = minimize(lambda theta : obj_L1(sub_pcs_train, sub_scores_train, theta, l), 
                          x0, method="BFGS", jac=True)
        
        ret += obj(sub_pcs_val, sub_scores_val, res_L1.x)[0]

    return ret

shuffled_indices = np.random.permutation(len(train_pcs))

res_xval = minimize_scalar(lambda x: obj_cross_val(train_pcs[shuffled_indices], train_scores[shuffled_indices], x), tol=5)
l=res_xval.x


x0 = np.append(np.append(train_scores.mean(), np.repeat(0, 16)), 
               np.append(np.log(train_scores.var()), np.repeat(0, 16))
              )
res_L1 = minimize(lambda theta : obj_L1(train_pcs, train_scores, theta, res_xval.x), x0, method="BFGS", jac=True)
res = minimize(lambda theta : obj(train_pcs, train_scores, theta), x0, method="BFGS", jac=True)

test_pcs, test_scores = extract_standardized_pcs_and_scores(test_scores_and_pcs, range(1,17))
print(f'train loss={res.fun}, train loss w/ reg = {obj(train_pcs, train_scores, res_L1.x)[0]}')
print(f'test loss={obj(test_pcs, test_scores, res.x)[0]}, test loss w/ reg = {obj(test_pcs, test_scores, res_L1.x)[0]}')

x0 = np.append(np.append(train_scores.mean(), np.repeat(0, 4)), 
               np.append(np.log(train_scores.var()), np.repeat(0, 4))
              )
res_4 = minimize(lambda theta : obj(train_pcs[:,:5], train_scores, theta), x0, method="BFGS", jac=True)
train_adjusted_scores_4_pcs = adjust_scores(train_pcs[:,:5], train_scores, res_4.x)
train_adjusted_scores_16_pcs = adjust_scores(train_pcs, train_scores, res.x)
train_adjusted_scores_16_pcs_L1 = adjust_scores(train_pcs, train_scores, res_L1.x)


test_adjusted_scores_4_pcs = adjust_scores(test_pcs[:,:5], test_scores, res_4.x)
test_adjusted_scores_16_pcs = adjust_scores(test_pcs, test_scores, res.x)
test_adjusted_scores_16_pcs_L1 = adjust_scores(test_pcs, test_scores, res_L1.x)
train_4_pcs = pd.DataFrame({'ancestry_pred' : train_scores_and_pcs.ancestry_pred,
                          'adjusted_score' : train_adjusted_scores_4_pcs,
                          'type' : '4 pcs', 
                           'cohort' : 'train'})

test_4_pcs = pd.DataFrame({'ancestry_pred' : test_scores_and_pcs.ancestry_pred,
                          'adjusted_score' : test_adjusted_scores_4_pcs,
                          'type' : '4 pcs', 
                           'cohort' : 'test'})

train_16_pcs = pd.DataFrame({'ancestry_pred' : train_scores_and_pcs.ancestry_pred,
                          'adjusted_score' : train_adjusted_scores_16_pcs,
                          'type' : '16 pcs', 
                           'cohort' : 'train'})

test_16_pcs = pd.DataFrame({'ancestry_pred' : test_scores_and_pcs.ancestry_pred,
                          'adjusted_score' : test_adjusted_scores_16_pcs,
                          'type' : '16 pcs', 
                           'cohort' : 'test'})

train_16_pcs_L1 = pd.DataFrame({'ancestry_pred' : train_scores_and_pcs.ancestry_pred,
                          'adjusted_score' : train_adjusted_scores_16_pcs_L1,
                          'type' : '16 pcs L1 reg', 
                           'cohort' : 'train'})

test_16_pcs_L1 = pd.DataFrame({'ancestry_pred' : test_scores_and_pcs.ancestry_pred,
                          'adjusted_score' : test_adjusted_scores_16_pcs_L1,
                          'type' : '16 pcs L1 reg', 
                           'cohort' : 'test'})

asthma_adjusted_scores = pd.concat([train_4_pcs, test_4_pcs,
                                   train_16_pcs, test_16_pcs,
                                   train_16_pcs_L1, test_16_pcs_L1])

g = sns.FacetGrid(asthma_adjusted_scores.query('cohort=="test"'), col='type', hue = "ancestry_pred", 
                  col_wrap=3)
g.map_dataframe(sns.kdeplot, x='adjusted_score', common_norm=False)
g.map(map_pdf, 'adjusted_score')
g.add_legend()
g.set_titles(col_template="{col_name}")



g = sns.FacetGrid(asthma_adjusted_scores.query('cohort=="test"'), col='type', 
                  col_wrap=3, hue='ancestry_pred')

g.map_dataframe(qqplot_new, "adjusted_score")
g.add_legend()


high_fracs_asthma = get_high_fracs_and_ci(asthma_adjusted_scores, 0.95, by=['type', 'ancestry_pred', 'cohort'])

g = sns.FacetGrid(high_fracs_asthma, col='type', 
                  hue='cohort')

g.map_dataframe(err_plot, x = 'ancestry_pred', y='frac_high', 
                y_ci_high = 'ci_high', y_ci_low = 'ci_low')
g.refline(y=0.05)
g.add_legend()
