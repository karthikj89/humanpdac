import scanpy as sc
from de_utils import *
import sys
import os
from collections import defaultdict
import pybedtools

os.environ['R_HOME'] = "/ahg/regevdata/users/kjag/.conda/envs/deanalysis/lib/R"


info = sys.argv[1]
randomeffect = sys.argv[2]
gene_idx_start = int(sys.argv[3])
gene_idx_end = sys.argv[4]

treatmenttype1, _, treatmenttype2, celltype, level = info.split('_')
print(treatmenttype1, _, treatmenttype2, celltype, level)

totaldata = sc.read('/ahg/regevdata/projects/Pancreas/cellbender/processed/totaldata-clean-infercnv-annotated.h5ad')

if celltype=='Immune':
    totaldata = totaldata[(totaldata.obs['Level %s Annotation'%level]=='Lymphoid')|(totaldata.obs['Level %s Annotation'%level]=='Myeloid')].copy()
else:
    totaldata = totaldata[(totaldata.obs['Level %s Annotation'%level]==celltype)].copy()

print(totaldata.shape)

to_keep = []
for treatment in totaldata.obs.treatment_status:
    if treatmenttype1=='Treated':
        if treatment=='Untreated':
            to_keep.append(treatment)
        elif treatment=='Healthy':
            to_keep.append('Other')
        else:
            to_keep.append('Treated')
    else:
        if treatment==treatmenttype1 or treatment==treatmenttype2:
            to_keep.append(treatment)
        else:
            to_keep.append('Other')
totaldata.obs['de_analysis'] = to_keep

totaldata = totaldata[totaldata.obs['de_analysis']!='Other']

totaldata = totaldata[:,(np.sum(totaldata.X!=0, axis=0) > 10)].copy()

if gene_idx_end=='None':
    gene_idx_end = len(totaldata.var_names)
    genes = totaldata.var_names[gene_idx_start:]
else:
    gene_idx_end = int(gene_idx_end)
    genes = totaldata.var_names[gene_idx_start:gene_idx_end]

for i in range(5):
    totaldata.obs['PCA%d'%i] = totaldata.obsm['X_pca'][:,i]

totaldata = totaldata[:,genes].copy()
print(totaldata.shape)
groupby = []
subjectlabels = []
for subject in set(totaldata.obs.pid):
    subsubset = totaldata[totaldata.obs['pid']==subject]
    if subsubset.shape[0] > 0:
        groupby.append(subsubset.X.mean(axis=0))
        subjectlabels.append(subject)
groupby = np.vstack(groupby)

pid2treatment = dict(list(set([(pid, treatment_status) for treatment_status, pid in zip(totaldata.obs['de_analysis'], totaldata.obs['pid'])])))

pseudobulkadata = sc.AnnData(groupby)
pseudobulkadata.var_names = totaldata.var_names
pseudobulkadata.obs_names = subjectlabels
pseudobulkadata.obs['de_analysis'] = [pid2treatment.get(pid) for pid in pseudobulkadata.obs_names]

counts = Counter(totaldata.obs.pid)
pseudobulkadata.obs['cell_counts'] = [counts.get(pid) for pid in pseudobulkadata.obs_names]
totaldata.obs['cell_counts'] = [counts.get(pid) for pid in totaldata.obs.pid]

randomeffect_bool = True if randomeffect=='True' else False

#if randomeffect_bool:
#    pseudobulkadata = pseudobulkadata[pseudobulkadata.obs['cell_counts'] > 50].copy()
#    totaldata = totaldata[totaldata.obs['cell_counts'] > 50].copy()
res = fit_DE_model(totaldata, treatmenttype1 + '_' + treatmenttype2, celltype, n_jobs=1)
#res = fit_DE_model(pseudobulkadata, treatmenttype1 + '_' + treatmenttype2, celltype, n_jobs=1)
res.to_csv("/ahg/regevdata/projects/Pancreas/src/deanalysis/%s_%d_%d.csv"%(info, gene_idx_start, gene_idx_end))
