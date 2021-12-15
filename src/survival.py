import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial import distance
from scipy.optimize import minimize
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns

from pdac_utils import *
from scipy.optimize import minimize

markers = {
            'Alpha': ['GCG'],
            'Beta': ['INS', 'IAPP'],
            'Delta': ['SST'],
            'Gamma':[ 'PPY'],
            'Epsilon': ['GHRL'],
            'Neuroendocrine': ['SYP', 'CHGA', 'VGF'],
            'Endothelial': ['PECAM1', 'VWF'],
            'Ductal/tumor': ['CFTR', 'KRT19', 'KRT7', 'KRT17', 'EPCAM', 'CEACAM6', 'COL17A1', 'MECOM'],
            'Acinar': ['CPB1', 'PRSS3', 'AMY1A'],
            'Macrophage': ['CD68', 'CD163', 'MRC1', 'CD80', 'CD86', 'TGFB1', 'CSF1'],
            'cDC1': ['XCR1', 'CST3', 'CLEC9A', 'LGALS2'],
            'cDC2':[ 'CD1A', 'CD207', 'CD1E', 'FCER1A', 'NDRG2'],
            'Activated' 'DC': ['FSCN1', 'LAMP3', 'CCL19', 'CCR7'],
            'pDC': ['GZMB', 'IRF7', 'LILRA4', 'TCF4', 'CXCR3', 'IRF4'],
            'T cell': ['CD4', 'CD8A', 'CD8B', 'CD3D', 'THEMIS', 'CD96', 'IKZF1', 'GZMA', 'FOXP3'],
            'B cell': ['BANK1', 'CD19'],
            'NK cell': ['KLRD1', 'KIR2DL3', 'IL18R1', 'KIR2DL1', 'KIR3DL2'],
            'Plasma': ['SDC1', 'IGLC2'],
            'Neutrophil': ['CSF3R', 'CXCL8'],
            'CAF': ['COL1A1', 'FN1', 'PDPN', 'DCN', 'VIM', 'FAP', 'ACTA2', 'IL6', 'C3', 'LIF', 'POSTN', 'FBLN1'],
            'Schwann': ['SOX10', 'S100B', 'NGFR'],
            'Intrapancreatic neurons': ['TH', 'CHAT', 'ENO2',  'TAC1'],
}

class Deconv:
    
    def __init__(self, scrnaseq, anno):
        sc.pp.highly_variable_genes(scrnaseq, batch_key='pid', n_top_genes=2000)
        self.basis_ = self.compute_basis(scrnaseq, anno)
        self.malignant_topics_ = None
        self.fibro_topics_ = None
        self.highly_variable_genes_ = scrnaseq.var_names[scrnaseq.var['highly_variable']]
        self.genes_ = []
        for k, v in markers.items():
            self.genes_ = self.genes_ + v
        
        self.genes_intersect_ = None
        self.estimated_prop_ = None
        self.estimated_malignant_prop_ = None
        self.estimated_fibro_prop_ = None
    
    def compute_basis(self, adata, anno):
        tokeep_cts = ['CD8+ T', 'Dendritic', 'Macrophage', 'CD4+ T', 
                      'Treg', 'CAF', 'Epithelial (malignant)', 'Schwann',  'Neutrophil', 
                      'Natural killer',  'Intra-pancreatic neurons',  'Plasma', 'Schwann', 'Acinar']

        unique_cell_types = list(set(adata.obs[anno]).intersection(tokeep_cts))
        centroids = []
        for ct in unique_cell_types:
            cell_type_data = adata[adata.obs[anno] == ct]
            centroids.append(np.mean(cell_type_data.X, axis=0))
        centroids = np.vstack(centroids)
        centroids = pd.DataFrame(centroids, index=unique_cell_types, columns=adata.var_names)        
        print(centroids.shape)
        #sns.clustermap(centroids)
        return centroids
    
    def run_deconv(self, bulk_adata):
        
        # cost function for deconvolution
        def cost_function(weights, basis):
            weighted_avg = np.dot(basis.T, weights)
            pearson = -pearsonr(bulk_sample, weighted_avg)[0]
            mse = mean_squared_error(bulk_sample, weighted_avg)
            cos = distance.cosine(bulk_sample, weighted_avg)
            pearson = pearson #+ entropy(weights)
            return pearson
        
        self.genes_intersect_ = list(bulk_adata.var_names.intersection(self.genes_))
        bulk_samples = pd.DataFrame(bulk_adata.X, index=bulk_adata.obs_names, columns=bulk_adata.var_names)
        
        basis = self.basis_[self.genes_intersect_]
        
        #common_genes = list(set(bulk_samples.columns).intersection(self.malignant_topics_.index).intersection(self.highly_variable_genes_))
        #malignant_topics = self.malignant_topics_.T[common_genes].T
        #malignant_topic_deconv = cosine_similarity(malignant_topics.T, bulk_samples[common_genes])
        #malignant_topic_deconv = np.dot(bulk_samples[common_genes], malignant_topics)
        
        for col in self.malignant_topics_:
            gene_subset = list(set(self.malignant_topics_.sort_values(by=col, ascending=False).index[0:200]).intersection(bulk_adata.var_names))
            subset = bulk_adata[:,gene_subset]
            bulk_adata.obs[col] = np.sum(subset.X, axis=1)
            #sc.tl.score_genes(bulk_adata, list(self.malignant_topics_.sort_values(by=col, ascending=False).index[0:200]), score_name=col, n_bins=0, ctrl_size=None)
        self.estimated_malignant_prop_ = bulk_adata.obs[self.malignant_topics_.columns]
        
        #common_genes = list(set(bulk_samples.columns).intersection(self.fibro_topics_.index).intersection(self.highly_variable_genes_))
        #fibro_topics = self.fibro_topics_.T[common_genes].T
        #fibro_topic_deconv = cosine_similarity(fibro_topics.T, bulk_samples[common_genes])
        #fibro_topic_deconv = np.dot(bulk_samples[common_genes], fibro_topics)
        #self.estimated_fibro_prop_ = fibro_topic_deconv
        for col in self.fibro_topics_:
            gene_subset = list(set(self.fibro_topics_.sort_values(by=col, ascending=False).index[0:200]).intersection(bulk_adata.var_names))
            subset = bulk_adata[:,gene_subset]
            bulk_adata.obs[col] = np.sum(subset.X, axis=1)
            #sc.tl.score_genes(bulk_adata, list(self.fibro_topics_.sort_values(by=col, ascending=False).index[0:200]), score_name=col, n_bins=0, ctrl_size=None)
        self.estimated_fibro_prop_ = bulk_adata.obs[self.fibro_topics_.columns]
        
        bulk_samples = bulk_samples[self.genes_intersect_]
        basis = self.basis_[self.genes_intersect_]
        estimated_prop = []
        for i in bulk_samples.index:
            print("Deconvoluting sample " + str(i))
            bulk_sample = bulk_samples[bulk_samples.index==i].values.reshape(-1)
            # Non-negative least squares algorithm
            weights = np.ones(basis.shape[0])/basis.shape[0]
            constraint = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
            res = minimize(cost_function, 
                           weights,
                           basis,
                           method='SLSQP', 
                           tol=1e-8, 
                           bounds=tuple((0,1) for x in weights), 
                           constraints=constraint
                          )
            estimated_prop.append(res.x)
            
        self.estimated_prop_ = np.array(estimated_prop)
        
        bulktopics = ['Moffitt_basal', 'Moffitt_classical', 'Collison_classical', 'Collison_QM', 'Bailey_squamous', 'Bailey_progenitor']
        genemarkers = load_genemarkers()
        for subtype in bulktopics:
            topic_genes = genemarkers[subtype]
            sc.tl.score_genes(bulk_adata, topic_genes, score_name=subtype)
        bulktopic_scores = bulk_adata.obs[bulktopics]
        
        features = pd.DataFrame(np.hstack([self.estimated_prop_, self.estimated_malignant_prop_, self.estimated_fibro_prop_, bulktopic_scores.values]))
        features.index = bulk_adata.obs_names
        features.columns = list(self.basis_.index) + list(self.malignant_topics_.columns) + list(self.fibro_topics_.columns) + bulktopics
        return features
    
    def malignant_topics(self):
        malignant = sc.read('/ahg/regevdata/projects/Pancreas/src/cNMF/naive_tumor_cNMF/tumor_nmf.h5ad')
        other_topics  = ['CRTln', 'CRTn', 'CRTx', 'GART', 'RT']
        malignant.obs['new_treatment'] = ['Other' if treatment in other_topics else treatment for treatment in malignant.obs.treatment_status]
        malignant = malignant[malignant.obs['new_treatment']!='Healthy']
        malignant = malignant[malignant.obs['Level 1 Annotation']!='REMOVE']
        malignant = malignant[malignant.obs['Level 1 Annotation']!='None']

        malignant_topic_map = {9:'Neuroendocrine-like',
        17:'Mesenchymal',
        31:'Classical-like',
        33:'Acinar-like',
        42:'Neuronal-like',
        46:'Basaloid',
        47:'Squamoid',
        6:'Cycling',
        7:'Ribosomal',
        16:'Adhesive_state',
        21:'Interferon signaling',
        22:'Cycling (G2/M)',
        27:'TNF-NFkB signaling',
        34:'MYC signaling',
        }

        df = pd.DataFrame(malignant.varm['genescores_nmf49']).fillna(0)
        lineage_topics = [9,17,31,33,42,46,47,6,7,16, 21,22,27,34]
        df = df[lineage_topics]
        df.columns = [malignant_topic_map.get(col) for col in df.columns]
        df.index = malignant.var_names

        for col in df.columns:
            malignant.var[col] = list(df[col])

        malignant_topicweights = malignant.var[df.columns]
        self.malignant_topics_ = malignant_topicweights
        
    def fibroblast_topics(self):
        fibro = sc.read('/ahg/regevdata/projects/Pancreas/src/cNMF/naive_fibroblast_cNMF/fibroblast_nmf.h5ad')

        other_topics  = ['CRTln', 'CRTn', 'CRTx', 'GART', 'RT']
        fibro.obs['new_treatment'] = ['Other' if treatment in other_topics else treatment for treatment in fibro.obs.treatment_status]
        fibro = fibro[fibro.obs['new_treatment']!='Healthy']
        fibro = fibro[fibro.obs['Level 1 Annotation']!='REMOVE']
        fibro = fibro[fibro.obs['Level 1 Annotation']!='None']

        fibro_topic_map = {0:'Immunomodulatory',
        1:'Neurotropic',
        2:'Adhesive',
        3:'Myofibroblastic'}

        df = pd.DataFrame(fibro.varm['genescores_nmf4']).fillna(0)
        fibro_topics = [0,1,2,3]
        df = df[fibro_topics]
        df.columns = [fibro_topic_map.get(col) for col in df.columns]
        df.index = fibro.var_names

        for col in df.columns:
            fibro.var[col] = list(df[col])

        fibro_topicweights = fibro.var[df.columns]
        self.fibro_topics_ = fibro_topicweights
    
    