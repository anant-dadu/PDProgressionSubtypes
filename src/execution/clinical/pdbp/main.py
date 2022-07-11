# input parameters

from definitions import ROOT_DIR_INSIDE
import os
PDBP_CLINICAL_RAW_DATA_DIR_INSIDE = ROOT_DIR_INSIDE / 'raw_data/clinical/pdbp'
PDBP_CLINICAL_GEN_DATA_DIR_INSIDE = ROOT_DIR_INSIDE / 'generated_data/clinical/pdbp'
PPMI_CLINICAL_GEN_DATA_DIR_INSIDE = ROOT_DIR_INSIDE / 'generated_data/clinical/ppmi'
os.makedirs(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'clustering', exist_ok=True)
os.makedirs(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction', exist_ok=True)
os.makedirs(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'preprocessed', exist_ok=True)
os.makedirs(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'representation_learning', exist_ok=True)


import json
import os
import pandas as pd
import numpy as np
import pickle
import copy
from dotmap import DotMap

import logzero
import logging
from logzero import logger
logzero.loglevel(logging.INFO)

import src.data_processing.clinical.pdbp.main as fe
import src.data_processing.clinical.pdbp.load_data as ldp

import src.models.representation_learning.main as rl
import src.models.representation_learning.methods as rlm

import src.models.clustering.methods as clm
import src.models.clustering.main as cl
import sys
save_name = sys.argv[1]
files_of_interest = ['epworth', 'HDRS', 'updrs', 'moca', 'schwab', 'pdq39', 'upsit']
data_parameters_list = [
    {
        'visits_list': ['Baseline', '12 months', '24 months', '36 months'],
        'data_dir': '{}/download20190315/'.format(str(PDBP_CLINICAL_RAW_DATA_DIR_INSIDE)),
        'preprocessing_type': 'new',
        'selection_availability': 0,
        'merge_screen_baseline': [],
        'flipping': 1,
        'remove_outlier': 1,
        'last_visits':['Baseline', '12 months', '24 months', '36 months'],
        'gmm_model': 'gmm_model$paper_experiment_replication_adj',
        'gmm_model_3d': 'gmm_model_3d$paper_experiment_replication_adj',
        'name': 'paper_experiment_replication',
        'dataset': 'pdbp',
        'files_of_interest': ['epworth', 'HDRS', 'updrs', 'moca', 'schwab', 'pdq39', 'upsit'] 
    },
]

data_names = []
for data_parameters in data_parameters_list:
    temp = data_parameters['name']
    data_names.append(temp)

for e in range(len(data_parameters_list)):
    data_parameters_list[e] = DotMap(data_parameters_list[e])

visits_files = {
    0: {i: ['Baseline', '12 months', '24 months', '36 months'] for i in files_of_interest},
    1: {i: ['Baseline', '12 months', '24 months', '36 months'] for i in files_of_interest}
}

logger.info('Data Started Loading')
preprocessed_data = fe.create_data(data_parameters_list, data_names, files_of_interest, visits_files, ldpc=ldp)
with open(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'preprocessed/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(preprocessed_data, f)
logger.info('Data Done Loading')
input_data = copy.deepcopy(preprocessed_data)

logger.info('Representation Learning Started')
representation_learning_data = rl.perform_dimensionality_reduction(preprocessed_data, rlm)
with open(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'representation_learning/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(representation_learning_data, f)
logger.info('Representation Learning Started')

logger.info('Clustering Started from PPMI')
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'clustering/{}.pkl'.format(save_name), 'rb') as f:
    clustering_ppmi_data = pickle.load(f)

model_gmm = {}
for i, j in clustering_ppmi_data['gmm_models'].items():
    model_gmm['gmm_model'+'$' + i] = j

for i, j in clustering_ppmi_data['gmm_models_3d'].items():
    model_gmm['gmm_model_3d'+'$' + i] = j

mapping = {'HC': 'Control', 'SWEDD': 'Case', 'PD': 'Case'}
input_data = copy.deepcopy(preprocessed_data)
input_data.update(representation_learning_data)

clustering_data = cl.perform_clustering(input_data, clm, mapping=mapping, trained_model=model_gmm)

with open(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'clustering/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(clustering_data, f)

logger.info('Clustering Done')

d_M_label = clustering_data['d_M_label']
data_names = preprocessed_data['data_names']
M_chosen = preprocessed_data['M_chosen']
X_and_Y = {}
for e, data_parameters in enumerate(data_parameters_list):
    M_label_PD = d_M_label[data_names[e]][d_M_label[data_names[e]]['GMM_2d_adj'].isin(['PD_h', 'PD_m', 'PD_l', 'Control'])].rename(columns={'GMM_2d_adj': 'GMM'})[['GMM']]
    X = M_chosen[data_names[e]].loc[M_label_PD.index, :]
    F = pd.merge(X, M_label_PD, left_index=True, right_index=True)
    temp = [tuple(list(i) + ['numerical']) for i in list(F.columns)]
    temp[-1] = ('label', 'GMM', 'category')
    F.columns = pd.MultiIndex.from_tuples(temp)
    X_and_Y[data_names[e]] = F.copy()

with open(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(X_and_Y, f)

experiment_name = data_names[0]
full_dataframe = X_and_Y[experiment_name].copy()

id_list = full_dataframe['label']['GMM']['category'].map(lambda x: 'control' if x=='HC' else 'case' )
id_list.index = id_list.index.map(lambda x: 'PP-' + str(x))
id_list_zip = list(zip( id_list, list(id_list.index)))
class_labels = {
    'Control': 0,
    'PD_h': 1,
    'PD_l': 2,
    'PD_m': 3,
}

os.makedirs(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE/ 'prediction/{}'.format(save_name), exist_ok=True)
to_add = np.array(list(full_dataframe['label']['GMM']['category'].map(lambda x: class_labels[x]))).reshape(-1, 1)
full_dataframe = full_dataframe.drop(columns='label')
numpy_data = np.concatenate([to_add, np.array(full_dataframe)], axis=1)


# import sys; sys.exit()
for dim_replication in [4,6,8,10,20]:
    # latent_weight_10dim = input_data['nmf4_dr_weights']['{}_BL'.format(experiment_name)+'_'+str(dim_replication)].iloc[:, :-1]
    latent_weight_10dim = input_data['nmf4_dr_weights']['{}_Baseline'.format(experiment_name)+'_'+str(dim_replication)].iloc[:, :-1]
    temp_experiment_name = 'replication#'+experiment_name+'#'+str(dim_replication)
    
    latent_weight_10dim = latent_weight_10dim.loc[full_dataframe.index]
    dims = latent_weight_10dim.shape[-1]
    column_list = [
        ('Baseline', i, 'numerical') for i in latent_weight_10dim.columns]
    numpy_data = np.concatenate([to_add, np.array(latent_weight_10dim)], axis=1)
    
    with open( PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_column_multi_index.csv'.format(save_name, temp_experiment_name), 'w') as f:
        f.write('\n'.join([','.join(list(i)) for i in [['label', 'target', 'category']] + list(column_list)]))
    
    with open(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_id_multi_index.csv'.format(save_name, temp_experiment_name), 'w') as f:
        f.write('\n'.join([','.join(list(i)) for i in id_list_zip]))
    
    np.savetxt(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_numpy_array.gz'.format(save_name, temp_experiment_name), numpy_data, delimiter=',')
    
    with open(PDBP_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_class_labels.pkl'.format(save_name, temp_experiment_name), 'wb') as the_file:
        pickle.dump(class_labels, the_file)
# P = input_data['P_d'][data_names[0]].copy().set_index('attribute')
# print (P.head(30).to_markdown())
# fc = open('table.txt', 'w')
# fc.write(P.to_markdown())
# fc.close()
# fc = open('table.txt', 'w')
# fc.write(P.head(130).to_markdown())
# fc.close()