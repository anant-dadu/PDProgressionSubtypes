# input parameters

from definitions import ROOT_DIR_INSIDE

import json
import os
import sys
import pandas as pd
import numpy as np
import pickle
import copy
from dotmap import DotMap

import logzero
import logging
from logzero import logger
logzero.loglevel(logging.INFO)

import src.data_processing.clinical.ppmi.main as fe
import src.data_processing.clinical.ppmi.load_data as ldpc

import src.models.representation_learning.main as rl
import src.models.representation_learning.methods as rlm

import src.models.clustering.methods as clm
import src.models.clustering.main as cl

# specify the date to save name
save_name = sys.argv[1]

# specify directories to save files
PPMI_CLINICAL_GEN_DATA_DIR_INSIDE = ROOT_DIR_INSIDE / 'generated_data/clinical/ppmi'
os.makedirs(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'clustering', exist_ok=True)
os.makedirs(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction', exist_ok=True)
os.makedirs(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'preprocessed', exist_ok=True)
os.makedirs(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'representation_learning', exist_ok=True)


files_of_interest = ['neuro_cranial', 'updrs1', 'updrs1pq', 'updrs2pq', 'updrs3', 'benton', 'epworth', 'geriatric',
                     'hopkins_verbal', 'letter_seq', 'moca', 'quip', 'rem', 'aut', 'semantic', 'stai', 'sdm']

data_parameters_list_subtype = [
{
        'visits_list': ['BL', 'V04', 'V06', 'V08', 'V10', 'V12'],
        'data_dir': 'all_data',
        'preprocessing_type': 'new',
        'selection_availability': 0,
        'merge_screen_baseline': ['neuro_cranial', 'moca'],
        'flipping': 1,
        'last_visits': ['BL', 'V04', 'V06', 'V08', 'V10', 'V12'],
        'files_of_interest': ['neuro_cranial', 'quip', 'benton', 'updrs1', 'updrs1pq', 'updrs2pq', 'updrs3', 'epworth', 'geriatric',
                     'hopkins_verbal', 'letter_seq', 'moca', 'rem', 'aut', 'semantic', 'stai', 'sdm'],
        'remove_outlier': 1,
        'name': 'paper_experiment_flip_outlier',
        'dataset': 'ppmi'
    },
]
data_parameters_list_replication = [
    {
        'visits_list': ['BL', 'V04', 'V06', 'V08'],
        'data_dir': 'all_data',
        'preprocessing_type': 'new',
        'selection_availability': 0,
        'merge_screen_baseline': ['neuro_cranial', 'moca'],
        'flipping': 1,
        'last_visits': ['BL', 'V04', 'V06', 'V08'],
        'files_of_interest': ['neuro_cranial', 'quip', 'benton', 'updrs1', 'updrs1pq', 'updrs2pq', 'updrs3', 'epworth', 'geriatric',
                     'hopkins_verbal', 'letter_seq', 'moca', 'rem', 'aut', 'semantic', 'stai', 'sdm'],
        'remove_outlier': 1,
        'name': 'paper_experiment_replication',
        'dataset': 'ppmi'
    },
]

data_parameters_list = data_parameters_list_subtype + data_parameters_list_replication
data_names = []
for data_parameters in data_parameters_list:
    data_names.append(data_parameters['name'])

for e in range(len(data_parameters_list)):
    data_parameters_list[e] = DotMap(data_parameters_list[e])

visits_files = {
    0: {i: ['BL', 'V04', 'V06', 'V08', 'V10', 'V12'] for i in files_of_interest},
    1: {i: ['BL', 'V04', 'V06', 'V08', 'V10', 'V12'] for i in files_of_interest}
}

logger.info('Data Started Loading')
preprocessed_data = fe.create_data(data_parameters_list, data_names, files_of_interest, visits_files, ldpc=ldpc)
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'preprocessed/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(preprocessed_data, f)
input_data = copy.deepcopy(preprocessed_data)
logger.info('Data Loaded')

logger.info('Representation Learning (using NMF)')
representation_learning_data = rl.perform_dimensionality_reduction(preprocessed_data, rlm)
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'representation_learning/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(representation_learning_data, f)
input_data.update(representation_learning_data)
logger.info('NMF Representation Learning done')


logger.info('Clustering Started')
clustering_data = cl.perform_clustering(input_data, clm)
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'clustering/{}.pkl'.format(save_name), 'wb') as f:
   pickle.dump(clustering_data, f)
input_data.update(clustering_data)
logger.info('Clustering Done')

logger.info('Start: Files for Predictive Models')
d_M_label = clustering_data['d_M_label']
data_names = preprocessed_data['data_names']
M_chosen = preprocessed_data['M_chosen']
X_and_Y = {}
for e, data_parameters in enumerate(data_parameters_list):
    M_label_PD = d_M_label[data_names[e]][d_M_label[data_names[e]]['GMM_2d_adj'].isin(['PD_h', 'PD_m', 'PD_l', 'HC'])].rename(columns={'GMM_2d_adj': 'GMM'})[['GMM']]
    X = M_chosen[data_names[e]].loc[M_label_PD.index, :]
    F = pd.merge(X, M_label_PD, left_index=True, right_index=True)
    temp = [tuple([i[1], i[0]+'#'+i[1]] + ['numerical']) for i in list(F.columns)]
    temp[-1] = ('label', 'GMM', 'category')
    F.columns = pd.MultiIndex.from_tuples(temp, names=['feature groups', 'feature', 'type'])
    X_and_Y[data_names[e]] = F.copy()

with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(X_and_Y, f)

experiment_name = data_names[0]
full_dataframe = X_and_Y[data_names[0]].copy()
dset = list(preprocessed_data['P_d'].keys())

id_list = full_dataframe['label']['GMM']['category'].map(lambda x: 'control' if x=='HC' else 'case' )
id_list.index = id_list.index.map(lambda x: 'PP-' + str(x))
id_list_zip = list(zip( id_list, list(id_list.index)))
class_labels = {
    'HC': 0,
    'PD_h': 1,
    'PD_l': 2,
    'PD_m': 3,
}

to_add = np.array(list(full_dataframe['label']['GMM']['category'].map(lambda x: class_labels[x]))).reshape(-1, 1)
full_dataframe = full_dataframe.drop(columns='label')
numpy_data = np.concatenate([to_add, np.array(full_dataframe)], axis=1)
P = input_data['P_d']['paper_experiment_flip_outlier']

save_name = save_name + '/subtype_prediction'
os.makedirs(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE/ 'prediction/{}'.format(save_name), exist_ok=True)

with open( PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_column_multi_index.csv'.format(save_name, experiment_name), 'w') as f:
    f.write('\n'.join([','.join(list(i)) for i in [['label', 'target', 'category']] + list(full_dataframe.columns)]))
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_id_multi_index.csv'.format(save_name, experiment_name), 'w') as f:
    f.write('\n'.join([','.join(list(i)) for i in id_list_zip]))
import pickle
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_numpy_array.pkl'.format(save_name, experiment_name), 'wb') as handle:
    pickle.dump(numpy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_class_labels.pkl'.format(save_name, experiment_name), 'wb') as the_file:
    pickle.dump(class_labels, the_file)


feature_mapping = {}
for i in range(len(P)):
    feature_mapping[P.iloc[i]['attribute']] = P.iloc[i]['feature'] 

K = list(full_dataframe.columns)
NK = []
for i,j,k in K:
    nf = feature_mapping[j.split('#')[0]]
    NK.append((i+'-'+nf, j, k))
    
with open( PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/assessment_{}_column_multi_index.csv'.format(save_name, experiment_name), 'w') as f:
    f.write('\n'.join([','.join(list(i)) for i in [['label', 'target', 'category']] + list(NK)]))
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/assessment_{}_id_multi_index.csv'.format(save_name, experiment_name), 'w') as f:
    f.write('\n'.join([','.join(list(i)) for i in id_list_zip]))

with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/assessment_{}_numpy_array.pkl'.format(save_name, experiment_name), 'wb') as handle:
    pickle.dump(numpy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/assessment_{}_class_labels.pkl'.format(save_name, experiment_name), 'wb') as the_file:
    pickle.dump(class_labels, the_file, protocol=pickle.HIGHEST_PROTOCOL)

for dim_replication in [4,6,8,10,20]:
    latent_weight_10dim = input_data['nmf4_dr_weights']['{}_BL'.format(experiment_name)+'_'+str(dim_replication)].iloc[:, :-1]
    temp_experiment_name = 'replication#'+experiment_name+'#'+str(dim_replication)
    latent_weight_10dim = latent_weight_10dim.loc[full_dataframe.index]
    dims = latent_weight_10dim.shape[-1]
    column_list = [('BL', i, 'numerical') for i in latent_weight_10dim.columns]
    numpy_data = np.concatenate([to_add, np.array(latent_weight_10dim)], axis=1)

    with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_numpy_array.pkl'.format(save_name, temp_experiment_name), 'wb') as handle:
        pickle.dump(numpy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open( PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_column_multi_index.csv'.format(save_name, temp_experiment_name), 'w') as f:
        f.write('\n'.join([','.join(list(i)) for i in [['label', 'target', 'category']] + list(column_list)]))
    
    with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_id_multi_index.csv'.format(save_name, temp_experiment_name), 'w') as f:
        f.write('\n'.join([','.join(list(i)) for i in id_list_zip]))
        
    with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'prediction/{}/{}_class_labels.pkl'.format(save_name, temp_experiment_name), 'wb') as the_file:
        pickle.dump(class_labels, the_file)

logger.info('End: Files for Predictive Models')
