import src.data_processing.biospecimen.ppmi.load_data as ldpb
import src.data_processing.biospecimen.ppmi.main as fe_meb
from definitions import ROOT_DIR_INSIDE
import os
from pathlib import Path
PPMI_CLINICAL_GEN_DATA_DIR_INSIDE = ROOT_DIR_INSIDE / 'generated_data/clinical/ppmi'
PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE = ROOT_DIR_INSIDE / 'generated_data/biospecimen/ppmi'
os.makedirs(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE / 'clustering', exist_ok=True)
os.makedirs(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE / 'prediction', exist_ok=True)
os.makedirs(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE / 'preprocessed', exist_ok=True)
os.makedirs(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE / 'representation_learning', exist_ok=True)

from dotmap import DotMap
import copy
import pickle
import pandas as pd
import numpy as np
import os
import sys

visits_of_interest = ['BL', 'V04', 'V06', 'V08', 'V10', 'V12']
csf_old_feat = ['abeta 1-42', 'ttau', 'ptau', 'csf hemoglobin', 'csf alpha-synuclein']
csf_feat = ['c24:1 cer', 'c20 gl2', 'c24 glccer', 'total sm', 'c22 gl2', 'c24:1 glccer', 'c22 cer', 'c16 cer', 'c23 glccer', 'c18 cer', 'c16 glccer', 'c20 sm', 'c22 sm', 'c24 gl2', 'c20 glccer', 'total glccer', 'c22 glccer', 'total gl2', 'c20 cer', 'c24 sm', 'c18 glccer', 'c23 sm', 'c16 gl2', 'total cer', 'c16 sm', 'c23 cer', 'c23 gl2', 'c18 sm', 'c24 cer', 'c18 gl2', 'c24:1 gl2', 'c24:1 sm']
plasma_feat = ['c20 cer', 'c24:1 glccer', 'c16 glccer', 'c22 sm', 'c24 gl2', 'c24:1 sm', 'total gl2', 'c20 sm', 'c24:1 cer', 'c22 cer', 'c22 gl2', 'total sm', 'c24 glccer', 'c20 gl2', 'c23 glccer', 'c24 cer', 'c18 sm', 'c18 cer', 'c16 cer', 'c23 gl2', 'c23 sm', 'c20 glccer', 'total ceramide', 'c22 glccer', 'total glccer', 'c24:1 gl2', 'c18 gl2', 'c24 sm', 'c23 cer', 'c16 sm', 'c18 glccer', 'c16 gl2']
serum_feat = ['nfl']
whole_blood_feat = ['gcase activity']
urine_feat = ['total di-22:6-bmp', "2,2' di-22:6-bmp", 'total di-18:1-bmp']
files_of_interest = ['vital_signs', 'family_history',
                     'socio', 'screening', 'genetics', 'apoe_genetics',
                     'csf_hemoglobin', 'csf_alpha_syn', 'csf_total_tau', 'csf_abeta_42', 'csf_p_tau181p', 'dna_grs']
files_of_interest += list(map(lambda x: 'serum_' + x, serum_feat))
files_of_interest += list(map(lambda x: 'urine_' + x, urine_feat))
data_parameters = [{'files_of_interest': copy.copy(files_of_interest)}]

files_of_interest += list(map(lambda x: 'csf_' + x, csf_feat)) + list(map(lambda x: 'plasma_' + x, plasma_feat))
files_of_interest += list(map(lambda x: 'whole_blood_' + x, whole_blood_feat))
data_parameters.append({'files_of_interest': copy.copy(files_of_interest)})
data_names = ['half', 'full']
# data_names = ['$'.join(i['files_of_interest'])[-50:] for i in data_parameters]

old_data, csf_data, plasma_data, serum_data, urine_data, whole_blood_data = ldpb.read_from_raw_files('all_data', csf_feat, plasma_feat, serum_feat, urine_feat, whole_blood_feat)

data = copy.deepcopy(old_data)
data.update(csf_data)
data.update(plasma_data)
data.update(serum_data)
data.update(urine_data)
data.update(whole_blood_data)

unique_name = sys.argv[1]
experiment_name = 'paper_experiment_flip_outlier'

os.makedirs(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}'.format(unique_name, experiment_name), exist_ok=True)
os.makedirs(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'preprocessed/{}'.format(unique_name), exist_ok=True)

with open(PPMI_CLINICAL_GEN_DATA_DIR_INSIDE / 'clustering/{}.pkl'.format(unique_name), 'rb') as f:
    clustering_data = pickle.load(f)

Labels = clustering_data['d_M_PD_HC_gmm_chosen'][experiment_name].rename(columns={'GMM_2d_adj': 'GMM'})['GMM']
full_data = {}
data_dict = {}
input_data = {}
for e, data_parameter in enumerate(data_parameters):
    input_data[data_names[e]] = {'data': copy.deepcopy(data), 'Labels': Labels}
    input_data[data_names[e]]['files_of_interest'] = data_parameter['files_of_interest']
    temp, data_dict[data_names[e]] = fe_meb.create_data(**input_data[data_names[e]], join_type='outer',ldpb=ldpb)
    input_data[data_names[e]].update(temp)


for e, data_name in enumerate(data_names):
    remove_features = ['csf_hemoglobin-BL']#, 'HAFSIBPD'] + ['FULSIBPD', 'MATAUPD', 'PATAUPD', 'KIDSPD']
    input_data[data_name].update({'remove_features': remove_features})
    input_data[data_name].update(fe_meb.delete_features(**input_data[data_name]))

mean_features = ['WGTKG-BL', 'HTCM-BL', 'csf_alpha_syn-BL', 'csf_total_tau-BL', 'csf_abeta_42-BL', 'csf_p_tau181p-BL', 'dna_grs-BL']
mean_features += ['HRSTND-BL', 'DIASTND-BL', 'SYSSTND-BL', 'HRSUP-BL', 'DIASUP-BL', 'SYSSUP-BL', 'TEMPC-BL', 'KIDSNUM', 'HAFSIB', 'FULSIB', 'PATAU', 'MATAU']
new_feats = list(map(lambda x: 'csf_' + x + '-BL', csf_feat))
new_feats += list(map(lambda x: 'plasma_' + x+ '-BL', plasma_feat))
new_feats += list(map(lambda x: 'serum_' + x+ '-BL', serum_feat))
new_feats += list(map(lambda x: 'whole_blood_' + x+ '-BL', whole_blood_feat))
new_feats += list(map(lambda x: 'urine_' + x+ '-BL', urine_feat))
mean_features = mean_features + new_feats
categorical_features = ['PAGPARPD', 'MAGPARPD', 'BIOMOMPD', 'apoe_genetics', 'BIODADPD']
categorical_features += ['BIOMOMPD', 'FULSIBPD', 'MATAUPD', 'PATAUPD']

temp = {}
temp1 = {}
temp2 = {}
for e, data_name in enumerate(data_names):
    input_data[data_name]['mean_features'] = list(set(mean_features).intersection(set(list(input_data[data_name]['M_after_removal1'].columns))))
    input_data[data_name]['categorical_features'] = list(set(categorical_features).intersection(set(list(input_data[data_name]['M_after_removal1'].columns))))
    input_data[data_name].update(fe_meb.perform_mean(**input_data[data_name]))
    one_hot_features = []
    numerical_features = []
    for col in input_data[data_name]["M_cleaned"].columns:
        if 'cjkjkjkjkjkr' in col:
        # if ':' in col:
            numerical_features.append(col)
        if not 'cjkjkjkjkjr' in col and (not col == 'GMM'):
            if (not col == 'csf_c24 gl2-BL') and len( input_data[data_name]["M_cleaned"][col].unique()) < 7:
                # print (col, len(M_cleaned[col].unique()), max(M_cleaned[col]), min(M_cleaned[col]), end=' ')
                one_hot_features.append(col)
            else:
                numerical_features.append(col)
    # import pdb; pdb.set_trace()
    input_data[data_name]['one_hot_features'] = one_hot_features
    input_data[data_name]['numerical_features'] = numerical_features
    input_data[data_name].update(fe_meb.perform_onehot_encoding(**input_data[data_name]))
    input_data[data_name].update(fe_meb.perform_normalization(**input_data[data_name]))
    temp[data_name] = input_data[data_name]["M_cleaned"].copy()
    temp1[data_name] = input_data[data_name]["normalized_X"].copy()
    temp2[data_name] = input_data[data_name]["encoded_not_normalized"].copy()


for e, data_name in enumerate(data_names):
    column_indexes = []
    for col in list(temp1[data_name].columns):
        if col[:3] == 'chr' :
            # column_indexes.append(('genetics', col, 'numerical'))
            column_indexes.append(('genetics', col, 'frequency'))
        elif col[:3] == 'dna' or col[:3] == 'apoe':
            column_indexes.append(('genetics', col, 'numerical'))
        else:
            if not col == 'GMM':
                column_indexes.append(('biospecimen', col, 'numerical'))
            else:
                column_indexes.append(('label', col, 'category'))

    input_data[data_name]["normalized_X"].columns = pd.MultiIndex.from_tuples(column_indexes,
                                                                              names=['feature groups', 'feature',
                                                                                     'type'])
    input_data[data_name]["encoded_not_normalized"].columns = pd.MultiIndex.from_tuples(column_indexes,
                                                                                        names=['feature groups',
                                                                                               'feature', 'type'])

for e, data_name in enumerate(data_names):
    column_indexes = [[], [], []]
    for col in list(temp[data_name].columns):
        if col in input_data[data_name]['one_hot_features']:
            ftpe = 'category'
        else:
            ftpe = 'numerical'
        if col[:3] == 'chr':
            column_indexes[0].append('genetics')
            column_indexes[1].append(col)
            # column_indexes[2].append(ftpe)
            column_indexes[2].append('frequency')
        elif col[:3] == 'dna' or col[:4] == 'apoe':
            column_indexes[0].append('genetics')
            column_indexes[1].append(col)
            column_indexes[2].append(ftpe)
        else:
            if not col=='GMM':
                column_indexes[0].append('biospecimen')
                column_indexes[1].append(col)
                column_indexes[2].append(ftpe)
            else:
                column_indexes[0].append('label')
                column_indexes[1].append(col)
                column_indexes[2].append('category')
    # column_indexes[1] = list(map(lambda x: x.replace(' ', ''), column_indexes[1]))
    temp[data_name].columns = pd.MultiIndex.from_arrays(column_indexes, names=['feature groups', 'feature', 'type'])
    input_data[data_name]["M_cleaned"] = temp[data_name].copy()

with open(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE / 'preprocessed/{}/{}.pkl'.format(unique_name, experiment_name), 'wb') as f:
    pickle.dump(input_data, f)

for e, data_name in enumerate(data_names):
    X_and_Y = {'M_cleaned': input_data[data_name]['M_cleaned'].copy(), 'normalized_X': input_data[data_name]["normalized_X"].copy(), 'encoded_not_normalized': input_data[data_name]["encoded_not_normalized"].copy()}
    with open(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}/{}_X_and_Y.pkl'.format(unique_name, experiment_name, data_name), 'wb') as f:
        pickle.dump(X_and_Y, f)
    M_cleaned = input_data[data_name]['M_cleaned']
    encoded_not_normalized = input_data[data_name]["encoded_not_normalized"]
    normalized_X = input_data[data_name]["normalized_X"]
    full_dataframe = encoded_not_normalized.copy()
    full_dataframe = M_cleaned.copy()
    give_name = ''
    # give_name = '_cleaned'
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
    with open(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}/{}_column_multi_index.csv'.format(unique_name, experiment_name, data_name+give_name), 'w') as f:
        f.write('\n'.join(['$'.join(list(i)).replace(',', "").replace('$', ",") for i in
                       [['label', 'target', 'category']] + list(full_dataframe.columns)]))
    with open(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}/{}_id_multi_index.csv'.format(unique_name, experiment_name, data_name+give_name), 'w') as f:
        f.write('\n'.join([','.join(list(i)) for i in id_list_zip]))
    # np.savetxt(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}/{}_numpy_array.gz'.format(unique_name, experiment_name, data_name+give_name), numpy_data, delimiter=',')
    import pickle
    with open(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}/{}_numpy_array.pkl'.format(unique_name, experiment_name, data_name+give_name), 'wb') as handle:
        pickle.dump(numpy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # np.save(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}/{}_numpy_array.npy'.format(unique_name, experiment_name, data_name+give_name), numpy_data)
    with open(PPMI_BIOSPECIMEN_GEN_DATA_DIR_INSIDE/ 'prediction/{}/{}/{}_class_labels.pkl'.format(unique_name, experiment_name, data_name+give_name), 'wb') as the_file:
        pickle.dump(class_labels, the_file)

# from collections import defaultdict
# import pandas as pd
# dset = data_names[0]
# df_dict = defaultdict(list)
# inital_patients = set(Labels.index)
# current_patients = set(Labels.index)
# # print (len(inital_patients))
# for key in data_dict[dset]:
#     # if key[:5] == 'urine' or key[:5]=='csf_c' or key[:6]=='plasma' or key[:5]=='whole' or key[:10]=='csf_total ':
#     #    continue
#     inital_patients = set(data_dict[dset][key].index).intersection(inital_patients)
#     df_dict['feature'].append(key)
#     df_dict['columns'].append(len(data_dict[dset][key].columns))
#     df_dict['till_now'].append(len(inital_patients))
#     df_dict['initial'].append(len(set(data_dict[dset][key].index).intersection(current_patients)))
# df = pd.DataFrame(df_dict)
# df['difference'] = len(current_patients) - df['initial']
# df = df.sort_values(by='difference')
# l = list(df['feature'])
# import textwrap
# for i in [100]:#[5, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 30, 100]:
#     df_dict = defaultdict(list)
#     inital_patients = set(Labels.index)
#     current_patients = set(Labels.index)
#     for key in l[:i]:
#         # if key[:5] == 'urine' or key[:5]=='csf_c' or key[:6]=='plasma' or key[:5]=='whole' or key[:10]=='csf_total ':
#         #    continue
#         inital_patients = set(data_dict[dset][key].index).intersection(inital_patients)
#         if key == 'genetics':
#             continue
#         df_dict['columns'].append(key)
#         df_dict['feature'].append(key)
#         df_dict['till_now'].append(len(inital_patients))
#         df_dict['initial'].append(len(set(data_dict[dset][key].index).intersection(current_patients)))
#         df_dict['zmeasurments'].append(textwrap.fill(str(list(data_dict[dset][key].columns)), 40))
#     df = pd.DataFrame(df_dict)
#     df['difference'] = len(current_patients) - df['initial']
#     df = df.sort_values(by='difference').reset_index(drop=True)
#     # print (i, 'Features', df['columns'].sum(), len(inital_patients))
# # print (df.to_markdown())

