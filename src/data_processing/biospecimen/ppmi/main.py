import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np

from collections import defaultdict
from logzero import logger
from dotmap import DotMap


# files_of_interest = ['vital_signs', 'family_history',
#                     'socio', 'screening', 'genetics', 'apoe_genetics',
#                     'csf_hemoglobin', 'csf_alpha_syn', 'csf_total_tau', 'csf_abeta_42', 'csf_p_tau181p', 'dna_grs']
# files_of_interest += list(map(lambda x: 'csf_' + x, csf_feat)) + list(map(lambda x: 'plasma_' + x, plasma_feat))
# files_of_interest += list(map(lambda x: 'serum_' + x, serum_feat))
# files_of_interest += list(map(lambda x: 'whole_blood_' + x, whole_blood_feat))
# files_of_interest += list(map(lambda x: 'urine_' + x, urine_feat))

import copy
from src.data_processing.biospecimen.ppmi import utils

def create_data(data, Labels, files_of_interest, ldpb, visit_list=['BL'], join_type='inner', **args):
    feats_involving_time = []
    feats_not_involving_time = []
    for i in files_of_interest:
        if 'EVENT_ID' in list(data[i].reset_index().columns):
            feats_involving_time.append(i)
        else:
            feats_not_involving_time.append(i)

    convert_to_time = ldpb.convert_to_timeseq(data, feats_involving_time, visit_list,
                                              False)  # ['BL', 'V02', 'V03', 'V04', 'V06', 'V08'])
    data_no_time = {i: j for i, j in data.items() if i in feats_not_involving_time}
    M_no_time = pd.concat([data_no_time[i] for i in feats_not_involving_time if i in files_of_interest], axis=1,
                          join=join_type)
    M_time = pd.concat([convert_to_time[i] for i in feats_involving_time if i in files_of_interest], axis=1,
                       join=join_type)
    data_dict = copy.copy(data_no_time)
    data_dict.update(convert_to_time)
    M_time.columns = [i + '-' + j for i, j in M_time.columns]
    full_M = pd.merge(M_time, M_no_time, left_index=True, right_index=True, how=join_type)
    full_withLabels = pd.merge(full_M, Labels, left_index=True, right_index=True, how='inner')
    return {'full_withLabels': full_withLabels, 'feats_involving_time': feats_involving_time,
            'feats_not_involving_time': feats_not_involving_time}, data_dict


def delete_features(full_withLabels, remove_features, **args):
    for col in full_withLabels.columns:
        if len(full_withLabels[col].value_counts().dropna()) == 1:
            remove_features.append(col)
    selected_features = [i for i in list(full_withLabels.columns) if not i in remove_features]
    M_after_removal1 = full_withLabels[selected_features]
    return {'M_after_removal1': M_after_removal1.copy()}


def perform_mean(M_after_removal1, mean_features, categorical_features, **args):
    M_after_removal1.loc[:, mean_features] = M_after_removal1.loc[:, mean_features].fillna(
        M_after_removal1.loc[:, mean_features].mean())
    M_after_removal2 = M_after_removal1.copy()
    for catfeat in categorical_features:
        max_val = M_after_removal2[catfeat].value_counts().idxmax()
        M_after_removal2[catfeat] = M_after_removal2[catfeat].fillna(max_val)
    M_cleaned = M_after_removal2.dropna()
    return {'M_after_removal2': M_after_removal1.copy(), 'M_after_removal3': M_after_removal2.copy(),
            'M_cleaned': M_cleaned.copy()}


def perform_onehot_encoding(M_cleaned, one_hot_features, numerical_features, **args):
    converted_M = M_cleaned.copy()
    for col in one_hot_features:
        converted_M[col] = converted_M[col].map(lambda x: col + '_' + str(x))
    convert_dict = {}
    for i in one_hot_features:
        convert_dict[i] = "category"
    for i in numerical_features:
        convert_dict[i] = float
    converted_M = converted_M.astype(convert_dict)
    return {'converted_M': converted_M.copy()}


def perform_normalization(converted_M, **args):
    normalized_X = converted_M.copy()
    encoded_not_normalized = converted_M.copy()
    for col in normalized_X.columns:
        if col == 'GMM':
            continue
        if normalized_X[col].dtypes == 'float':
            normalized_X[col] = utils.convert_normalize_inplace(normalized_X, col, 'min_max')
        elif normalized_X[col].dtype.name == 'category':
            normalized_X = utils.convert_onehot_inplace(normalized_X, col)
            encoded_not_normalized = utils.convert_onehot_inplace(encoded_not_normalized, col)
        else:
            print(col, 'Please Check Category')
    return {'encoded_not_normalized': encoded_not_normalized,
            'normalized_X': normalized_X}

if __name__ == '__main__':
    import import_ipynb
    visits_of_interest = ['BL', 'V04', 'V06', 'V08', 'V10', 'V12']
    csf_old_feat = ['abeta 1-42', 'ttau', 'ptau', 'csf hemoglobin', 'csf alpha-synuclein']
    csf_feat = ['c24:1 cer', 'c20 gl2', 'c24 glccer', 'total sm', 'c22 gl2', 'c24:1 glccer', 'c22 cer', 'c16 cer', 'c23 glccer', 'c18 cer', 'c16 glccer', 'c20 sm', 'c22 sm', 'c24 gl2', 'c20 glccer', 'total glccer', 'c22 glccer', 'total gl2', 'c20 cer', 'c24 sm', 'c18 glccer', 'c23 sm', 'c16 gl2', 'total cer', 'c16 sm', 'c23 cer', 'c23 gl2', 'c18 sm', 'c24 cer', 'c18 gl2', 'c24:1 gl2', 'c24:1 sm']
    plasma_feat = ['c20 cer', 'c24:1 glccer', 'c16 glccer', 'c22 sm', 'c24 gl2', 'c24:1 sm', 'total gl2', 'c20 sm', 'c24:1 cer', 'c22 cer', 'c22 gl2', 'total sm', 'c24 glccer', 'c20 gl2', 'c23 glccer', 'c24 cer', 'c18 sm', 'c18 cer', 'c16 cer', 'c23 gl2', 'c23 sm', 'c20 glccer', 'total ceramide', 'c22 glccer', 'total glccer', 'c24:1 gl2', 'c18 gl2', 'c24 sm', 'c23 cer', 'c16 sm', 'c18 glccer', 'c16 gl2']
    serum_feat = ['nfl']
    whole_blood_feat = ['gcase activity']
    urine_feat = ['total di-22:6-bmp', "2,2' di-22:6-bmp", 'total di-18:1-bmp']
    from src.data_processing.biospecimen.ppmi import load_data as ldpb
    with open('../data_produced/preprocessed_data/biospecimen_path_data.pkl', 'rb') as f:
        input_data = pickle.load(f)