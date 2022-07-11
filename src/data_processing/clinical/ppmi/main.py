import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from logzero import logger
from collections import defaultdict
from dotmap import DotMap

def create_data(data_parameters_list, data_names, files_of_interest, visits_files, ldpc):
    data = {}
    for e, data_parameters in enumerate(data_parameters_list):
        data[data_names[e]] = ldpc.read_from_raw_files_old(data_parameters.data_dir, merge_screen_baseline=data_parameters.merge_screen_baseline)
    data_visits = {}
    selected_patients = {}
    Labels = {}
    P_d = {}
    for e, data_parameters in enumerate(data_parameters_list):
        data_visits[data_names[e]], selected_patients[data_names[e]] = ldpc.load_required_files(data[data_names[e]], files_of_interest, data_parameters.visits_list, remove_outlier=data_parameters.remove_outlier)
        Labels[data_names[e]] = data_visits[data_names[e]]["info"].copy()
        ECAT = Labels[data_names[e]][['ENROLL_CAT']]
        data_visits[data_names[e]], P_d[data_names[e]] = ldpc.flipping_data(data_visits[data_names[e]], ECAT, data_parameters.files_of_interest, visit_id='V12', fil=data_parameters.flipping)

    convert_to_time = {}
    for e, data_parameters in enumerate(data_parameters_list):
        collection_time = {i: list(set(j).intersection(set(data_parameters.visits_list))) for i, j in visits_files[data_parameters.selection_availability].items()}
        convert_to_time[data_names[e]] = ldpc.convert_to_timeseq(data_visits[data_names[e]], collection_time)

    M_chosen = {}
    cleaned_data = {}
    cleaned_data_no_interpolate = {}
    M_orig = {}
    M_orig_outlier = {}
    M_zs = {}
    for e, data_parameters in enumerate(data_parameters_list):
        cleaned_data[data_names[e]] = {}
        cleaned_data_no_interpolate[data_names[e]] = {}
        for i, j in convert_to_time[data_names[e]].items():
            cleaned_data[data_names[e]][i] = j.stack(level=0).interpolate(method='linear', axis=1, limit_direction='both').unstack()
            cleaned_data_no_interpolate[data_names[e]][i] = j.stack(level=0).unstack()
            # cleaned_data[data_names[e]][i] = cleaned_data[data_names[e]][i].dropna()
            # import pdb; pdb.set_trace()
            # print (i, cleaned_data[data_names[e]][i].isna().sum().sum())
            # import pdb; pdb.set_trace()
        P = P_d[data_names[e]]
        reverse_feature = list(P[P['reverse']==True]['attribute'])
        M_chosen[data_names[e]], _, M_orig_outlier[data_names[e]] , _ = ldpc.create_matrix_normalize_form(cleaned_data[data_names[e]], data_parameters.files_of_interest, type='minmax', reverse_feature=reverse_feature, labels=Labels[data_names[e]], remove_outlier=data_parameters.remove_outlier)
        _ , M_orig[data_names[e]], _, M_zs[data_names[e]] = ldpc.create_matrix_normalize_form(cleaned_data_no_interpolate[data_names[e]], data_parameters.files_of_interest, type='minmax', reverse_feature=reverse_feature, labels=Labels[data_names[e]])

    return_data = {}
    return_data['P_d'] = P_d
    return_data["M_chosen"] = M_chosen
    return_data["M_zs"] = M_zs
    return_data["M_orig"] = M_orig
    return_data["M_orig_outlier"] = M_orig_outlier
    return_data["Labels"] = Labels
    return_data["convert_to_time"] = convert_to_time
    return_data['data'] = data
    return_data['data_parameters_list'] = data_parameters_list
    return_data['data_names'] = data_names
    return_data['files_of_interest'] = files_of_interest
    return_data["selected_patients"] = selected_patients
    return_data['data_visits'] = data_visits
    return return_data


if __name__ == "__main__":
    from src.data_processing.clinical.ppmi import load_data as ldpc

    # input parameters
    data_parameters_list = [
        {
            'visits_list': ['BL', 'V04', 'V06', 'V08', 'V10', 'V12'],
            'data_dir': 'raw_data/clinical/ppmi/all_data',
            'preprocessing_type': 'new',
            'selection_availability': 0,
            'merge_screen_baseline': ['neuro_cranial', 'moca'],
            'flipping': 0,
        },
    ]

    data_names = []
    for data_parameters in data_parameters_list:
        temp = '_'.join(data_parameters['visits_list'])
        temp += '_' + data_parameters['data_dir']
        temp += '_' + data_parameters['preprocessing_type']
        temp += '_' + str(data_parameters['selection_availability'])
        temp += '_'.join(data_parameters['merge_screen_baseline'])
        data_names.append(temp)

    for e in range(len(data_parameters_list)):
        data_parameters_list[e] = DotMap(data_parameters_list[e])

    files_of_interest = ['neuro_cranial', 'updrs1', 'updrs1pq', 'updrs2pq', 'updrs3', 'benton', 'epworth', 'geriatric',
                         'hopkins_verbal', 'letter_seq', 'moca', 'quip', 'rem', 'aut', 'semantic', 'stai', 'sdm']
    visits_files = {
            0: {i: ['BL', 'V04', 'V06', 'V08', 'V10', 'V12'] for i in files_of_interest},
            1: {i: ['BL', 'V04', 'V06', 'V08', 'V10', 'V12'] for i in files_of_interest},
        }
    return_data = create_data(data_parameters_list, data_names, files_of_interest, visits_files, ldpc=ldpc)
