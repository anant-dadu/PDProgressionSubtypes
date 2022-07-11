import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from collections import defaultdict
from dotmap import DotMap


def create_data(data_parameters_list, data_names, files_of_interest, visits_files, ldpc):
    data = {}
    for e, data_parameters in enumerate(data_parameters_list):
        # if data_parameters.preprocessing_type == 'old':
        d_fname = {}
        for f in os.listdir("{}".format(data_parameters.data_dir)):
            d_fname['_'.join(f.split('_')[:-1])] = "{}".format(data_parameters.data_dir) + f
        m = DotMap(d_fname)
        # print (m)
        data[data_names[e]] = ldpc.read_from_raw_files_old(m, data_parameters.merge_screen_baseline)
    # else:
    #    pass

    data_visits = {}
    data_visits_n = {}
    selected_patients = {}
    Labels = {}
    P_d = {}
    for e, data_parameters in enumerate(data_parameters_list):
        data_visits[data_names[e]], selected_patients[data_names[e]] = ldpc.load_required_files(data[data_names[e]],
                                                                                                files_of_interest,
                                                                                                data_parameters.visits_list)
        Labels[data_names[e]] = data_visits[data_names[e]]["info"].copy()
        ECAT = Labels[data_names[e]][['ENROLL_CAT']]
        data_visits[data_names[e]], P_d[data_names[e]] = ldpc.flipping_data(data_visits[data_names[e]], ECAT,
                                                                                files_of_interest, getP=False,
                                                                                visit_id='36 months',
                                                                                fil=data_parameters.flipping)
    # if data_parameters.flipping:
    #        data_visits[data_names[e]], P_d[data_names[e]] = ldpc.flipping_data(data_visits[data_names[e]], ECAT, files_of_interest, getP=False, visit_id='36 months')
    #    else:
    #        P_d[data_names[e]] = flipping_data(data_visits[data_names[e]], ECAT, files_of_interest, getP=True, visit_id='V12')

    convert_to_time = {}
    for e, data_parameters in enumerate(data_parameters_list):
        if data_parameters.preprocessing_type == 'old':
            convert_to_time[data_names[e]] = ldpc.convert_to_timeseq_old(data_visits[data_names[e]])
        else:
            collection_time = {i: list(set(j).intersection(set(data_parameters.visits_list))) for i, j in
                               visits_files[data_parameters.selection_availability].items()}
            convert_to_time[data_names[e]] = ldpc.convert_to_timeseq(data_visits[data_names[e]], collection_time)

    M_chosen = {}
    M_orig = {}
    cleaned_data = {}
    cleaned_data_no_interpolate = {}
    M_orig_outlier = {}
    M_zs = {}
    for e, data_parameters in enumerate(data_parameters_list):
        if data_parameters.preprocessing_type == 'old':
            temp = {};
            c = 0
            for i, j in convert_to_time[data_names[e]].items():
                temp['t{}'.format(c + 1)] = j
                c += 1
            _, M_chosen[data_names[e]], _, _ = ldpc.create_matrix_normalize_form_old(temp, files_of_interest)
        else:
            cleaned_data[data_names[e]] = {}
            cleaned_data_no_interpolate[data_names[e]] = {}
            for i, j in convert_to_time[data_names[e]].items():
                cleaned_data[data_names[e]][i] = j.stack(level=0).interpolate(method='linear', axis=1,
                                                                              limit_direction='both').unstack()
                cleaned_data_no_interpolate[data_names[e]][i] = j.copy()
                # print (i, cleaned_data[data_names[e]][i].isna().sum().sum())
                # import pdb; pdb.set_trace()
            P = P_d[data_names[e]]
            reverse_feature = list(P[P['reverse']==True]['attribute'])
            M_chosen[data_names[e]], _, M_orig_outlier[data_names[e]] , _ = ldpc.create_matrix_normalize_form(cleaned_data[data_names[e]], data_parameters.files_of_interest, type='minmax', reverse_feature=reverse_feature, labels=Labels[data_names[e]], remove_outlier=data_parameters.remove_outlier)
            _ , M_orig[data_names[e]], _, M_zs[data_names[e]] = ldpc.create_matrix_normalize_form(cleaned_data_no_interpolate[data_names[e]], data_parameters.files_of_interest, type='minmax', reverse_feature=reverse_feature, labels=Labels[data_names[e]])
    
    return_data = {}
    return_data['P_d'] = P_d
    return_data["M_chosen"] = M_chosen
    return_data["M_orig"] = M_orig
    return_data["M_orig_outlier"] = M_orig_outlier
    return_data["M_zs"] = M_zs
    return_data["Labels"] = Labels
    return_data["convert_to_time"] = convert_to_time
    return_data['data'] = data
    return_data['data_parameters_list'] = data_parameters_list
    return_data['data_names'] = data_names
    return_data['files_of_interest'] = files_of_interest
    return_data["selected_patients"] = selected_patients
    return_data['data_visits'] = data_visits
    return return_data
