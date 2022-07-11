import pandas as pd
import pickle
import numpy as np
import os
from MachineLearningStreamlitBase.train_model import generateFullData

# Read Data File both pre-processed and raw data

with open("data/paper_experiment_flip_outlier_numpy_array.pkl", "rb") as the_handle:
    numpy_array = pickle.load(the_handle)

with open("data/paper_experiment_flip_outlier_class_labels.pkl", "rb") as the_handle:
    Z_map = pickle.load(the_handle)

map_Z = {j:i for i, j in Z_map.items()}
column_index = pd.read_csv("data/paper_experiment_flip_outlier_column_multi_index.csv", header=None)
row_index = pd.read_csv("data/paper_experiment_flip_outlier_id_multi_index.csv", header=None)
original_data = pd.DataFrame(numpy_array, index=row_index, columns=column_index)
original_data.columns = ['_'.join(col) for col in original_data.columns]
original_data.index = ['_'.join(col) for col in original_data.index]
original_data['label_target_category'] = original_data['label_target_category'].map(map_Z)
original_data = original_data.reset_index()
original_data = original_data[original_data['index'].str.contains('case')]
original_data['label_target_category'] = original_data['label_target_category'].map(lambda x: 'PD_h' if x == 'PD_m' else x)
# original_data = pd.read_csv("data/scriptToWrangleJessicaDataFreeze5/ALSregistry.AdrianoChio.wrangled.nodates.freeze5.csv")
# replication_data = pd.read_csv("data/scriptToWrangleJessicaDataFreeze5/ALSregistry.JessicaMandrioli.wrangled.nodates.freeze5.csv")
# replication_data = original_data.copy()
# Z_map = {'PD_h': 0, 'PD_l': 1, 'PD_m': 2}
Z_map = {'PD_h': 1, 'PD_l': 0}
# select top columns
# Select Feature and Label Column

selected_cols_x_top = [col for col in original_data.columns if ('BL_' in col and '#BL' in col) or ('V04_' in col and '#V04' in col)]

with open("feature_list.txt", 'r') as f:
    selected_cols_x_top = f.read().strip().split('\n')



# with open("feature_list.txt", 'w') as f:
#    f.write("\n".join(selected_cols_x_top))


# with open("feature_list.txt", 'w') as f:
#    f.write("\n".join(model.get_score(importance_type='total_gain')))



selected_cols_y = "label_target_category"
# mapping class to index
obj = generateFullData()
# print unique value counts for each selected feature

for col in original_data[selected_cols_x_top].columns:
    print ('*'*50, col)
    print (len(original_data[col].value_counts()))

original_encoded_data = original_data[['index', selected_cols_y]].copy().rename(columns={'index': 'ID'})
# replication_encoded_data = replication_data[['index', selected_cols_y]].copy().rename(columns={'index': 'ID'})
categorical_variable = [
    'smoker', 'elEscorialAtDx', 'anatomicalLevel_at_onset', 'site_of_onset',
    'onset_side', 'ALSFRS1'
]
numerical_variable = [
    "FVCPercentAtDx", "weightAtDx_kg", "rateOfDeclineBMI_per_month", "age_at_onset", "firstALSFRS_daysIntoIllness"
]
categorical_variable = []
numerical_variable = selected_cols_x_top
col_dict_map = {}
for col in categorical_variable:
    distinct_vals = original_data[col].dropna().unique()
    dict_map = {i:e for e, i in enumerate(list(distinct_vals))}
    col_dict_map[col] = dict_map
    mode = original_data[col].dropna().index[0]
    for val in distinct_vals:
        # temp = original_data[col].fillna(mode)
        # original_encoded_data['{}_{}'.format(col, val)] = pd.Series(temp == val).astype(int)
        original_encoded_data[col] = original_data[col].map(lambda x: dict_map.get(x, np.nan))
        # temp = replication_data[col].fillna(mode)
        # replication_encoded_data['{}_{}'.format(col, val)] = pd.Series(temp == val).astype(int)
        # replication_encoded_data[col] = replication_data[col].map(lambda x: dict_map.get(x, np.nan))


for col in numerical_variable:
    original_encoded_data[col] = list(original_data[col])
    # replication_encoded_data[col] = list(replication_data[col])

original_encoded_data = original_encoded_data[original_encoded_data[selected_cols_y].notna()]
# replication_encoded_data = replication_encoded_data[replication_encoded_data[selected_cols_y].notna()]
original_encoded_data[selected_cols_y] = original_encoded_data[selected_cols_y].map(lambda x: Z_map[x])
# replication_encoded_data[selected_cols_y] = replication_encoded_data[selected_cols_y].map(lambda x: Z_map[x])
selected_raw_columns = [col for col in original_encoded_data.columns if not col in [selected_cols_y, 'ID'] ]
# _ = obj.trainXGBModel_multiclass(data=original_encoded_data, feature_names=selected_raw_columns, label_name=selected_cols_y, replication_set=replication_encoded_data)
# _ = obj.trainLightGBMModel_multiclass(data=original_encoded_data, feature_names=selected_raw_columns, label_name=selected_cols_y, replication_set=replication_encoded_data)

# import sys; sys.exit()
os.makedirs('saved_models', exist_ok=True)
for class_name, ind in Z_map.items():
    print ('*'*30, class_name, '*'*30)
    original_encoded_data_temp = original_encoded_data.copy()
    # replication_encoded_data_temp = replication_encoded_data.copy()
    original_encoded_data_temp[selected_cols_y] = original_encoded_data_temp[selected_cols_y].map (lambda x: 1 if x==ind else 0)
    # replication_encoded_data_temp[selected_cols_y] = replication_encoded_data_temp[selected_cols_y].map (lambda x: 1 if x==ind else 0)
    # data_pol = pd.concat([data, data_rep], axis=0)
    model, train = obj.trainXGBModel_binaryclass(data=original_encoded_data_temp, feature_names=selected_raw_columns, label_name=selected_cols_y, replication_set=None)
    with open('saved_models/trainXGB_gpu_{}.data'.format(class_name), 'wb') as f:
        pickle.dump(train, f)
    import joblib
    joblib.dump( model, 'saved_models/trainXGB_gpu_{}.model'.format(class_name) )
    # with open('saved_models/trainXGB_gpu_{}.model'.format(class_name), 'wb') as f:
    #     pickle.dump(model, f)

result_aucs = {}
for class_name in Z_map:
    with open('saved_models/trainXGB_gpu_{}.data'.format(class_name), 'rb') as f:
        temp = pickle.load(f)
    result_aucs[class_name] = (temp[3]['AUC_train'], temp[3]['AUC_test'])
    print (class_name, result_aucs[class_name])

with open('saved_models/trainXGB_gpu.aucs', 'wb') as f:
    pickle.dump(result_aucs, f)

with open('saved_models/trainXGB_categorical_map.pkl', 'wb') as f:
    pickle.dump(col_dict_map, f)

with open('saved_models/trainXGB_class_map.pkl', 'wb') as f:
    pickle.dump(Z_map, f)
