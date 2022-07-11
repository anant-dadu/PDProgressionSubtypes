import pandas as pd
import numpy as np
import warnings
import copy
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
sns.set()

def get_correct_order(cluster_given, dataset='ppmi'):
    if len(cluster_given) == 2:
        if 'NHY' in cluster_given['Cognitive']:
            cluster_given['Cognitive'], cluster_given['Motor'] = cluster_given['Motor'], cluster_given['Cognitive']
        return cluster_given

    if dataset == 'ppmi':
        fname = 'delayed_recall'
        fname2 = 'NHY'
    else:
        fname = 'MOCA_DelydRecall'
        # fname = "MOCA_Total"
        fname2 = 'MDSUPDRS_PartIIIScore'
        # fname2 = 'MDSUPDRSGaitScore'
    if fname in cluster_given.get('Sleep', []):
        cluster_given['Sleep'], cluster_given['Cognitive'] = cluster_given['Cognitive'], cluster_given['Sleep']
    if fname in cluster_given['Motor']:
        cluster_given['Motor'], cluster_given['Cognitive'] = cluster_given['Cognitive'], cluster_given['Motor']
    if fname2 in cluster_given.get('Sleep', []):
        cluster_given['Sleep'], cluster_given['Motor'] = cluster_given['Motor'], cluster_given['Sleep']
    if fname2 in cluster_given['Cognitive']:
        cluster_given['Cognitive'], cluster_given['Motor'] = cluster_given['Motor'], cluster_given['Cognitive']
    if fname in cluster_given.get('Sleep', []):
        cluster_given['Sleep'], cluster_given['Cognitive'] = cluster_given['Cognitive'], cluster_given['Sleep']
    if fname in cluster_given['Motor']:
        cluster_given['Motor'], cluster_given['Cognitive'] = cluster_given['Cognitive'], cluster_given['Motor']
    return cluster_given

def get_cluster_projection_matrix(weights, components, visit_id, num_cluster=3, dataset='ppmi'):
    # import pdb; pdb.set_trace()
    weights = weights[['latent_weight{}'.format(i) for i in range(1, components.shape[0]+1)]]
    t_bcomponents = components.iloc[:, components.columns.get_level_values(1) == visit_id]
    t_bcomponents.columns = [i for i, j in t_bcomponents.columns]
    x = t_bcomponents.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    t_bcomponents = pd.DataFrame(x_scaled, columns=t_bcomponents.columns)
    X_Norm = t_bcomponents.values.transpose()  # preprocessing.normalize(bcomponents[data_names[e]])
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X_Norm)
    clustered_feats = []
    transposed = t_bcomponents.T
    for i in range(num_cluster):
        clustered_feats.append(list(set(transposed[kmeans.labels_ == i].index.values.tolist())))
    if len(clustered_feats) == 3:
        cluster_kmeans = {'Sleep': clustered_feats[0], 'Motor': clustered_feats[1], 'Cognitive': clustered_feats[2]}
        # if dataset == 'ppmi':
        #    cluster_kmeans = {'Sleep': clustered_feats[1], 'Motor': clustered_feats[0], 'Cognitive': clustered_feats[2]}
    else:
        cluster_kmeans = {'Cognitive': clustered_feats[0], 'Motor': clustered_feats[1]}
    cluster_kmeans = get_correct_order(cluster_kmeans, dataset=dataset)
    # if len(weights) == 476 and len(clustered_feats) == 3:
    #    temp = copy.copy(cluster_kmeans['Sleep'])
    #    cluster_kmeans['Sleep'] = copy.copy(cluster_kmeans['Motor'])
    #    cluster_kmeans['Motor'] = copy.copy(temp)

    # import pdb; pdb.set_trace()
    component_belong = {}
    df = {}
    for dim, dim_list in cluster_kmeans.items():
        x = components[dim_list].mean(axis=1)
        x = x.reset_index()
        x['index'] = x['index'].map(lambda x: str(x))
        x['component']= x['index'].map(lambda x: int(x))
        # adjust_weights_v12 = x[x['index'].str.contains('V12')].groupby('component').mean()
        adjust_weights = x.groupby('component').mean()
        adjust_weights = adjust_weights / adjust_weights.sum()
        # import pdb; pdb.set_trace()
        y = np.array(adjust_weights).reshape(-1) * np.array(weights)
        y = y.sum(axis=1)
        df['{} dimension'.format(dim)] = list(y)
        # import pdb; pdb.set_trace()
        component_belong['latent_weight{}'.format(1+int(adjust_weights.idxmax()[0]))] = '{} dimension'.format(dim)
    adjusted_latent_weight = pd.DataFrame(df, index=weights.index)
    # import pdb;pdb.set_trace()
    # if len(clustered_feats) == 3 and len(weights) == 476:
    #     component_belong = {'latent_weight2': 'Sleep dimension', 'latent_weight3': 'Motor dimension',
    #                    'latent_weight1': 'Cognitive dimension'}
    # if True: # not len(weights) == 476:
    #    if len(clustered_feats) == 3:
    #        mena = ['Cognitive dimension', 'Sleep dimension', 'Motor dimension'][::-1]
    #        for e, k in enumerate(weights.var().sort_values(ascending=False).index):
    #                component_belong[k] = mena[e]
    #    else:
    #        mena = ['Cognitive dimension', 'Motor dimension'][::-1]
    #        for e, k in enumerate(weights.var().sort_values(ascending=False).index):
    #            component_belong[k] = mena[e]
    # if len(clustered_feats) == 3 and len(weights) == 476:
    if False and dataset == 'pdbp':
        if len(clustered_feats) == 2:
            mena = ['Cognitive dimension', 'Motor dimension'][::-1]
            for e, k in enumerate(weights.var().sort_values(ascending=False).index):
                component_belong[k] = mena[e]


            adjusted_latent_weight = adjusted_latent_weight[adjusted_latent_weight.var().sort_values(ascending=False).index]
            adjusted_latent_weight.columns = ['Cognitive dimension', 'Motor dimension'][::-1]
        else:
            mena =  ['Cognitive dimension', 'Sleep dimension', 'Motor dimension'][::-1]
            for e, k in enumerate(weights.var().sort_values(ascending=False).index):
                component_belong[k] = mena[e]

            adjusted_latent_weight = adjusted_latent_weight[
                adjusted_latent_weight.var().sort_values(ascending=False).index]
            adjusted_latent_weight.columns =  ['Cognitive dimension', 'Sleep dimension', 'Motor dimension'][::-1]



    weights_renamed = weights.rename(columns=component_belong)
    # import pdb; pdb.set_trace()
    return cluster_kmeans, adjusted_latent_weight, weights_renamed 

def perform_dimensionality_reduction(input_data, rlm):
    data_names = input_data["data_names"]
    M_chosen = input_data["M_chosen"]
    data_parameters_list = input_data["data_parameters_list"]
    Labels = input_data["Labels"]
    return_data = {}
    dr_weights = {}
    nmf_model = {}
    nmf_dr_weights = {}
    nmf3_dr_weights = {}
    nmf3_model = {}
    nmf4_dr_weights={}
    nmf4_model={}
    cluster_final_visit_3d = {}
    cluster_final_visit_2d = {}
    for e, data_parameters in enumerate(data_parameters_list):
        for e2, last_visit in enumerate(data_parameters.last_visits):
            if e2 == (len(data_parameters.last_visits) - 1):
                st_add = ''
            else:
                st_add = '_' + last_visit
            visit_in_consider = [visit for  e_v, visit in enumerate(data_parameters.visits_list) if e_v <= data_parameters.visits_list.index(last_visit)]
            print ('Creating Progression space for {}: '.format(st_add), visit_in_consider)
            temp_M_chosen = M_chosen[data_names[e]].loc[:, M_chosen[data_names[e]].columns.get_level_values(1).isin(visit_in_consider)]
            dr_weights[data_names[e]+st_add], nmf_model[data_names[e]+st_add] = rlm.apply_dimensionality_reduction(temp_M_chosen, how={'NMF': 2})
            temp_2d = dr_weights[data_names[e] + st_add].copy()
            nmf_dr_weights[data_names[e]+st_add] = temp_2d[temp_2d['model_name']=='NMF']
            nmf3_dr_weights[data_names[e]+st_add], nmf3_model[data_names[e]+st_add] = rlm.apply_dimensionality_reduction(temp_M_chosen, how={'NMF': 3})
            for dim_replication in [4,6,8,10,20]:
                nmf4_dr_weights[data_names[e]+st_add+'_'+str(dim_replication)], nmf4_model[data_names[e]+st_add+'_'+str(dim_replication)] = rlm.apply_dimensionality_reduction(temp_M_chosen, how={'NMF': dim_replication})
            d_how = [{'PCA': 2}, {'ICA': 2}]
            for i in d_how:
                temp, model = rlm.apply_dimensionality_reduction(temp_M_chosen, how=i)
                dr_weights[data_names[e]+st_add] = pd.concat([dr_weights[data_names[e]+st_add], temp], axis=0)
            dr_weights[data_names[e]+st_add]['ENROLL_CAT'] = pd.merge(temp_M_chosen, Labels[data_names[e]][['ENROLL_CAT']], left_index=True, right_index=True)['ENROLL_CAT']
            df = pd.DataFrame(nmf3_model[data_names[e] + st_add].components_, columns=temp_M_chosen.columns)
            cluster_final_visit_3d[data_names[e]+st_add] = get_cluster_projection_matrix(nmf3_dr_weights[data_names[e]+st_add], df, last_visit, dataset=data_parameters.dataset)
            df = pd.DataFrame(nmf_model[data_names[e] + st_add].components_, columns=temp_M_chosen.columns)
            cluster_final_visit_2d[data_names[e] + st_add] = get_cluster_projection_matrix(nmf_dr_weights[data_names[e]+st_add], df, last_visit, num_cluster=2, dataset=data_parameters.dataset)
    return_data['dr_weights'] = dr_weights
    return_data['nmf_dr_weights'] = nmf_dr_weights
    return_data['nmf_model'] = nmf_model
    return_data['nmf_projection_cluster'] = cluster_final_visit_2d
    return_data['nmf3_dr_weights'] = nmf3_dr_weights
    return_data['nmf3_model'] = nmf3_model
    return_data['nmf3_projection_cluster'] = cluster_final_visit_3d
    return_data['nmf4_dr_weights'] = nmf4_dr_weights
    return_data['nmf4_model'] = nmf4_model
    return return_data


def plot_2d_scatter_all(input_data, palette, select_data=[5]):
    data_parameters_list = input_data['data_parameters_list']
    data_names = input_data['data_names']
    dr_weights = input_data['dr_weights']
    interested_data = {}
    sns.set()
    for e, data_parameters in enumerate(data_parameters_list):
        if not e in select_data:
            continue
        g = sns.relplot(x="latent_weight1", y="latent_weight2", hue="ENROLL_CAT", col="model_name",
                        data=dr_weights[data_names[e]], facet_kws={'sharey': False, 'sharex': False}, palette=palette)
        g.fig.suptitle(data_names[e], y=1.05)


def plot_2d_scatter_hc_pd(input_data, palette, utils, enroll_cats=['PD', 'HC']):
    data_parameters_list = input_data['data_parameters_list']
    data_names = input_data['data_names']
    dr_weights = input_data['dr_weights']
    interested_data = {}
    for e, data_parameters in enumerate(data_parameters_list):
        temp = dr_weights[data_names[e]]
        temp2 = temp[(temp['model_name'] == 'NMF') & (temp['ENROLL_CAT'].isin(enroll_cats))]
        interested_data[data_names[e]] = temp2
    utils.create_scatter_plot(interested_data, ncols=3, xcol='latent_weight1', ycol='latent_weight2', palette=palette,
                              hue='ENROLL_CAT')
