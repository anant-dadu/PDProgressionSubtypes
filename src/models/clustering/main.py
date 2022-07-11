import pandas as pd
import numpy as np
from dotmap import DotMap
import pandas as pd
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.cluster import KMeans


def perform_clustering(input_data, cm, mapping={'HC': 'HC', 'PD': 'PD', 'SWEDD': 'SWEDD'}, trained_model={}):
    data_names = input_data["data_names"]
    M_chosen = input_data["M_chosen"]
    data_parameters_list = input_data["data_parameters_list"]
    Labels = input_data["Labels"]
    dr_weights = input_data['dr_weights']
    adjusted_2d_weights = input_data['nmf_projection_cluster'] 
    adjusted_3d_weights = input_data['nmf3_projection_cluster'] 
    return_data = {}
    d_M_PD_HC_gmm_chosen = {}
    d_M_PD_HC_gmm_chosen_3d = {}
    d_M_label = {}
    d_M_label_3d = {}

    gmm_models = {}
    gmm_models_3d = {}
    d_cluster_kmeans = DotMap()
    d_selected_M_chosen = DotMap()
    d_components = DotMap()
    d_cluster_kmeans_3d = DotMap()
    d_selected_M_chosen_3d = DotMap()
    d_components_3d = DotMap()
    d_components_3d_unnormalied = DotMap()
    d_components_2d_unnormalied = DotMap()
    d_cluster_kmeans_2cluster = DotMap()
    d_selected_M_chosen_2cluster = DotMap()
    d_components_2cluster = DotMap()
    d_bics_2d = {}
    d_bics_3d = {}
    for e, data_parameters in enumerate(data_parameters_list):
        data_parameters = data_parameters.toDict()
        nmf2_dr_weights = dr_weights[data_names[e]][dr_weights[data_names[e]]['model_name'] == 'NMF']
        nmf3_dr_weights = input_data['nmf3_dr_weights'][data_names[e]]
        M_cat = pd.merge(M_chosen[data_names[e]], Labels[data_names[e]][['ENROLL_CAT']], left_index=True,
                         right_index=True)

        M_label_columns = ['GMM_2d', 'GMM_2d_adj']
        M_label = pd.DataFrame(index=M_chosen[data_names[e]].index, columns=M_label_columns)
        M_label[M_cat.ENROLL_CAT == mapping["HC"]] = mapping['HC']
        M_label[M_cat.ENROLL_CAT == mapping["SWEDD"]] = mapping['SWEDD']

        M_PD_gmm_chosen = nmf2_dr_weights.iloc[:, :-2][M_cat.ENROLL_CAT == mapping["PD"]].copy()
        M_PD_HC_gmm_chosen = nmf2_dr_weights.iloc[:, :-2][M_cat.ENROLL_CAT.isin([mapping['HC'], mapping['PD']])].copy()
        # import pdb; pdb.set_trace()
        gmm_models[data_names[e]], predictions, d_bics_2d[data_names[e]] = cm.apply_GMM(M_PD_gmm_chosen, 3, trained_model.get(data_parameters.get('gmm_model', 'dummy'), None))

        M_PD_gmm_chosen_adj = adjusted_2d_weights[data_names[e]][1][M_cat.ENROLL_CAT == mapping["PD"]].copy()
        M_PD_HC_gmm_chosen_adj = adjusted_2d_weights[data_names[e]][1][M_cat.ENROLL_CAT.isin([mapping['HC'], mapping['PD']])].copy()
        gmm_models[data_names[e]+'_adj'], predictions_adj, d_bics_2d[data_names[e]+'_adj'] = cm.apply_GMM(M_PD_gmm_chosen_adj, 3, trained_model.get(data_parameters.get('gmm_model', 'dummy'), None))

        M_label_columns_3d = ['GMM_3d', 'GMM_3d_adj']
        M_label_3d = pd.DataFrame(index=M_chosen[data_names[e]].index, columns=M_label_columns_3d)
        M_label_3d[M_cat.ENROLL_CAT == mapping["HC"]] = mapping['HC']
        M_label_3d[M_cat.ENROLL_CAT == mapping["SWEDD"]] = mapping['SWEDD']
        M_gmm_chosen_3d = nmf3_dr_weights.iloc[:, :-1]
        M_PD_gmm_chosen_3d = M_gmm_chosen_3d[M_cat.ENROLL_CAT == mapping["PD"]]
        M_PD_HC_gmm_chosen_3d = M_gmm_chosen_3d[M_cat.ENROLL_CAT.isin([mapping['HC'], mapping['PD']])]
        gmm_models_3d[data_names[e]] , predictions_3d, d_bics_3d[data_names[e]] = cm.apply_GMM(M_PD_gmm_chosen_3d, 3, trained_model.get(
            data_parameters.get('gmm_model_3d', 'dummy'), None))

        M_gmm_chosen_3d_adj = adjusted_3d_weights[data_names[e]][1].copy()
        M_PD_gmm_chosen_3d_adj = M_gmm_chosen_3d_adj[M_cat.ENROLL_CAT == mapping["PD"]]
        M_PD_HC_gmm_chosen_3d_adj = M_gmm_chosen_3d_adj[M_cat.ENROLL_CAT.isin([mapping['HC'], mapping['PD']])]
        gmm_models_3d[data_names[e]+'_adj'] , predictions_3d_adj, d_bics_3d[data_names[e]+'_adj'] = cm.apply_GMM(M_PD_gmm_chosen_3d_adj, 3, trained_model.get(
            data_parameters.get('gmm_model_3d', 'dummy'), None))

        M_label.loc[M_PD_gmm_chosen.index, 'GMM_2d'] = predictions
        M_label.loc[M_PD_gmm_chosen.index, 'GMM_2d_adj'] = predictions_adj
        M_label.replace({0: 'PD_1', 1: 'PD_2', 2: 'PD_3'}, inplace=True)
        M_label_3d.loc[M_PD_gmm_chosen.index, 'GMM_3d'] = predictions_3d
        M_label_3d.loc[M_PD_gmm_chosen.index, 'GMM_3d_adj'] = predictions_3d_adj
        M_label_3d.replace({0: 'PD_1', 1: 'PD_2', 2: 'PD_3'}, inplace=True)

        M_PD_HC_gmm_chosen = pd.merge(M_PD_HC_gmm_chosen, M_label, left_index=True, right_index=True)
        M_PD_HC_gmm_chosen_adj = pd.merge(M_PD_HC_gmm_chosen_adj, M_label, left_index=True, right_index=True)
        def get_label_correct(M_PD_HC_gmm_chosen, cname, mapping):
            M_PD_HC_gmm_chosen[cname] = M_PD_HC_gmm_chosen[cname].replace({0: 'PD_1', 1: 'PD_2', 2: 'PD_3'})
            # import pdb; pdb.set_trace()
            mean = np.array(M_PD_HC_gmm_chosen[M_PD_HC_gmm_chosen[cname].isin([mapping['HC']])].iloc[:, :-2]).mean(axis=0)
            mean1 = np.array(M_PD_HC_gmm_chosen[M_PD_HC_gmm_chosen[cname].isin(['PD_1'])].iloc[:, :-2]).mean(axis=0)
            mean2 = np.array(M_PD_HC_gmm_chosen[M_PD_HC_gmm_chosen[cname].isin(['PD_2'])].iloc[:, :-2]).mean(axis=0)
            mean3 = np.array(M_PD_HC_gmm_chosen[M_PD_HC_gmm_chosen[cname].isin(['PD_3'])].iloc[:, :-2]).mean(axis=0)
            mean1 = np.sum((mean1 - mean))
            mean2 = np.sum((mean2 - mean))
            mean3 = np.sum((mean3 - mean))
            z = np.argsort([mean1, mean2, mean3]) + 1
            d_replace = {'PD_' + str(z[0]): 'PD_l', 'PD_' + str(z[1]): 'PD_m', 'PD_' + str(z[2]): 'PD_h'}
            return d_replace
        d_replace = get_label_correct(M_PD_HC_gmm_chosen, 'GMM_2d',  mapping) 
        M_PD_HC_gmm_chosen['GMM_2d'] = M_PD_HC_gmm_chosen['GMM_2d'].map(lambda x: d_replace.get(x, x))
        M_label['GMM_2d'] = M_label['GMM_2d'].map(lambda x: d_replace.get(x, x))
        d_replace = get_label_correct(M_PD_HC_gmm_chosen_adj, 'GMM_2d_adj', mapping) 
        M_PD_HC_gmm_chosen['GMM_2d_adj'] = M_PD_HC_gmm_chosen_adj['GMM_2d_adj'].map(lambda x: d_replace.get(x, x))
        M_label['GMM_2d_adj'] = M_label['GMM_2d_adj'].map(lambda x: d_replace.get(x, x))
        d_M_PD_HC_gmm_chosen[data_names[e]] = M_PD_HC_gmm_chosen.copy()
        d_M_PD_HC_gmm_chosen[data_names[e]+'_adj'] = M_PD_HC_gmm_chosen_adj.copy()
        d_M_label[data_names[e]] = M_label.copy()

        M_PD_HC_gmm_chosen_3d = pd.merge(M_PD_HC_gmm_chosen_3d, M_label_3d, left_index=True, right_index=True)
        M_PD_HC_gmm_chosen_3d_adj = pd.merge(M_PD_HC_gmm_chosen_3d_adj, M_label_3d, left_index=True, right_index=True)
        d_replace = get_label_correct(M_PD_HC_gmm_chosen_3d, 'GMM_3d',  mapping)  
        M_PD_HC_gmm_chosen_3d['GMM_3d'] = M_PD_HC_gmm_chosen_3d['GMM_3d'].map(lambda x: d_replace.get(x, x))
        M_label_3d['GMM_3d'] = M_label_3d['GMM_3d'].map(lambda x: d_replace.get(x, x))
        d_replace = get_label_correct(M_PD_HC_gmm_chosen_3d_adj, 'GMM_3d_adj', mapping)  
        M_PD_HC_gmm_chosen_3d_adj['GMM_3d_adj'] = M_PD_HC_gmm_chosen_3d_adj['GMM_3d_adj'].map(lambda x: d_replace.get(x, x))
        M_label_3d['GMM_3d_adj'] = M_label_3d['GMM_3d_adj'].map(lambda x: d_replace.get(x, x))
        d_M_PD_HC_gmm_chosen_3d[data_names[e]] = M_PD_HC_gmm_chosen_3d.copy()
        d_M_PD_HC_gmm_chosen_3d[data_names[e]+'_adj'] = M_PD_HC_gmm_chosen_3d_adj.copy()
        d_M_label_3d[data_names[e]] = M_label_3d.copy()

        def create_components(components, num_cluster, M_chosen, data_name, data_parameters, d_cluster_kmeans,
                              d_selected_M_chosen, d_components, d_components_3d_unnormalied=None):

            components.columns = M_chosen[data_names[e]].columns
            for visit_id in data_parameters['visits_list']:
                t_bcomponents = components.iloc[:, components.columns.get_level_values(1) == visit_id]
                t_bcomponents.columns = [i for i, j in t_bcomponents.columns]
                x = t_bcomponents.values
                if not d_components_3d_unnormalied is None:
                    d_components_3d_unnormalied[data_name][visit_id] = t_bcomponents.copy()
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(x)
                t_bcomponents = pd.DataFrame(x_scaled, columns=t_bcomponents.columns)
                X_Norm = t_bcomponents.values.transpose()  # preprocessing.normalize(bcomponents[data_names[e]])
                d_components[data_name][visit_id] = t_bcomponents.copy()
                kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X_Norm)
                clustered_feats = []
                transposed = t_bcomponents.T
                for i in range(num_cluster):
                    clustered_feats.append(list(set(transposed[kmeans.labels_ == i].index.values.tolist())))
                if len(clustered_feats) == 3:
                    cluster_kmeans = {'Sleep': clustered_feats[0], 'Motor': clustered_feats[1],
                                      'Cognitive': clustered_feats[2]}
                else:
                    cluster_kmeans = {'Cognitive': clustered_feats[0], 'Motor': clustered_feats[1]}
                # for en, i in enumerate(clustered_feats):
                #    if ('NP1HALL' in i) and len(cluster_kmeans['Sleep'])==0:
                #        cluster_kmeans['Sleep'] = i
                #    elif ('NP3HMOVR' in i) and len(cluster_kmeans['Motor'])==0:
                #        cluster_kmeans['Motor'] = i
                #    elif len(cluster_kmeans['Cognitive'])==0:
                #        cluster_kmeans['Cognitive'] = i
                d_cluster_kmeans[data_name][visit_id] = cluster_kmeans
                selected_M_chosen = M_chosen[data_name].loc[:, M_chosen[data_names[e]].columns.get_level_values(1).isin(
                    data_parameters['visits_list'])]
                d_selected_M_chosen[data_name][visit_id] = selected_M_chosen.loc[:,
                                                           selected_M_chosen.columns.get_level_values(0).isin(
                                                               transposed.index.values.tolist())]
            # return d_cluster_kmeans, d_selected_M_chosen, d_components

        components = pd.DataFrame(input_data['nmf3_model'][data_names[e]].components_)
        create_components(components, 3, M_chosen, data_names[e], data_parameters, d_cluster_kmeans_3d,
                          d_selected_M_chosen_3d, d_components_3d, d_components_3d_unnormalied)
        components = pd.DataFrame(input_data['nmf_model'][data_names[e]].components_)
        create_components(components, 3, M_chosen, data_names[e], data_parameters, d_cluster_kmeans,
                          d_selected_M_chosen, d_components)
        components = pd.DataFrame(input_data['nmf_model'][data_names[e]].components_)
        create_components(components, 2, M_chosen, data_names[e], data_parameters, d_cluster_kmeans_2cluster,
                          d_selected_M_chosen_2cluster, d_components_2cluster, d_components_2d_unnormalied)
    return_data['d_bics_3d'] = d_bics_3d
    return_data['d_bics_2d'] = d_bics_2d
    return_data['d_selected_M_chosen'] = d_selected_M_chosen
    return_data['d_cluster_kmeans'] = d_cluster_kmeans
    return_data['d_selected_M_chosen_2cluster'] = d_selected_M_chosen_2cluster
    return_data['d_cluster_kmeans_2cluster'] = d_cluster_kmeans_2cluster
    return_data['d_selected_M_chosen_3d'] = d_selected_M_chosen_3d
    return_data['d_cluster_kmeans_3d'] = d_cluster_kmeans_3d
    return_data['d_components'] = d_components
    return_data['d_components_3d'] = d_components_3d
    return_data['d_components_3d_normalized'] = d_components_3d_unnormalied
    return_data['d_components_2d_normalized'] = d_components_2d_unnormalied
    return_data['d_components_2cluster'] = d_components_2cluster
    return_data['d_M_PD_HC_gmm_chosen'] = d_M_PD_HC_gmm_chosen
    return_data['d_M_label'] = d_M_label
    return_data['gmm_models'] = gmm_models
    return_data['d_M_PD_HC_gmm_chosen_3d'] = d_M_PD_HC_gmm_chosen_3d
    return_data['d_M_label_3d'] = d_M_label_3d
    return_data['gmm_models_3d'] = gmm_models_3d
    return return_data


def plot_2d_scatter_hc_pd(input_data, palette, utils, label='', ncols=3, fsize=None, style=None, xl=False, yl=False):
    data_parameters_list = input_data['data_parameters_list']
    data_names = input_data['data_names']
    dr_weights = input_data['dr_weights']
    d_M_PD_HC_gmm_chosen = input_data['d_M_PD_HC_gmm_chosen' + label]
    interested_data = {}
    for e, data_parameters in enumerate(data_parameters_list):
        interested_data[data_names[e]] = d_M_PD_HC_gmm_chosen[data_names[e]]
    utils.create_scatter_plot(interested_data, ncols=ncols, xcol='latent_weight1', ycol='latent_weight2',
                              palette=palette, hue='GMM', fsize=fsize, style=style, xlabel_all=xl, ylabel_all=yl)


def plot_distributions_hc_pd(input_data, palette, utils, select_data=-1):
    data_names = input_data['data_names']
    d_M_PD_HC_gmm_chosen = input_data['d_M_PD_HC_gmm_chosen']
    d_M_label = input_data['d_M_label']
    utils.create_plot_cluster(d_M_label[data_names[select_data]], d_M_PD_HC_gmm_chosen[data_names[select_data]])