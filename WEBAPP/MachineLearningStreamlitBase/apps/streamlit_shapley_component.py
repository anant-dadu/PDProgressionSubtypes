import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
import plotly.express as px
import plotly
import copy
import matplotlib.pyplot as plt
# st.set_option('deprecation.showPyplotGlobalUse', False)

dict_map_result = {
    'smoker': "Smoking status",
    'cognitiveStatus2': "Cognitive status 2",    
    'elEscorialAtDx': "El Escorial category at diagnosis",
    'anatomicalLevel_at_onset': "Anatomical level at onset",
    'site_of_onset': "Site of symptom onset",
    'onset_side': "Onset side",
    'ALSFRS1': "ALSFRS-R part 1 score",
    'FVCPercentAtDx': "FVC% at diagnosis",
    'weightAtDx_kg': "Weight at diagnosis (kg)",
    'rateOfDeclineBMI_per_month': "Rate of BMI decline (per month)",
    'age_at_onset': "Age at symptom onset",
    'firstALSFRS_daysIntoIllness': "Time of first ALSFRS-R measurement (days from symptom onset)"
}

dict_map_result = {
    "V04_MCATOT#V04_numerical": "MOCA Total (Year1)",
    "BL_NP1SLPD#BL_numerical": "NP1SLPD (Baseline)",
    "V04_a_trait#V04_numerical": "a_trait (Year1)",
    "V04_NP1SLPD#V04_numerical": "NP1SLPB (Year1)",
    "BL_urinary#BL_numerical": "urinary (Baseline)",
    "V04_delayed_recall#V04_numerical": "Delayed recall (Year1)",
    "V04_SDMTOTAL#V04_numerical": "SDMTOTAL (Year1)",
    "V04_total#V04_numerical": "total sdm (Year1)",
    "V04_NP2DRES#V04_numerical": "NP2DRES (Year1)",
    "BL_VLTANIM#BL_numerical": "VLTANIM (Year1)",
    "V04_DRMAGRAC#V04_numerical": "DRMAGRAC (Year1)",
    "BL_DRMFIGHT#BL_numerical": "DRMFIGHT (Baseline)",
    "BL_NP3POSTR#BL_numerical": "NP3POSTR (Baseline)",
    "V04_HVLTRT2#V04_numerical": "HVLTRT2 (Year1)",
    "V04_NP2HOBB#V04_numerical": "NP2HOBB (Year1)",
    "V04_NHY#V04_numerical": "NHY (Year1)",
    "V04_ESS4#V04_numerical": "ESS4 (Year1)",
    "BL_NP2DRES#BL_numerical": "NP2DRES (Year1)",
    "V04_VLTVEG#V04_numerical": "VLTVEG (Year1)",
    "V04_HVLTRDLY#V04_numerical": "HVLTRDLY (Year1)",

}


def app():
    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def load_model1():
        with open('saved_models/trainXGB_class_map.pkl', 'rb') as f:
            class_names = list(pickle.load(f))
        return class_names

    class_names = ["PD_h"] # load_model1()

    st.write("## SHAP Model Interpretation")

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def load_model2():
        with open('saved_models/trainXGB_gpu.aucs', 'rb') as f:
            result_aucs = pickle.load(f)
        return result_aucs

    result_aucs = load_model2()

    if len(result_aucs[class_names[0]]) == 3:
        mean_train_aucs = round(np.mean([result_aucs[i][0] for i in class_names]), 2)
        mean_test_aucs = round(np.mean([result_aucs[i][1] for i in class_names]), 2)
        df_res = pd.DataFrame({'class name': class_names + ['MEAN'], 'Discovery Cohort AUC': ["{:.2f}".format(result_aucs[i][0]) for i in class_names] + [mean_train_aucs], 'Replication Cohort AUC':  ["{:.2f}".format(result_aucs[i][1]) for i in class_names] + [mean_test_aucs]})
        replication_avail = True
    else:
        df_res = pd.DataFrame({'class name': class_names, 'Train AUC': ["{:.2f}".format(result_aucs[i][0]) for i in class_names], 'Test AUC':  ["{:.2f}".format(result_aucs[i][1]) for i in class_names]})
        replication_avail = False
    
    @st.cache(allow_output_mutation=True, ttl=24 * 3600)
    def get_shapley_value_data(train, replication=True, dict_map_result={}):
        dataset_type = '' 
        shap_values = np.concatenate([train[0]['shap_values_train'], train[0]['shap_values_test']], axis=0)
        # shap_values = train[0]['shap_values_train']
        X = pd.concat([train[1]['X_train'], train[1]['X_valid']], axis=0)
        exval = train[2]['explainer_train'] 
        auc_train = train[3]['AUC_train']
        auc_test = train[3]['AUC_test']
        ids = list(train[3]['ID_train'.format(dataset_type)]) + list(train[3]['ID_test'.format(dataset_type)])
        labels_pred = list(train[3]['y_pred_train'.format(dataset_type)]) + list(train[3]['y_pred_test'.format(dataset_type)]) 
        labels_actual = list(train[3]['y_train'.format(dataset_type)]) + list(train[3]['y_test'.format(dataset_type)]) 
        shap_values_updated = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns)
        train_samples = len(train[1]['X_train'])
        test_samples = len(train[1]['X_valid'])
        X.columns = ['{}'.format(dict_map_result[col]) if dict_map_result.get(col, None) is not None else col for col in list(X.columns)]
        # X.columns = ['({}) {}'.format(dict_map_result[col], col) if dict_map_result.get(col, None) is not None else col for col in list(X.columns)]
        # shap_values_updated = copy.deepcopy(shap_values_updated)
        shap_values_updated = shap_values_updated
        patient_index = [hashlib.md5(str(s).encode()).hexdigest() for e, s in enumerate(ids)]
        return (X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_updated, train_samples, test_samples)
    
    st.write("## Introduction")
    st.write(
        """
        The Shapley additive explanations (SHAP) approach was used to evaluate each feature’s influence in the ensemble learning. This approach, used in game theory, assigned an importance (Shapley) value to each feature to determine a player’s contribution to success. Shapley explanations enhance understanding by creating accurate explanations for each observation in a dataset. They bolster trust when the critical variables for specific records conform to human domain knowledge and reasonable expectations. 
        We used the one-vs-rest technique for multiclass classification. Based on that, we trained a separate binary classification model for each class.
        """
    )
    feature_set_my = class_names[0]

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def load_model3():
        with open('saved_models/trainXGB_gpu_{}.data'.format(feature_set_my), 'rb') as f:
            train = pickle.load(f)
        return train

    train = load_model3()
    # data_load_state = st.text('Loading data...')
    cloned_output = get_shapley_value_data(train, replication=replication_avail, dict_map_result=dict_map_result)

    # data_load_state.text("Done Data Loading! (using st.cache)")
    X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_up, len_train, len_test = cloned_output 
    

    # st.write("## Results")
    # st.write("### Performance of Surrogate Model")
    # st.table(df_res.set_index('class name').astype(str))

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def load_model3():
        shap_values_list = []
        for classname in class_names:
            with open('saved_models/trainXGB_gpu_{}.data'.format(classname), 'rb') as f:
                train_temp = pickle.load(f)
                shap_values_list.append(np.concatenate([train_temp[0]['shap_values_train'], train_temp[0]['shap_values_test']], axis=0))
        shap_values = np.mean(shap_values_list, axis=0)
        return shap_values

    shap_values = load_model3()
    import sklearn
    # st.write(sum(labels_actual[:len_train]), sum(np.array(labels_pred[:len_train])>0.5))
    # st.write(sum(labels_actual[len_train:]), sum(np.array(labels_pred[len_train:])>0.5))
    
        
    st.write('## Summary Plot')
    st.write("""Shows top-20 features that have the most significant impact on the classification model.""")
    if True: # st.checkbox("Show Summary Plot"):
        shap_type = 'trainXGB'
        col1, col2, col2111 = st.columns(3)
        with col1:
            st.write('---')
            # temp = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns)
            @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
            def generate_plot1():
                fig, ax = plt.subplots(figsize=(10,15))
                # st.write(shap_values)
                shap.plots.beeswarm(shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns), show=False, max_display=20, order = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns).mean(0).abs, plot_size=0.47)# 0.47# , return_objects=True
                # shap.plots.beeswarm(temp, order=temp.mean(0).abs, show=False, max_display=20) # , return_objects=True
                return fig

            # fig.savefig('up_summary_plot1.svg', bbox_inches='tight', dpi=250)
            # fig.savefig('up_summary_plot1.eps', bbox_inches='tight')
            st.pyplot(generate_plot1())
            st.write('---')
        with col2:
            st.write('---')

            @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
            def generate_plot2():
                fig, ax = plt.subplots(figsize=(10,15))
                # temp = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns)
                shap.plots.bar(shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns).mean(0), show=False, max_display=20, order=shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns).mean(0).abs)
                # shap.plots.bar(temp, order=temp.mean(0).abs, show=False, max_display=20)
                return fig

            # fig.savefig('summary_plot2.pdf', bbox_inches='tight')
            # fig.savefig('summary_plot2.eps', bbox_inches='tight')
            st.pyplot(generate_plot2())
            st.write('---')
        with col2111:
            st.write('---')

            @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
            def generate_plot3():
                fig, ax = plt.subplots(figsize=(10,15))
                # temp = shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns)
                shap.plots.bar(shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns).abs.mean(0), show=False, max_display=20, order=shap.Explanation(values=np.array(shap_values), base_values=np.array([exval]*len(X)), data=np.array(X.values), feature_names=X.columns).mean(0).abs)
                return fig

            # shap.plots.bar(temp, order=temp.mean(0).abs, show=False, max_display=20)
            # fig.savefig('summary_plot3.pdf', bbox_inches='tight')
            # fig.savefig('summary_plot3.eps', bbox_inches='tight')
            st.pyplot(generate_plot3())
            st.write('---')
    


    

    
    import random
    select_random_samples = np.random.choice(shap_values.shape[0], 800)

    new_X = X.iloc[select_random_samples]
    new_shap_values = shap_values[select_random_samples,:]
    new_labels_pred = np.array(labels_pred, dtype=np.float64)[select_random_samples] 


    
    st.write('## Statistics for Individual Classes') 
    # st.write("#### Select the class")
    # feature_set_my = st.selectbox("", ['Select']+ class_names, index=0)
    feature_set_my = "PD_h"
    if not feature_set_my== "Select":
        @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
        def load_model9():
            with open('saved_models/trainXGB_gpu_{}.data'.format(feature_set_my), 'rb') as f:
                train = pickle.load(f)
            return train
        train = load_model9()
        # data_load_state = st.text('Loading data...')
        # cloned_output = copy.deepcopy(get_shapley_value_data(train, replication=replication_avail, dict_map_result=dict_map_result))
        cloned_output = get_shapley_value_data(train, replication=replication_avail, dict_map_result=dict_map_result)
        # data_load_state.text("Done Data Loading! (using st.cache)")
        X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_up, len_train, len_test = cloned_output 
        col0, col00 = st.columns(2)
        with col0:
            st.write("### Data Statistics")
            st.info ('Total Features: {}'.format(X.shape[1]))
            st.info ('Total Samples: {} (Discovey: {}, Replication: {})'.format(X.shape[0], len_train, len_test))
    
        with col00:
            st.write("### ML Model Performance")
            st.info ('AUC Discovery Cohort: {}'.format(round(auc_train,2)))
            st.info ('AUC Replication Cohort: {}'.format( round(auc_test,2)))

        col01, col02 = st.columns(2)
        with col01:
            st.write("### Discovery Cohort Confusion Matrix")
            Z = sklearn.metrics.confusion_matrix(labels_actual[:len_train], np.array(labels_pred[:len_train])>0.5)
            Z_df = pd.DataFrame(Z, columns=['Predicted 0', 'Predicted 1'], index= ['Actual 0', 'Actual 1'])
            st.table(Z_df.astype(str))
    
        with col02:
            st.write("### Replication Cohort Confusion Matrix")
            Z = sklearn.metrics.confusion_matrix(labels_actual[len_train:], np.array(labels_pred[len_train:])>0.5)
            Z_df = pd.DataFrame(Z, columns=['Predicted 0', 'Predicted 1'], index= ['Actual 0', 'Actual 1'])
            st.table(Z_df.astype(str))
        
        labels_actual_new = np.array(labels_actual, dtype=np.float64)
        y_pred = (shap_values.sum(1) + exval) > 0
        misclassified = y_pred != labels_actual_new 

        st.write('### Pathways for Misclassified Samples')
        if misclassified[len_train:].sum() == 0:
            st.info('No Misclassified Examples!!!')
        elif True: # st.checkbox("Show Misclassifies Pathways"):
            col6, col7 = st.columns(2)
            with col6:
                st.info('Misclassifications (test): {}/{}'.format(misclassified[len_train:].sum(), len_test))
                fig, ax = plt.subplots()
                r = shap.decision_plot(exval, shap_values[misclassified], list(X.columns), link='logit', return_objects=True, new_base_value=0)
                st.pyplot(fig)
            with col7:
                # st.info('Single Example')
                sel_patients = [patient_index[e] for e, i in enumerate(misclassified) if i==1]
                select_pats = st.selectbox('Select random misclassified patient', options=list(sel_patients))
                id_sel_pats = sel_patients.index(select_pats)
                fig, ax = plt.subplots()
                shap.decision_plot(exval, shap_values[misclassified][id_sel_pats], X.iloc[misclassified,:].iloc[id_sel_pats], link='logit', feature_order=r.feature_idx, highlight=0, new_base_value=0)
                st.pyplot()
        st.write('## Decision Plots')
        st.write("""
        We selected 800 subsamples to understand the pathways of predictive modeling. SHAP decision plots show how complex models arrive at their predictions (i.e., how models make decisions). 
        Each observation’s prediction is represented by a colored line.
        At the top of the plot, each line strikes the x-axis at its corresponding observation’s predicted value. 
        This value determines the color of the line on a spectrum. 
        Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model’s base value. 
        This shows how each feature contributes to the overall prediction.
        """)
        cols = st.columns(2)
        st.write('### Prediction pathways')
        if st.checkbox("View patterns"): # st.checkbox("Show Prediction Pathways (Feature Clustered)"):
                # col3, col4, col5 = st.columns(3)
                # st.write('Typical Prediction Path: Uncertainity (0.2-0.8)')
                r = shap.decision_plot(exval, np.array(new_shap_values), list(new_X.columns), feature_order='hclust', return_objects=True, show=False)
                T = new_X.iloc[(new_labels_pred >= 0) & (new_labels_pred <= 1)]
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sh = np.array(new_shap_values)[(new_labels_pred >= 0) & (new_labels_pred <= 1), :]
                fig, ax = plt.subplots()
                shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True, new_base_value=0)
                cols[0].pyplot(fig)
                # with col4:
                #     st.write('Typical Prediction Path: Positive Class (>=0.9)')
                #     fig, ax = plt.subplots()
                #     T = new_X.iloc[np.array(new_labels_pred, dtype=np.float64) >= 0.9]
                #     import warnings
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore")
                #         sh = np.array(new_shap_values)[new_labels_pred >= 0.9, :]
                #     shap.decision_plot(exval, sh, T, show=False, link='logit',  feature_order=r.feature_idx, new_base_value=0)
                #     st.pyplot(fig)
                # with col5:
                #     st.write('Typical Prediction Path: Negative Class (<=0.1)')
                #     fig, ax = plt.subplots()
                #     T = new_X.iloc[new_labels_pred <= 0.1]
                #     import warnings
                #     with warnings.catch_warnings():
                #            warnings.simplefilter("ignore")
                #            sh = np.array(new_shap_values)[new_labels_pred <= 0.1, :]
                #     shap.decision_plot(exval, sh, T, show=False, link='logit', feature_order=r.feature_idx, new_base_value=0)
                #     st.pyplot(fig)
    

        # cols[1].write('### Pathways for Prediction (Feature Importance)')

        # if st.checkbox("View patterns"): # st.checkbox("Show Prediction Pathways (Feature Importance)"):
                # col31, col41, col51 = st.columns(3)
                # with col31:
                # st.write('Typical Prediction Path: Uncertainity (0.2-0.8)')
                r = shap.decision_plot(exval, np.array(new_shap_values), list(new_X.columns), return_objects=True, show=False)
                T = new_X.iloc[(new_labels_pred >= 0) & (new_labels_pred <= 1)]
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sh = np.array(new_shap_values)[(new_labels_pred >= 0) & (new_labels_pred <= 1), :]
                fig, ax = plt.subplots()
                shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True, new_base_value=0)
                cols[1].pyplot(fig)
                # with col41:
                #     st.write('Typical Prediction Path: Positive Class (>=0.9)')
                #     fig, ax = plt.subplots()
                #     T = new_X.iloc[np.array(new_labels_pred, dtype=np.float64) >= 0.9]
                #     import warnings
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore")
                #         sh = np.array(new_shap_values)[new_labels_pred >= 0.9, :]
                #     shap.decision_plot(exval, sh, T, show=False, link='logit',  feature_order=r.feature_idx, new_base_value=0)
                #     st.pyplot(fig)
                # with col51:
                #     st.write('Typical Prediction Path: Negative Class (<=0.1)')
                #     fig, ax = plt.subplots()
                #     T = new_X.iloc[new_labels_pred <= 0.1]
                #     import warnings
                #     with warnings.catch_warnings():
                #            warnings.simplefilter("ignore")
                #            sh = np.array(new_shap_values)[new_labels_pred <= 0.1, :]
                #     shap.decision_plot(exval, sh, T, show=False, link='logit', feature_order=r.feature_idx, new_base_value=0)
                #     st.pyplot(fig)













    # st.write('## Dependence Plots')
    # st.write("""We can observe the interaction effects of different features in for predictions. To help reveal these interactions dependence_plot automatically lists (top-3) potential features for coloring.
    #     Furthermore, we can observe the relationship betweem features and SHAP values for prediction using the dependence plots, which compares the actual feature value (x-axis) against the SHAP score (y-axis).
    #    It shows that the effect of feature values is not a simple relationship where increase in the feature value leads to consistent changes in model output but a complicated non-linear relationship.""")
    if False: # st.checkbox("Show Dependence Plots"):
        feature_name = st.selectbox('Select a feature for dependence plot', options=list(X.columns))
        try:
            inds = shap.utils.potential_interactions(
                shap.Explanation(values=np.array(shap_values), base_values=np.array([exval] * len(X)),
                                 data=np.array(X.values), feature_names=X.columns)[:, feature_name],
                shap.Explanation(values=np.array(shap_values), base_values=np.array([exval] * len(X)),
                                 data=np.array(X.values), feature_names=X.columns))
        except:
            st.info("Select Another Feature")
        st.write('Top3 Potential Interactions for ***{}***'.format(feature_name))
        col3, col4, col5 = st.columns(3)
        with col3:
            shap.dependence_plot(feature_name, np.array(shap_values), X,
                                 interaction_index=list(X.columns).index(list(X.columns)[inds[0]]))
            st.pyplot()
        with col4:
            shap.dependence_plot(feature_name, np.array(shap_values), X,
                                 interaction_index=list(X.columns).index(list(X.columns)[inds[1]]))
            st.pyplot()
        with col5:
            shap.dependence_plot(feature_name, np.array(shap_values), X,
                                 interaction_index=list(X.columns).index(list(X.columns)[inds[2]]))
            st.pyplot()