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
import joblib
import xgboost as xgb




feature_mapping = {
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

feature_mapping = {
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
feature_mapping = {}

def app():
    st.markdown("""<style>.big-font {font-size:100px !important;}</style>""", unsafe_allow_html=True) 
    st.markdown(
        """<style>
        .boxBorder {
            border: 2px solid #990066;
            padding: 10px;
            outline: #990066 solid 5px;
            outline-offset: 5px;
            font-size:25px;
        }</style>
        """, unsafe_allow_html=True) 
    st.markdown('<div class="boxBorder"><font color="RED">Disclaimer: This predictive tool is only for research purposes</font></div>', unsafe_allow_html=True)
    st.write("## Model Perturbation Analysis")

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def load_model2():
        with open('saved_models/trainXGB_class_map.pkl', 'rb') as f:
            class_names = list(pickle.load(f))
        return class_names

    class_names = load_model2()

    # @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
    def load_model():
        M_dict = {}
        for classname in class_names:
            M_dict[classname] = joblib.load('saved_models/trainXGB_gpu_{}.model'.format(classname))
        return M_dict

    M_dict = load_model()

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def load_model1():
        with open('saved_models/trainXGB_gpu_{}.data'.format(class_names[0]), 'rb') as f:
            train = pickle.load(f)
        with open('saved_models/trainXGB_categorical_map.pkl', 'rb') as f:
            col_dict_map = pickle.load(f)
        return train, col_dict_map

    train, col_dict_map = load_model1()

    X = train[1]['X_valid']# .copy()
    ids = list(train[3]['ID_test'])
    X.index = ids
    labels_pred =  list(train[3]['y_pred_test']) 
    labels_actual = list(train[3]['y_test']) 
    # select_patient = st.selectbox("Select the patient", list(X.index), index=0)
    
    categorical_columns = []
    numerical_columns = []
    X_new = X.fillna('Not available')
    for col in X_new.columns:
        # if len(X_new[col].value_counts()) <= 10:
        if col_dict_map.get(col, None) is not None:
            categorical_columns.append(col)
        else:
            numerical_columns.append(col) 
    
    st.write('### Please enter the following {} factors to perform prediction or select a random patient'.format(len(categorical_columns + numerical_columns)))
    # st.write("***Categorical Columns:***", categorical_columns) 
    # st.write("***Numerical Columns:***", numerical_columns) 
    from collections import defaultdict
    if st.button("Random Patient"):
        import random
        select_patient = random.choice(list(X.index))
    else:
        select_patient = list(X.index)[0]
        # select_patient = "case_PP-3001" # 904

    select_patient_index = ids.index(select_patient) 
    new_feature_input = defaultdict(list) 
    for key, val in col_dict_map.items():
        rval = {j:i for i,j in val.items()}
        X_new[key] = X_new[key].map(lambda x: rval.get(x, x))
    
    st.write('--'*10)
    st.write('##### Note: X denoted NA values')
    col1, col2, col3, col4 = st.columns(4)
    for i in range(0, len(categorical_columns), 4):
        with col1:
            if (i+0) >= len(categorical_columns):
                continue
            c1 = categorical_columns[i+0] 
            idx = list(X_new[c1].unique()).index(X_new.loc[select_patient, c1]) 
            f1 = st.selectbox("{}".format(feature_mapping[c1]), list(X_new[c1].unique()), index=idx)
            new_feature_input[c1].append(col_dict_map[c1].get(f1, np.nan))
        with col2:
            if (i+1) >= len(categorical_columns):
                continue
            c2 = categorical_columns[i+1] 
            idx = list(X_new[c2].unique()).index(X_new.loc[select_patient, c2]) 
            f2 = st.selectbox("{}".format(feature_mapping[c2]), list(X_new[c2].unique()), index=idx)
            new_feature_input[c2].append(col_dict_map[c2].get(f2, np.nan))
        with col3:
            if (i+2) >= len(categorical_columns):
                continue
            c3 = categorical_columns[i+2] 
            idx = list(X_new[c3].unique()).index(X_new.loc[select_patient, c3]) 
            f3 = st.selectbox("{}".format(feature_mapping[c3]), list(X_new[c3].unique()), index=idx)
            new_feature_input[c3].append(col_dict_map[c3].get(f3, np.nan))
        with col4:
            if (i+3) >= len(categorical_columns):
                continue
            c4 = categorical_columns[i+3] 
            idx = list(X_new[c4].unique()).index(X_new.loc[select_patient, c4]) 
            f4 = st.selectbox("{}".format(feature_mapping[c4]), list(X_new[c4].unique()), index=idx)
            new_feature_input[c4].append(col_dict_map[c4].get(f4, np.nan))
    
    for col in numerical_columns:
        X_new[col] = X_new[col].map(lambda x: float(x) if not x=='Not available' else np.nan)
    for i in range(0, len(numerical_columns), 4):
        with col1:
            if (i+0) >= len(numerical_columns):
                continue
            c1 = numerical_columns[i+0] 
            idx = X_new.loc[select_patient, c1]
            f1 = st.number_input("{}".format(feature_mapping.get(c1, c1)), min_value=X_new[c1].min(),  max_value=X_new[c1].max(), value=idx)
            new_feature_input[c1].append(f1)
        with col2:
            if (i+1) >= len(numerical_columns):
                continue
            c2 = numerical_columns[i+1] 
            idx = X_new.loc[select_patient, c2]
            f2 = st.number_input("{}".format(feature_mapping.get(c2, c2)), min_value=X_new[c2].min(),  max_value=X_new[c2].max(), value=idx)
            new_feature_input[c2].append(f2)
        with col3:
            if (i+2) >= len(numerical_columns):
                continue
            c3 = numerical_columns[i+2] 
            idx = X_new.loc[select_patient, c3]
            f3 = st.number_input("{}".format(feature_mapping.get(c3, c3)), min_value=X_new[c3].min(),  max_value=X_new[c3].max(), value=idx)
            new_feature_input[c3].append(f3)
        with col4:
            if (i+3) >= len(numerical_columns):
                continue
            c4 = numerical_columns[i+3] 
            idx = X_new.loc[select_patient, c4]
            f4 = st.number_input("{}".format(feature_mapping.get(c4, c4)), min_value=X_new[c4].min(),  max_value=X_new[c4].max(), value=idx)
            new_feature_input[c4].append(f4)
    
    st.write('--'*10)
    st.write("### Do you want to see the effect of changing a factor on this patient?")
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "goldenred", "magenta", "yellow", "pink", "grey"]
    for e, classname in enumerate(class_names):
        color_discrete_map[classname] = color_discrete_map_list[e] 
    
    show_whatif = st.checkbox("Enable what-if analysis")
    col01, col02 = st.columns(2)
    with col01:
        st.write('### Prediction on actual feature values')
        if not show_whatif:
            dfl = pd.DataFrame(new_feature_input)
            ndfl = dfl.copy()
            for key, val in col_dict_map.items():
                rval = {j: i for i, j in val.items()}
                ndfl[key] = ndfl[key].map(lambda x: rval.get(x, x))
            # st.write('### Prediction with what-if analysis')

            feature_print_what = ndfl.iloc[0].fillna('Not available')
            feature_print_what.index = feature_print_what.index.map(lambda x: feature_mapping.get(x, x))
            feature_print_what = feature_print_what.reset_index()
            feature_print_what.columns = ["Feature Name", "Feature Value"]
            feature_print = feature_print_what.copy()
            dfl = dfl[X.columns].replace('Not available', np.nan)
            predicted_prob = defaultdict(list)
            predicted_class = -1
            max_val = -1
            for key, val in M_dict.items():
                predicted_prob['predicted_probability'].append(
                    val.predict(xgb.DMatrix(dfl.iloc[0, :].values.reshape(1, -1), feature_names=dfl.columns))[0])
                predicted_prob['classname'].append(key)
                if predicted_prob['predicted_probability'][-1] > max_val:
                    predicted_class = key
                    max_val = predicted_prob['predicted_probability'][-1]
            K = pd.DataFrame(predicted_prob)
            K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
            K['color'] = ['zed' if i == predicted_class else 'red' for i in list(predicted_prob['classname'])]
            t1 = dfl.copy()
            t2 = ndfl.copy().fillna('Not available')
        else:
            # st.write(X_new.loc[select_patient, :])
            # X_new.to_csv('/app/HELLOJI.csv', index=False)
            # print ('oHELKLO')
            # X_new.loc[select_patient, :] =  [np.nan, 'definite', 'bulbar', 'bulbar', np.nan, 2, 92, 75, 0, 72.833, 314]
            feature_print = X_new.loc[select_patient, :].fillna('Not available')
            # feature_print.iloc[:, 1] = ['never', 'definite', 'bulbar', 'bulbar', 'Not available', '2']
            feature_print.index = feature_print.index.map(lambda x: feature_mapping.get(x, x))
            feature_print = feature_print.reset_index()
            feature_print.columns = ["Feature Name", "Feature Value"]
            # feature_print.
            predicted_prob = defaultdict(list)
            predicted_class = -1
            max_val = -1
            for key, val in M_dict.items():
                predicted_prob['predicted_probability'].append(
                    val.predict(xgb.DMatrix(X.loc[select_patient, :].values.reshape(1, -1), feature_names=X.columns))[
                        0])
                predicted_prob['classname'].append(key)
                if predicted_prob['predicted_probability'][-1] > max_val:
                    predicted_class = key
                    max_val = predicted_prob['predicted_probability'][-1]
            K = pd.DataFrame(predicted_prob)
            K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
            K['color'] = ['zed' if i == predicted_class else 'red' for i in list(predicted_prob['classname'])]
            t1 = pd.DataFrame(X.loc[select_patient, :]).T
            t2 = pd.DataFrame(X_new.loc[select_patient, :].fillna('Not available')).T

        st.table(feature_print.round(2).set_index("Feature Name").astype(str))

        # fig = px.bar(K, x='predicted_probability', y='classname', color='color', width=500, height=400, orientation='h')
        # # fig = px.bar(K, y='predicted_probability', x=sorted(list(predicted_prob['classname'])), width=500, height=400)
        # fig.update_layout(
        #     legend=None,
        #     yaxis_title="Class Labels",
        #     xaxis_title="Predicted Probability",
        #     font=dict(
        #         family="Courier New, monospace",
        #         size=12,
        #         color="RebeccaPurple"
        #     ),
        #     margin=dict(l=10, r=10, t=10, b=10),
        # )
        # st.plotly_chart(fig)

        import altair as alt
        K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"})
        f = alt.Chart(K).mark_bar().encode(
                    y=alt.Y('Class Labels:N',sort=alt.EncodingSortField(field="Predicted Probability", order='descending')),
                    x=alt.X('Predicted Probability:Q', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('color', legend=None),
                ).properties(width=500, height=300)
        st.write(f)
        # st.write('#### Trajectory for Predicted Class')
        st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))

        @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
        def load_model5():
            with open('saved_models/trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
                new_train = pickle.load(f)
            return new_train
        new_train = load_model5()
        exval = new_train[2]['explainer_train'] 
        explainer_train = shap.TreeExplainer(M_dict[predicted_class])

        shap_values_train = explainer_train.shap_values(t1)
        t1 = t2.copy() # ndfl.copy().fillna('Not available')
        t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
        # st.write(t1)
        # fig, ax = plt.subplots()

        # plt.savefig("/app/mar4_force_plot.pdf", bbox_inches='tight')
        # plt.savefig("/app/mar4_force_plot.eps", bbox_inches='tight')
        st.pyplot(shap.force_plot(exval, shap_values_train, t1.round(2), matplotlib=True, link='logit', contribution_threshold=0.10))
        fig, ax = plt.subplots()
        t2.columns = t2.columns.map(lambda x: feature_mapping.get(x, x))
        # st.write(t2.columns)
        # st.write(feature_mapping)
        r = shap.decision_plot(exval, shap_values_train, t2.round(2), link='logit', return_objects=True, new_base_value=0, highlight=0)
        st.pyplot(fig)
        # fig.savefig('/app/mar4_decisionplot.pdf', bbox_inches='tight')
        # fig.savefig('/app/mar4_decisionplot.eps', bbox_inches='tight')
        # fig.savefig('/app/new_shap_values.pdf', bbox_inches='tight')
    if show_whatif:
        with col02:
            dfl = pd.DataFrame(new_feature_input)
            ndfl = dfl.copy()
            for key, val in col_dict_map.items():
                rval = {j:i for i,j in val.items()}
                ndfl[key] = ndfl[key].map(lambda x: rval.get(x, x))
            st.write('### Prediction with what-if analysis')
            t2 = ndfl.copy().fillna('Not available')
            feature_print_what = ndfl.iloc[0].fillna('Not available')
            feature_print_what.index = feature_print_what.index.map(lambda x: feature_mapping.get(x, x))
            feature_print_what = feature_print_what.reset_index()
            feature_print_what.columns = ["Feature Name", "Feature Value"] 
            selected = []
            for i in range(len(feature_print_what)):
                if feature_print.iloc[i]["Feature Value"] == feature_print_what.iloc[i]["Feature Value"]:
                    pass
                else:
                    selected.append(feature_print.iloc[i]["Feature Name"])

            # st.table(feature_print)

            st.table(feature_print_what.round(2).astype(str).set_index("Feature Name").style.apply(lambda x: ['background: yellow' if (x.name in selected) else 'background: lightgreen' for i in x], axis=1))
            dfl = dfl[X.columns].replace('Not available', np.nan)
            predicted_prob = defaultdict(list)
            predicted_class = -1
            max_val = -1
            for key, val in M_dict.items():
                predicted_prob['predicted_probability'].append(val.predict(xgb.DMatrix(dfl.iloc[0, :].values.reshape(1, -1), feature_names=dfl.columns))[0])
                predicted_prob['classname'].append(key)
                if predicted_prob['predicted_probability'][-1] > max_val:
                    predicted_class = key
                    max_val = predicted_prob['predicted_probability'][-1] 
            K = pd.DataFrame(predicted_prob)
            K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
            K['color'] = ['zed' if i==predicted_class else 'red' for i in list(predicted_prob['classname']) ]
            import altair as alt
            K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"})
            f = alt.Chart(K).mark_bar().encode(
                y=alt.Y('Class Labels:N',sort=alt.EncodingSortField(field="Predicted Probability", order='descending')),
                    x=alt.X('Predicted Probability:Q', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('color', legend=None),
                ).properties( width=500, height=300)
            st.write(f)
            # fig = px.bar(K, x='predicted_probability', y='classname', color='color', width=500, height=400, orientation='h')
            # # fig = px.bar(K, y='predicted_probability', x=sorted(list(predicted_prob['classname'])), width=500, height=400)
            # fig.update_layout(
            # legend=None,
            # yaxis_title="Class Labels",
            # xaxis_title="Predicted Probability",
            # font=dict(
            #     family="Courier New, monospace",
            #     size=12,
            #     color="RebeccaPurple"
            # ),
            # margin=dict(l=10, r=10, t=10, b=10),
            # )  
            # st.plotly_chart(fig)
            st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))

            @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
            def load_model6():
                with open('saved_models/trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
                    new_train = pickle.load(f)
                return new_train

            new_train = load_model6()
            exval = new_train[2]['explainer_train']
            explainer_train = shap.TreeExplainer(M_dict[predicted_class])


            t1 = dfl.copy()
            shap_values_train = explainer_train.shap_values(t1)
            # t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
            t1 = t2.copy()  # ndfl.copy().fillna('Not available')
            t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
            # fig, ax = plt.subplots()
            st.pyplot ( shap.force_plot(exval, shap_values_train, t1.round(2), matplotlib=True, link='logit', contribution_threshold=0.10) )
            # st.pyplot()
            # plt.savefig("/app/mar4_force_plot.pdf", bbox_inches='tight')
            # plt.savefig("/app/mar4_force_plot.eps", bbox_inches='tight')

            # _ = shap.force_plot(exval, shap_values_train, t1, matplotlib=True, show=False, link='logit')
            # st.pyplot(shap.force_plot(exval, shap_values_train, t1, matplotlib=True,  link='logit'))
            # fig.savefig('/app/force_plot_new_shap_values_whatif.pdf', bbox_inches='tight')
            # fig.savefig('/app/force_plot_new_shap_values_whatif.eps', bbox_inches='tight')
            fig, ax = plt.subplots()
            t2.columns = t2.columns.map(lambda x: feature_mapping.get(x, x))
            # ndfl.columns = ndfl.columns.map(lambda x: feature_mapping.get(x, x))
            shap.decision_plot(exval, shap_values_train, t2.round(2), link='logit', feature_order=r.feature_idx, return_objects=True, new_base_value=0, highlight=0)
            # fig.savefig('/app/mar4_decisionplot.pdf', bbox_inches='tight')
            # fig.savefig('/app/mar4_decisionplot.eps', bbox_inches='tight')
            st.pyplot()


    # st.write('### Force Plots')
    # patient_name = st.selectbox('Select patient id', options=list(patient_index))
    # sample_id = patient_index.index(patient_name)
    # col8, col9 = st.columns(2)
    # with col8:
    #     st.info('Actual Label: ***{}***'.format('PD' if labels_actual[sample_id]==1 else 'HC'))
    #     st.info('Predicted PD class Probability: ***{}***'.format(round(float(labels_pred[sample_id]), 2)))
    # with col9:
    #     shap.force_plot(exval, shap_values[sample_id,:], X.iloc[sample_id,:], show=False, matplotlib=True)
    #     st.pyplot()
    
    # col10, col11 = st.columns(2)
    # with col10:
    #     fig, ax = plt.subplots()
    #     shap.decision_plot(exval, shap_values[sample_id], X.iloc[sample_id], link='logit', highlight=0, new_base_value=0)
    #     st.pyplot()



# fig = px.pie(pd.DataFrame(predicted_prob), values='predicted_probability', names='classname', color='classname', color_discrete_map=color_discrete_map)
        # fig.update_layout(legend=dict(
        #         yanchor="top",
        #         y=0.99,
        #         xanchor="right",
        #         x=1.05
        #     ))
