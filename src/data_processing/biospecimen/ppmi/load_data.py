import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

import pandas as pd
import pandas as pd
from sklearn import preprocessing
import math
from definitions import ROOT_DIR_INSIDE
PPMI_CLINICAL_RAW_DATA_DIR_INSIDE = ROOT_DIR_INSIDE / 'raw_data/clinical/ppmi'
import numpy as np
def funk(x):
    x = list(x.dropna())
    out = []
    for i in x:
        try:
            out.append(float(i))
        except:
            if type(i) is str:
                return i
    return np.mean(out)

def convert_to_timeseq(data_visits, feats_involving_time, visits_of_interest=None, comb_bl_sc = True):
    data = {}
    for feat in feats_involving_time:
        if feat == 'vital_signs':
            data[feat] = data_visits[feat].sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
        else:
            temp = data_visits[feat][['TESTVALUE']]
            temp.columns = [feat]
            data[feat] = temp.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
        if comb_bl_sc and ('BL' in data[feat].columns.get_level_values(1) and 'SC' in data[feat].columns.get_level_values(1)):
            for col in list(set(data[feat].columns.get_level_values(0))):
                data[feat].loc[:, (data[feat].columns.get_level_values(0)==col) & (data[feat].columns.get_level_values(1).isin(['BL']))] = data[feat].loc[:, (data[feat].columns.get_level_values(0)==col) & (data[feat].columns.get_level_values(1).isin(['BL', 'SC']))].apply(funk, axis=1)
        if visits_of_interest:
            data[feat] = data[feat].loc[:, data[feat].columns.get_level_values(1).isin(visits_of_interest)]
    return data

def read_from_raw_files(data_dir, csf_feats=None, plasma_feats=None,
                        serum_feats=None, urine_feats=None, whole_blood_feats=None):
    data_dir = str(PPMI_CLINICAL_RAW_DATA_DIR_INSIDE / data_dir)
    data = {}
    columns = ["PATNO", "EVENT_ID", "WGTKG", "HTCM", "TEMPC", 'SYSSUP', 'DIASUP', 'HRSUP', 'SYSSTND', 'DIASTND',
               'HRSTND']
    vital_signs = pd.read_csv("{}/Medical/Vital_Signs.csv".format(data_dir), index_col=["PATNO", "EVENT_ID"],
                              usecols=columns).reset_index().drop_duplicates(["PATNO", 'EVENT_ID']).set_index(
        ['PATNO', 'EVENT_ID'])
    data['vital_signs'] = vital_signs
    columns = ["PATNO", "SXMO", "SXYEAR"]  # , "PDDXDT"]  # first symptom onset month, year, diagnosis date
    pd_start = pd.read_csv("{}/Medical/PD_Features.csv".format(data_dir), index_col=["PATNO"],
                           usecols=columns).reset_index().drop_duplicates(["PATNO"]).set_index(['PATNO'])
    data['pd_start'] = pd_start
    columns = ["PATNO", "EVENT_ID", "PDMEDYN", "ONLDOPA", "ONDOPAG", "ONOTHER"]
    pd_medication = pd.read_csv("{}/Medical/Use_of_PD_Medication.csv".format(data_dir),
                                index_col=["PATNO", "EVENT_ID"], usecols=columns)
    data['pd_medication'] = pd_medication
    columns = ["PATNO", "BIOMOM", "BIOMOMPD", "BIODAD", "BIODADPD", "FULSIB", "FULSIBPD", "HAFSIB", "HAFSIBPD",
               "MAGPAR", "MAGPARPD", "PAGPAR", "PAGPARPD", "MATAU", "MATAUPD", "PATAU", "PATAUPD", "KIDSNUM", "KIDSPD"]
    feats_to_remove = ['BIOMOM', 'BIODAD', 'HAFSIB', 'HAFSIBPD', 'MAGPAR', 'PAGPAR', 'KIDSPD']
    columns = [col for col in columns if col not in feats_to_remove]
    family_history = pd.read_csv("{}/Subject Characteristics/Family_History__PD_.csv".format(data_dir),
                                 index_col=["PATNO"], usecols=columns).reset_index().drop_duplicates(
        ["PATNO"]).set_index(['PATNO'])
    data['family_history'] = family_history
    columns = ["PATNO", "EDUCYRS", "HANDED"]
    socio = pd.read_csv("{}/Subject Characteristics/Socio-Economics.csv".format(data_dir), index_col=["PATNO"],
                        usecols=columns).reset_index().drop_duplicates(["PATNO"]).set_index(['PATNO'])
    data['socio'] = socio
    columns = ["PATNO", "BIRTHDT", "GENDER", "HISPLAT", "RAINDALS", "RAASIAN", "RABLACK", "RAHAWOPI", "RAWHITE",
               "RANOS", 'ORIG_ENTRY']
    screening = pd.read_csv("{}/Subject Characteristics/Screening___Demographics.csv".format(data_dir),
                            index_col=["PATNO"], usecols=columns,
                            parse_dates=['BIRTHDT', 'ORIG_ENTRY']).reset_index().drop_duplicates(["PATNO"]).set_index(
        ['PATNO'])
    screening['age_at_screening'] = ((screening['ORIG_ENTRY'] - screening['BIRTHDT']).dt.days) // 365
    del screening['ORIG_ENTRY']
    del screening['BIRTHDT']
    data['screening'] = screening
    columns = ["PATNO", "CLINICAL_EVENT", "TYPE", "TESTNAME", "TESTVALUE", "UNITS"]
    biospecimen = pd.read_csv("{}/Biospecimen Analysis/Biospecimen_Analysis_Results.csv".format(data_dir),
                              index_col=["PATNO"], usecols=columns)
    biospecimen["CLINICAL_EVENT"].replace(
        ['Baseline Collection', 'Screening Visit', 'Visit 01', 'Visit 02', 'Visit 03', 'Visit 04', 'Visit 05',
         'Visit 06', 'Visit 07', 'Visit 08', 'Visit 09', 'Visit 10', 'Visit 11'],
        ['BL', 'SC', 'V01', 'V02', 'V03', 'V04', 'V06', 'V07', 'V08', 'V09', 'V10', 'V11', 'V12'], inplace=True)
    biospecimen['TESTNAME'] = biospecimen['TESTNAME'].map(lambda x: x.lower())
    biospecimen.rename(columns={'CLINICAL_EVENT': 'EVENT_ID'}, inplace=True)
    data['biospecimen_full'] = biospecimen
    genetics = pd.read_csv("{}/Genetic Analysis/genetic_analysis.csv".format(data_dir),
                           index_col=["PATNO"]).fillna(0)
    data['genetics'] = genetics
    all_feats = ["apoe genotype", "apoe_genotype", "csf alpha-synuclein", "a-synuclein", "ttau", "abeta 1-42", "ptau",
                 "grs", "csf hemoglobin"]
    # new_feats = ["gcase activity", "nfl", 'total sm', 'total gl2', 'total ceramide', 'total cholesterol', 'hdl', 'egf elisa', 'ldl', 'triglycerides',]
    nbiospecimen = biospecimen[biospecimen['TESTNAME'].isin(all_feats)]
    del nbiospecimen['UNITS']
    for feat in all_feats:
        temp = nbiospecimen[nbiospecimen['TESTNAME'] == feat]
        if len(temp) > 0 and feat not in ["apoe genotype", "apoe_genotype"]:
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'] = pd.to_numeric(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'], errors='coerce')
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].fillna(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].min(), inplace=True)
    csf = nbiospecimen  # biospecimen[(biospecimen["TYPE"] == 'Cerebrospinal Fluid') & ~(biospecimen["TESTVALUE"] == "below detection limit")][["EVENT_ID", "TESTNAME", "TESTVALUE"]]

    hemoglobin = csf[csf["TESTNAME"] == "csf hemoglobin"].reset_index().drop_duplicates(
        ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
    alpha_syn = csf[csf["TESTNAME"].isin(["csf alpha-synuclein"])].reset_index().drop_duplicates(
        ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
    total_tau = csf[csf["TESTNAME"] == "ttau"].reset_index().drop_duplicates(
        ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
    abeta_42 = csf[csf["TESTNAME"] == "abeta 1-42"].reset_index().drop_duplicates(
        ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
    p_tau181p = csf[csf["TESTNAME"] == "ptau"].reset_index().drop_duplicates(
        ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
    grs = csf[csf["TESTNAME"] == "grs"].reset_index().drop_duplicates(["PATNO", "EVENT_ID", "TESTNAME"]).set_index(
        ['PATNO', 'EVENT_ID'])
    apoe_genetics = csf[csf['TESTNAME'].isin(["apoe genotype", "apoe_genotype"])].reset_index().drop_duplicates(
        ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
    if True:
        KLP = apoe_genetics.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        apoe_genetics = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        apoe_genetics = apoe_genetics.reset_index().set_index(['PATNO'])
        del apoe_genetics['EVENT_ID']
        del apoe_genetics['TYPE']
        del apoe_genetics['TESTNAME']
        apoe_genetics.columns = ['apoe_genetics']
        KLP = grs.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        grs = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        KLP = hemoglobin.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        hemoglobin = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        KLP = alpha_syn.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        alpha_syn = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        KLP = total_tau.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        total_tau = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        KLP = abeta_42.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        abeta_42 = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        KLP = p_tau181p.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        p_tau181p = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
    data['csf'] = csf
    data['apoe_genetics'] = apoe_genetics
    data["csf_hemoglobin"] = hemoglobin
    data["csf_alpha_syn"] = alpha_syn
    data["csf_total_tau"] = total_tau
    data["csf_abeta_42"] = abeta_42
    data["csf_p_tau181p"] = p_tau181p
    data['dna_grs'] = grs

    csf_data = {}
    nbiospecimen = biospecimen[
        (biospecimen["TYPE"] == 'Cerebrospinal Fluid') & (biospecimen['TESTNAME'].isin(csf_feats))]
    del nbiospecimen['UNITS']
    for feat in csf_feats:
        temp = nbiospecimen[nbiospecimen['TESTNAME'] == feat]
        if len(temp) > 0 and feat not in ["apoe genotype", "apoe_genotype"]:
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'] = pd.to_numeric(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'], errors='coerce')
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].fillna(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].min(), inplace=True)
    ncsf = nbiospecimen.copy()
    for feat in csf_feats:
        temp = ncsf[ncsf["TESTNAME"] == feat].reset_index().drop_duplicates(
            ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
        KLP = temp.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        temp = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        csf_data['csf_' + feat] = temp
    plasma_data = {}
    nbiospecimen = biospecimen[(biospecimen["TYPE"] == 'Plasma') & (biospecimen['TESTNAME'].isin(plasma_feats))]
    del nbiospecimen['UNITS']
    for feat in plasma_feats:
        temp = nbiospecimen[nbiospecimen['TESTNAME'] == feat]
        if len(temp) > 0 and feat not in ["apoe genotype", "apoe_genotype"]:
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'] = pd.to_numeric(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'], errors='coerce')
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].fillna(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].min(), inplace=True)
    ncsf = nbiospecimen.copy()
    for feat in plasma_feats:
        temp = ncsf[ncsf["TESTNAME"] == feat].reset_index().drop_duplicates(
            ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
        KLP = temp.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        temp = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        plasma_data['plasma_' + feat] = temp
    serum_data = {}
    nbiospecimen = biospecimen[(biospecimen["TYPE"] == 'Serum') & (biospecimen['TESTNAME'].isin(serum_feats))]
    del nbiospecimen['UNITS']
    for feat in serum_feats:
        temp = nbiospecimen[nbiospecimen['TESTNAME'] == feat]
        if len(temp) > 0 and feat not in ["apoe genotype", "apoe_genotype"]:
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'] = pd.to_numeric(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'], errors='coerce')
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].fillna(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].min(), inplace=True)
    ncsf = nbiospecimen.copy()
    for feat in serum_feats:
        temp = ncsf[ncsf["TESTNAME"] == feat].reset_index().drop_duplicates(
            ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
        KLP = temp.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        temp = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        serum_data['serum_' + feat] = temp
    urine_data = {}
    nbiospecimen = biospecimen[(biospecimen["TYPE"] == 'Urine') & (biospecimen['TESTNAME'].isin(urine_feats))]
    del nbiospecimen['UNITS']
    for feat in urine_feats:
        temp = nbiospecimen[nbiospecimen['TESTNAME'] == feat]
        if len(temp) > 0 and feat not in ["apoe genotype", "apoe_genotype"]:
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'] = pd.to_numeric(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'], errors='coerce')
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].fillna(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].min(), inplace=True)
    ncsf = nbiospecimen.copy()
    for feat in urine_feats:
        temp = ncsf[ncsf["TESTNAME"] == feat].reset_index().drop_duplicates(
            ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
        KLP = temp.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        temp = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        urine_data['urine_' + feat] = temp
    whole_blood_data = {}
    nbiospecimen = biospecimen[
        (biospecimen["TYPE"] == 'Whole Blood') & (biospecimen['TESTNAME'].isin(whole_blood_feats))]
    del nbiospecimen['UNITS']
    for feat in whole_blood_feats:
        temp = nbiospecimen[nbiospecimen['TESTNAME'] == feat]
        if len(temp) > 0 and feat not in ["apoe genotype", "apoe_genotype"]:
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'] = pd.to_numeric(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'], errors='coerce')
            nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].fillna(
                nbiospecimen.loc[nbiospecimen['TESTNAME'] == feat, 'TESTVALUE'].min(), inplace=True)
    ncsf = nbiospecimen.copy()
    for feat in whole_blood_feats:
        temp = ncsf[ncsf["TESTNAME"] == feat].reset_index().drop_duplicates(
            ["PATNO", "EVENT_ID", "TESTNAME"]).set_index(['PATNO', 'EVENT_ID'])
        KLP = temp.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        temp = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        whole_blood_data['whole_blood_' + feat] = temp

    return data, csf_data, plasma_data, serum_data, urine_data, whole_blood_data