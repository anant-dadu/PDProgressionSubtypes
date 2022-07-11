import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import os
import copy
from collections import defaultdict
from scipy.stats import zscore
warnings.filterwarnings('ignore')
from definitions import ROOT_DIR_INSIDE


def read_from_raw_files_old(data_dir_rel, merge_screen_baseline=[]):
    data_dir = str(ROOT_DIR_INSIDE / 'raw_data/clinical/ppmi' / data_dir_rel)
    # data_dir = str(PPMI_CLINICAL_RAW_DATA_DIR_INSIDE/ data_dir_rel)
    cols = {}
    # Medical-Neurological Exam
    cols["neuro_cranial"] = ["PATNO", "EVENT_ID", "CN2RSP", "CN346RSP", "CN5RSP", "CN7RSP", "CN8RSP", "CN910RSP", "CN11RSP", "CN12RSP"]
    # UPDATED
    # cols["neuro_cranial"] = ["PATNO", "EVENT_ID", "CN1RSP"]# , "CN346RSP", "CN5RSP", "CN7RSP", "CN8RSP", "CN910RSP", "CN11RSP", "CN12RSP"]
    # lh
    neuro_cranial = pd.read_csv(
        "{}/Medical-Neurological Exam/Neurological_Exam_-_Cranial_Nerves.csv".format(data_dir),
        index_col=["PATNO", "EVENT_ID"], usecols=cols["neuro_cranial"])
    # neuro_cranial['neuro_total'] = neuro_cranial.sum(axis=1)

    # Motor Assessments
    cols["updrs1"] = ["PATNO", "EVENT_ID", "INFODT", "NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS"]
    cols["updrs1pq"] = ["PATNO", "EVENT_ID", "NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD",
                        "NP1FATG"]
    cols["updrs2pq"] = ["PATNO", "EVENT_ID", "NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN", "NP2HWRT",
                        "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK", "NP2FREZ"]
    cols["updrs3_temp"] = ["PATNO", "EVENT_ID", "PAG_NAME", "CMEDTM", "EXAMTM", "NP3SPCH", "NP3FACXP", "NP3RIGN",
                           "NP3RIGRU", "NP3RIGLU", "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR",
                           "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL",
                           "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR",
                           "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL",
                           "NP3RTALJ", "NP3RTCON", "DYSKPRES", "DYSKIRAT", "NHY", "ANNUAL_TIME_BTW_DOSE_NUPDRS",
                           "ON_OFF_DOSE", "PD_MED_USE"]
    cols["updrs3"] = ['PAG_NAME', "ON_OFF_DOSE", "PATNO", "EVENT_ID", "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",
                      "NP3RIGLU", "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR",
                      "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT",
                      "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU",
                      "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON"]
    cols["updrs4"] = ["PATNO", "EVENT_ID", "NP4WDYSK", "NP4DYSKI", "NP4OFF", "NP4FLCTI", "NP4FLCTX", "NP4DYSTN"]
    cols["schwab"] = ["PATNO", "EVENT_ID", "MSEADLG"]
    cols["pase_house"] = ["PATNO", "EVENT_ID", "LTHSWRK", "HVYHSWRK", "HMREPR", "LAWNWRK", "OUTGARDN", "CAREGVR",
                          "WRKVL", "WRKVLHR", "WRKVLACT"]
    updrs1 = pd.read_csv("{}/Motor Assessments/MDS_UPDRS_Part_I.csv".format(data_dir),
                         index_col=["PATNO", "EVENT_ID"], parse_dates=["INFODT"], usecols=cols["updrs1"])
    updrs1pq = pd.read_csv("{}/Motor Assessments/MDS_UPDRS_Part_I__Patient_Questionnaire.csv".format(data_dir),
                           index_col=["PATNO", "EVENT_ID"], usecols=cols["updrs1pq"])
    updrs2pq = pd.read_csv("{}/Motor Assessments/MDS_UPDRS_Part_II__Patient_Questionnaire.csv".format(data_dir),
                           index_col=["PATNO", "EVENT_ID"], usecols=cols["updrs2pq"])
    if data_dir == '2016-05-25 data':
        updrs3_temp = pd.read_csv("{}/Motor Assessments/MDS_UPDRS_Part_III__Post_Dose_.csv".format(data_dir),
                                  index_col=["PATNO", "EVENT_ID"], usecols=cols["updrs3_temp"])
        updrs3 = updrs3_temp[updrs3_temp.PAG_NAME == 'NUPDRS3']  # before dose
        updrs3a = updrs3_temp[updrs3_temp.PAG_NAME == 'NUPDRS3A']  # after dose
    else:
        required_cols = set(
            ["PATNO", "EVENT_ID", "PAG_NAME", "CMEDTM", "EXAMTM", "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",
             "NP3RIGLU", "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR", "NP3PRSPL",
             "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR",
             "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL",
             "NP3RTALJ", "NP3RTCON", "DYSKPRES", "DYSKIRAT", "NHY", "ANNUAL_TIME_BTW_DOSE_NUPDRS", "ON_OFF_DOSE",
             "PD_MED_USE"]).difference(
            set(['PAG_NAME', 'CMEDTM', 'EXAMTM', 'PD_MED_USE', 'ON_OFF_DOSE', 'ANNUAL_TIME_BTW_DOSE_NUPDRS']))
        updrs3 = pd.read_csv("{}/Motor Assessments/MDS_UPDRS_Part_III.csv".format(data_dir),
                             index_col=["PATNO", "EVENT_ID"])
        # updrs3 = updrs3[updrs3['PAG_NAME'] == 'NUPDRS3'].drop(['ON_OFF_DOSE', 'PAG_NAME'], axis=1) #updrs3[~updrs3['ON_OFF_DOSE'].isin([1])]# .drop(['ON_OFF_DOSE'], axis=1)
        # updrs3 = updrs3[updrs3['PAG_NAME'].isin(['NUPDR3OF', 'NUPDRS3'])].reset_index()[list(required_cols)].set_index(
        updrs3 = updrs3[updrs3['PAG_NAME'].isin(['NUPDRS3'])].reset_index()[list(required_cols)].set_index(
            ["PATNO", "EVENT_ID"])
    updrs4 = pd.read_csv("{}/Motor Assessments/MDS_UPDRS_Part_IV.csv".format(data_dir),
                         index_col=["PATNO", "EVENT_ID"], usecols=cols["updrs4"])
    schwab = pd.read_csv("{}/Motor Assessments/Modified_Schwab_+_England_ADL.csv".format(data_dir),
                         index_col=["PATNO", "EVENT_ID"], usecols=cols["schwab"])
    pase_house = pd.read_csv("{}/Motor Assessments/PASE_-_Household_Activity.csv".format(data_dir),
                             index_col=["PATNO", "EVENT_ID"], usecols=cols["pase_house"])
    # Non-motor Assessments
    cols["benton"] = ["PATNO", "EVENT_ID", "JLO_TOTRAW"]
    cols["cog_catg"] = ["PATNO", "EVENT_ID", "COGDECLN", "FNCDTCOG"]
    cols["epworth"] = ["PATNO", "EVENT_ID", "ESS1", "ESS2", "ESS3", "ESS4", "ESS5", "ESS6", "ESS7", "ESS8"]
    cols["geriatric"] = ["PATNO", "EVENT_ID", "GDSSATIS", "GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSGSPIR", "GDSAFRAD",
                         "GDSHAPPY", "GDSHLPLS", "GDSHOME", "GDSMEMRY", "GDSALIVE", "GDSWRTLS", "GDSENRGY", "GDSHOPLS",
                         "GDSBETER"]
    cols["geriatric_pos"] = ["GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSAFRAD", "GDSHLPLS", "GDSHOME", "GDSMEMRY",
                             "GDSWRTLS", "GDSHOPLS", "GDSBETER"]
    cols["geriatric_neg"] = ["GDSSATIS", "GDSGSPIR", "GDSHAPPY", "GDSALIVE", "GDSENRGY"]
    cols["hopkins_verbal"] = ["PATNO", "EVENT_ID", "HVLTRT1", "HVLTRT2", "HVLTRT3", "HVLTRDLY", "HVLTREC"]#, "HVLTFPRL", "HVLTFPUN"]
    cols["hopkins_verbal_pos"] = ["HVLTRT1", "HVLTRT2", "HVLTRT3", "HVLTRDLY", "HVLTREC"]
    cols["hopkins_verbal_neg"] = ["HVLTFPRL", "HVLTFPUN"]
    cols["letter_seq"] = ["PATNO", "EVENT_ID", "LNS_TOTRAW"]
    cols["moca"] = ["PATNO", "EVENT_ID", "MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH", "MCALION",
                    "MCARHINO", "MCACAMEL", "MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7", "MCASNTNC", "MCAVFNUM", "MCAVF",
                    "MCAABSTR", "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5", "MCADATE", "MCAMONTH", "MCAYR",
                    "MCADAY", "MCAPLACE", "MCACITY", "MCATOT"]
    cols["moca_visuospatial"] = ["MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH"]
    cols["moca_naming"] = ["MCALION", "MCARHINO", "MCACAMEL"]
    cols["moca_attention"] = ["MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7"]
    cols["moca_language"] = ["MCASNTNC", "MCAVF"]
    cols["moca_delayed_recall"] = ["MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5"]
    cols["moca_orientation"] = ["MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY"]
    cols["upsit"] = ["SUBJECT_ID", "SCENT_10_RESPONSE", "SCENT_09_RESPONSE", "SCENT_08_RESPONSE", "SCENT_07_RESPONSE",
                     "SCENT_06_RESPONSE", "SCENT_05_RESPONSE", "SCENT_04_RESPONSE", "SCENT_03_RESPONSE",
                     "SCENT_02_RESPONSE", "SCENT_01_RESPONSE", "SCENT_20_RESPONSE", "SCENT_19_RESPONSE",
                     "SCENT_18_RESPONSE", "SCENT_17_RESPONSE", "SCENT_16_RESPONSE", "SCENT_15_RESPONSE",
                     "SCENT_14_RESPONSE", "SCENT_13_RESPONSE", "SCENT_12_RESPONSE", "SCENT_11_RESPONSE",
                     "SCENT_30_RESPONSE", "SCENT_29_RESPONSE", "SCENT_28_RESPONSE", "SCENT_27_RESPONSE",
                     "SCENT_26_RESPONSE", "SCENT_25_RESPONSE", "SCENT_24_RESPONSE", "SCENT_23_RESPONSE",
                     "SCENT_22_RESPONSE", "SCENT_21_RESPONSE", "SCENT_40_RESPONSE", "SCENT_39_RESPONSE",
                     "SCENT_38_RESPONSE", "SCENT_37_RESPONSE", "SCENT_36_RESPONSE", "SCENT_35_RESPONSE",
                     "SCENT_34_RESPONSE", "SCENT_33_RESPONSE", "SCENT_32_RESPONSE", "SCENT_31_RESPONSE",
                     "SCENT_10_CORRECT", "SCENT_09_CORRECT", "SCENT_08_CORRECT", "SCENT_07_CORRECT", "SCENT_06_CORRECT",
                     "SCENT_05_CORRECT", "SCENT_04_CORRECT", "SCENT_03_CORRECT", "SCENT_02_CORRECT", "SCENT_01_CORRECT",
                     "SCENT_20_CORRECT", "SCENT_19_CORRECT", "SCENT_18_CORRECT", "SCENT_17_CORRECT", "SCENT_16_CORRECT",
                     "SCENT_15_CORRECT", "SCENT_14_CORRECT", "SCENT_13_CORRECT", "SCENT_12_CORRECT", "SCENT_11_CORRECT",
                     "SCENT_30_CORRECT", "SCENT_29_CORRECT", "SCENT_28_CORRECT", "SCENT_27_CORRECT", "SCENT_26_CORRECT",
                     "SCENT_25_CORRECT", "SCENT_24_CORRECT", "SCENT_23_CORRECT", "SCENT_22_CORRECT", "SCENT_21_CORRECT",
                     "SCENT_40_CORRECT", "SCENT_39_CORRECT", "SCENT_38_CORRECT", "SCENT_37_CORRECT", "SCENT_36_CORRECT",
                     "SCENT_35_CORRECT", "SCENT_34_CORRECT", "SCENT_33_CORRECT", "SCENT_32_CORRECT", "SCENT_31_CORRECT",
                     "TOTAL_CORRECT"]
    # cols["quip"] = ["PATNO", "EVENT_ID", "TMGAMBLE", "CNTRLGMB", "TMSEX", "CNTRLSEX", "TMBUY", "CNTRLBUY", "TMEAT",
    #                 "CNTRLEAT", "TMTORACT", "TMTMTACT", "TMTRWD"]
    cols["quip"] = ["PATNO", "EVENT_ID", "TMGAMBLE", "CNTRLGMB", "TMSEX", "CNTRLSEX", "TMBUY", "CNTRLBUY", "TMEAT",
                    "CNTRLEAT", "TMTORACT", "TMTMTACT", "TMTRWD"]
    cols["rem"] = ["PATNO", "EVENT_ID", "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL",
                   "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB", "STROKE", "HETRA", "RLS",
                   "NARCLPSY", "DEPRS", "EPILEPSY", "BRNINFM"]
    cols["aut"] = ["PATNO", "EVENT_ID", "SCAU1", "SCAU2", "SCAU3", "SCAU4", "SCAU5", "SCAU6", "SCAU7", "SCAU8", "SCAU9",
                   "SCAU10", "SCAU11", "SCAU12", "SCAU13", "SCAU14", "SCAU15", "SCAU16", "SCAU17", "SCAU18", "SCAU19",
                   "SCAU20", "SCAU21", "SCAU22", "SCAU23", "SCAU23A", "SCAU23AT", "SCAU24", "SCAU25", "SCAU26A",
                   "SCAU26AT", "SCAU26B", "SCAU26BT", "SCAU26C", "SCAU26CT", "SCAU26D", "SCAU26DT"]
    cols["aut_gastrointestinal_up"] = ["SCAU1", "SCAU2", "SCAU3"]
    cols["aut_gastrointestinal_down"] = ["SCAU4", "SCAU5", "SCAU6", "SCAU7"]
    cols["aut_urinary"] = ["SCAU8", "SCAU9", "SCAU10", "SCAU11", "SCAU12", "SCAU13"]
    cols["aut_cardiovascular"] = ["SCAU14", "SCAU15", "SCAU16"]
    cols["aut_thermoregulatory"] = ["SCAU17", "SCAU18"]
    cols["aut_pupillomotor"] = ["SCAU19"]
    cols["aut_skin"] = ["SCAU20", "SCAU21"]
    cols["aut_sexual"] = ["SCAU22", "SCAU23", "SCAU24",
                          "SCAU25"]  # 9 for NA, might skew the results signific for M/F better to remove
    cols["semantic"] = ["PATNO", "EVENT_ID", "VLTANIM", "VLTVEG", "VLTFRUIT"]
    cols["stai"] = ["PATNO", "EVENT_ID", "STAIAD1", "STAIAD2", "STAIAD3", "STAIAD4", "STAIAD5", "STAIAD6", "STAIAD7",
                    "STAIAD8", "STAIAD9", "STAIAD10", "STAIAD11", "STAIAD12", "STAIAD13", "STAIAD14", "STAIAD15",
                    "STAIAD16", "STAIAD17", "STAIAD18", "STAIAD19", "STAIAD20", "STAIAD21", "STAIAD22", "STAIAD23",
                    "STAIAD24", "STAIAD25", "STAIAD26", "STAIAD27", "STAIAD28", "STAIAD29", "STAIAD30", "STAIAD31",
                    "STAIAD32", "STAIAD33", "STAIAD34", "STAIAD35", "STAIAD36", "STAIAD37", "STAIAD38", "STAIAD39",
                    "STAIAD40"]
    cols["stai_a_state_pos"] = ["STAIAD3", "STAIAD4", "STAIAD6", "STAIAD7", "STAIAD9", "STAIAD12", "STAIAD13",
                                "STAIAD14", "STAIAD17", "STAIAD18"]
    # STAIAD1: I feel calm.. -> no anxiety
    cols["stai_a_state_neg"] = ["STAIAD1", "STAIAD2", "STAIAD5", "STAIAD8", "STAIAD10", "STAIAD11", "STAIAD15",
                                "STAIAD16", "STAIAD19", "STAIAD20"]
    cols["stai_a_trait_pos"] = ["STAIAD22", "STAIAD24", "STAIAD25", "STAIAD28", "STAIAD29", "STAIAD31", "STAIAD32",
                                "STAIAD35", "STAIAD37", "STAIAD38", "STAIAD40"]
    cols["stai_a_trait_neg"] = ["STAIAD21", "STAIAD23", "STAIAD26", "STAIAD27", "STAIAD30", "STAIAD33", "STAIAD34",
                                "STAIAD36", "STAIAD39"]
    cols["sdm"] = ["PATNO", "EVENT_ID", "SDMTOTAL"]
    # hh: higher healthy
    benton = pd.read_csv("{}/Non-motor Assessments/Benton_Judgment_of_Line_Orientation.csv".format(data_dir),
                         index_col=["PATNO", "EVENT_ID"], usecols=cols["benton"])
    # lh
    cog_catg = pd.read_csv("{}/Non-motor Assessments/Cognitive_Categorization.csv".format(data_dir),
                           index_col=["PATNO", "EVENT_ID"], usecols=cols["cog_catg"])
    # lh
    epworth = pd.read_csv("{}/Non-motor Assessments/Epworth_Sleepiness_Scale.csv".format(data_dir),
                          index_col=["PATNO", "EVENT_ID"], usecols=cols["epworth"])
    # lh
    geriatric = pd.read_csv("{}/Non-motor Assessments/Geriatric_Depression_Scale__Short_.csv".format(data_dir),
                            index_col=["PATNO", "EVENT_ID"], usecols=cols["geriatric"])
    geriatric["total_pos"] = geriatric[cols["geriatric_pos"]].sum(axis=1)  # score of you have depression
    geriatric["total_neg"] = geriatric[cols["geriatric_neg"]].sum(axis=1)  # score of not depression out of 5
    geriatric["total"] = geriatric["total_pos"] + 5 - geriatric["total_neg"]  # depression score
    geriatric = geriatric[["total"]]  # drop the rest # lh: lower healthy
    # hh
    # hopkins_verbal = pd.read_csv("{}/Non-motor Assessments/Hopkins_Verbal_Learning_Test.csv".format(data_dir), index_col=["PATNO", "EVENT_ID"], usecols=cols["hopkins_verbal"])
    # hopkins_verbal[cols["hopkins_verbal_neg"]] = -hopkins_verbal[cols["hopkins_verbal_neg"]]
    hopkins_verbal = pd.read_csv("{}/Non-motor Assessments/Hopkins_Verbal_Learning_Test.csv".format(data_dir),
                                 index_col=["PATNO", "EVENT_ID"], usecols=cols["hopkins_verbal"])
    # hh
    letter_seq = pd.read_csv("{}/Non-motor Assessments/Letter_-_Number_Sequencing__PD_.csv".format(data_dir),
                             index_col=["PATNO", "EVENT_ID"], usecols=cols["letter_seq"])
    # hh
    moca = pd.read_csv("{}/Non-motor Assessments/Montreal_Cognitive_Assessment__MoCA_.csv".format(data_dir),
                       index_col=["PATNO", "EVENT_ID"], usecols=cols["moca"])
    moca["visuospatial"] = moca[cols["moca_visuospatial"]].sum(axis=1)
    moca["naming"] = moca[cols["moca_naming"]].sum(axis=1)
    moca["attention"] = moca[cols["moca_attention"]].sum(axis=1)
    moca["language"] = moca[cols["moca_language"]].sum(axis=1)
    moca["delayed_recall"] = moca[cols["moca_delayed_recall"]].sum(axis=1)
    # moca = moca[["visuospatial", "attention", "language", "naming", "delayed_recall", "MCAABSTR", "MCAVFNUM", "MCATOT"]] # drop extra MCAVFNUM "MCATOT" , "MCAABSTR"
    # moca = moca[["visuospatial", "attention", "language", "MCAABSTR", "MCAVFNUM", "MCATOT", "naming"]] # drop extra MCAVFNUM "MCATOT" , "MCAABSTR"
    if 'del' in merge_screen_baseline:
        moca = moca[["visuospatial", "attention", "language", "delayed_recall", "MCAABSTR", "MCAVFNUM",
                     "MCATOT"]]  # drop extra MCAVFNUM "MCATOT" , "MCAABSTR"
    else:
        moca = moca[["visuospatial", "attention", "language", "delayed_recall", "naming", "MCAABSTR", "MCAVFNUM",
                     "MCATOT"]]  # drop extra MCAVFNUM "MCATOT" , "MCAABSTR"
    # hh
    upsit = pd.read_csv("{}/Non-motor Assessments/Olfactory_UPSIT.csv".format(data_dir),
                        index_col=["SUBJECT_ID"], usecols=cols["upsit"])
    # lh
    quip = pd.read_csv("{}/Non-motor Assessments/QUIP_Current_Short.csv".format(data_dir),
                       index_col=["PATNO", "EVENT_ID"], usecols=cols["quip"])
    # lh
    rem = pd.read_csv("{}/Non-motor Assessments/REM_Sleep_Disorder_Questionnaire.csv".format(data_dir),
                      index_col=["PATNO", "EVENT_ID"], usecols=cols["rem"])
    # lh
    aut = pd.read_csv("{}/Non-motor Assessments/SCOPA-AUT.csv".format(data_dir),
                      index_col=["PATNO", "EVENT_ID"], usecols=cols["aut"])
    aut["gastrointestinal_up"] = aut[cols["aut_gastrointestinal_up"]].sum(axis=1)
    aut["gastrointestinal_down"] = aut[cols["aut_gastrointestinal_down"]].sum(axis=1)
    aut["urinary"] = aut[cols["aut_urinary"]].sum(axis=1)
    aut["cardiovascular"] = aut[cols["aut_cardiovascular"]].sum(axis=1)
    aut["thermoregulatory"] = aut[cols["aut_thermoregulatory"]].sum(axis=1)
    aut["pupillomotor"] = aut[cols["aut_pupillomotor"]].sum(axis=1)
    aut["skin"] = aut[cols["aut_skin"]].sum(axis=1)
    # aut["sexual"] = aut[cols["aut_sexual"]].sum(axis=1) # NA is assigned as 9, throwing things off, in case adding it, edit the next line too
    aut = aut[["gastrointestinal_up", "gastrointestinal_down", "urinary", "cardiovascular", "thermoregulatory",
               "pupillomotor", "skin"]]  # , "sexual"]]
    # hh
    semantic = pd.read_csv("{}/Non-motor Assessments/Semantic_Fluency.csv".format(data_dir),
                           index_col=["PATNO", "EVENT_ID"], usecols=cols["semantic"])
    stai = pd.read_csv("{}/Non-motor Assessments/State-Trait_Anxiety_Inventory.csv".format(data_dir),
                       index_col=["PATNO", "EVENT_ID"], usecols=cols["stai"])
    # lh
    stai["a_state"] = stai[cols["stai_a_state_pos"]].sum(axis=1) + (
                5 * len(cols["stai_a_state_neg"]) - stai[cols["stai_a_state_neg"]].sum(axis=1))
    stai["a_trait"] = stai[cols["stai_a_trait_pos"]].sum(axis=1) + (
                5 * len(cols["stai_a_trait_neg"]) - stai[cols["stai_a_trait_neg"]].sum(axis=1))
    stai = stai[["a_state", "a_trait"]]
    # hh
    sdm = pd.read_csv("{}/Non-motor Assessments/Symbol_Digit_Modalities.csv".format(data_dir),
                      index_col=["PATNO", "EVENT_ID"], usecols=cols["sdm"])
    # Subject Characteristics
    cols["status"] = ["PATNO", "RECRUITMENT_CAT", "IMAGING_CAT", "ENROLL_DATE", "ENROLL_CAT"]
    cols["screening"] = ["PATNO", "BIRTHDT", "GENDER", "APPRDX", "CURRENT_APPRDX", "HISPLAT", "RAINDALS", "RAASIAN",
                         "RABLACK", "RAHAWOPI", "RAWHITE", "RANOS", 'ORIG_ENTRY']
    screening = pd.read_csv("{}/Subject Characteristics/Screening___Demographics.csv".format(data_dir),
                            index_col=["PATNO"], usecols=cols["screening"], parse_dates=['BIRTHDT', 'ORIG_ENTRY'])

    status = pd.read_csv("{}/Subject Characteristics/Patient_Status.csv".format(data_dir), index_col=["PATNO"],
                         usecols=cols["status"])
    if 'moca' in merge_screen_baseline:
        KLP = moca.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        moca = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
    if 'neuro_cranial' in merge_screen_baseline:
        KLP = neuro_cranial.sort_index().reset_index().replace('SC', 'BL').set_index(['PATNO', 'EVENT_ID'])
        neuro_cranial = KLP.loc[~KLP.index.duplicated(keep='first')].copy()
        neuro_cranial = neuro_cranial.replace({2: np.nan, 3: np.nan})

    # updrs1['updrs1_total'] = updrs1.sum(axis=1)
    # updrs1pq['updrs1pq_total'] = updrs1pq.sum(axis=1)
    # updrs2pq['updrs2pq_total'] = updrs2pq.sum(axis=1)
    # updrs3['updrs3_total'] = updrs3.sum(axis=1)
    # epworth['epworth_total'] = epworth.sum(axis=1)
    # rem['rem_total'] = rem.sum(axis=1)
    # aut['aut_total'] = aut.sum(axis=1)
    # import pdb; pdb.set_trace()
    data = {}
    data["benton"] = -(-benton)
    data["hopkins_verbal"] = -(-hopkins_verbal)
    data["letter_seq"] = -(-letter_seq)
    data["moca"] = -(-moca)
    data["upsit"] = -(-upsit)
    data["neuro_cranial"] = neuro_cranial
    data["updrs1"] = updrs1
    data["updrs1pq"] = updrs1pq
    data["updrs2pq"] = updrs2pq
    data["updrs3"] = updrs3
    data["updrs4"] = updrs4
    data["schwab"] = schwab
    data["pase_house"] = pase_house
    data["cog_catg"] = cog_catg
    data["epworth"] = epworth
    data["geriatric"] = geriatric
    data["quip"] = quip
    data["rem"] = rem
    data["aut"] = aut
    data["semantic"] = semantic  # -semantic
    data["stai"] = stai
    data["sdm"] = sdm  # -sdm
    data["status"] = status
    data["screening"] = screening
    return data


def test_measure(l_hc, l_pd, p_val_r=0.1):
    n_hc = len(l_hc)
    n_pd = len(l_pd)
    s_hc = np.std(l_hc)
    s_pd = np.std(l_pd)
    m_hc = np.mean(l_hc)
    m_pd = np.mean(l_pd)
    m_diff = m_hc - m_pd
    if 0.5<(s_hc/s_pd) and (s_hc/s_pd)<2:
        num = ((n_hc-1) * (s_hc**2)) + ((n_pd-1) * (s_pd**2))
        den = n_hc + n_pd - 2
        se_rt = (num / den)**0.5
        se = ( (((1/n_hc)+(1/n_pd)) ** 0.5)) * se_rt
    else:
        se = ((s_hc**2)/(n_hc) + (s_pd**2)/(n_pd))**0.5
    t_stat = m_diff / se
    df = n_hc + n_pd
    from scipy.stats import t
    p = 0.95
    df = n_hc + n_pd
    t_crit = t.ppf(p, df)
    p_val = (1 - t.cdf(t_stat, df))
    L_L = np.round(m_diff - t_crit * se, 2)
    L_U = np.round(m_diff + t_crit * se, 2)
    if p_val < p_val_r:
        return ('PASS', m_hc, m_pd, p_val, '[{},{}]'.format(L_L, L_U))
    else:
        return ('FAIL', m_hc, m_pd, p_val, '[{},{}]'.format(L_L, L_U))


def flipping_data(data, ECAT, files_of_interest, visit_id='BL', fil=0, remove_outlier=False):
    copy_data = copy.deepcopy(data)
    test_df = {i: {} for i in files_of_interest}
    for feat in files_of_interest:
        c = 0
        KP = data[feat].set_index(['PATNO'])
        DF = pd.merge(KP, ECAT, left_index=True, right_index=True)
        DF2 = DF[DF['EVENT_ID']==visit_id]
        DF2 = DF2.reset_index().set_index(['PATNO', 'EVENT_ID'])
        for col in list(DF2.columns):
            if col in ['ENROLL_CAT', 'INFODT', 'DYSKIRAT']:
                continue
            l_hc = list(DF2[DF2['ENROLL_CAT'] == 'HC'][col].dropna())
            l_pd = list(DF2[DF2['ENROLL_CAT'] == 'PD'][col].dropna())
            temp = np.array(data[feat][col].dropna())
            if fil == 0:
                decision, m_hc, m_pd, p_val, ci = test_measure(l_hc, l_pd, 0.05)
                decision = 'FAIL'
            else:
                decision, m_hc, m_pd, p_val, ci = test_measure(l_hc, l_pd, 0.05)
            if decision == 'PASS':# or feat in ['hopkins_verbal', 'moca']:
                test_df[feat][col] = str(np.round(p_val, 3)) + '**' + ' ' + ci
            else:
                test_df[feat][col] = str(np.round(p_val, 3)) + ' ' + ci
    test_df_df = defaultdict(list)
    for i, j in test_df.items():
            c = 0
            for k, m in j.items():
                test_df_df['attribute'].append(k)
                test_df_df['p value'].append(m)
                if c > 0:
                    test_df_df['feature'].append('')
                else:
                    test_df_df['feature'].append(i)
    P = pd.DataFrame(test_df_df)
    P = P.sort_values(by=['p value', 'feature']).set_index(pd.Index(range(len(P))))
    P['reverse'] = P['p value'].map(lambda x: '**' in x)
    # import pdb; pdb.set_trace()
    return copy_data, P

def load_required_files(data, datasets_of_interest, visits_of_interest, remove_outlier=False):
    last_visit = visits_of_interest[-1]
    # selecting participants with data from BL to last_visit
    cols_removed = ['PATNO', 'PAG_NAME','CMEDTM','EXAMTM','PD_MED_USE','ON_OFF_DOSE','ANNUAL_TIME_BTW_DOSE_NUPDRS', 'DYSKIRAT', 'EVENT_ID', 'INFODT']
    # removal of obvious features so they do not interfere during participant filtering
    all_columns_for_analysis = []
    for dataset in datasets_of_interest:
        dataset_noindx = data[dataset].reset_index()
        for col in dataset_noindx.columns:
            if not col in cols_removed:
                if remove_outlier:
                    temp_data = np.array( dataset_noindx[col].dropna() ).flatten()
                    upper_quartile = np.percentile(temp_data, 98)
                    lower_quartile = np.percentile(temp_data, 2)
                    IQR = (upper_quartile - lower_quartile) * 0
                    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
                    val_low = quartileSet[0]
                    val_high = quartileSet[1]
                    if val_high == val_low:
                        # print (col, val_low, val_high)
                        continue
                    else:
                        all_columns_for_analysis.append(col)
                else:
                    all_columns_for_analysis.append(col)
    all_columns_for_analysis.extend(['PATNO', 'EVENT_ID'])
    dataset_first_noindx = data[datasets_of_interest[0]].reset_index()
    temp_cols = list(set(dataset_first_noindx.columns).intersection(all_columns_for_analysis)) 
    dataset_first_noindx = dataset_first_noindx[temp_cols]
    patno_filtered_visited = dataset_first_noindx[ dataset_first_noindx.EVENT_ID == last_visit ]['PATNO']
    for dataset in datasets_of_interest[1:]:
        dataset_noindx = data[dataset].reset_index()
        temp_cols = list(set(dataset_noindx.columns).intersection(all_columns_for_analysis)) 
        dataset_noindx = dataset_noindx[temp_cols]
        temp_patno_df = dataset_noindx[dataset_noindx.EVENT_ID == last_visit]
        temp_patno = temp_patno_df['PATNO']
        patno_filtered_visited = patno_filtered_visited[patno_filtered_visited.isin(temp_patno)]
    new_selected_patients = patno_filtered_visited
    patno_filtered_visited = new_selected_patients
    data_visits = {}
    status_o = data["status"][data["status"].index.isin(patno_filtered_visited)].ENROLL_CAT
    screening_o = data["screening"][data["screening"].index.isin(patno_filtered_visited)]
    data_visits["info"] = pd.concat([status_o, screening_o], axis=1)
    for dataset in datasets_of_interest:
        dataset_noindx = data[dataset].reset_index()
        data_visits[dataset] = dataset_noindx[
            dataset_noindx['PATNO'].isin(patno_filtered_visited) & dataset_noindx['EVENT_ID'].isin(visits_of_interest)]
    return data_visits, patno_filtered_visited

def convert_to_timeseq(data_visits, selection_availability=None):
    data = {}
    cols_removed = ['PAG_NAME','CMEDTM','EXAMTM','PD_MED_USE','ON_OFF_DOSE','ANNUAL_TIME_BTW_DOSE_NUPDRS', 'DYSKIRAT', 'EVENT_ID', 'INFODT']
    data['updrs1'] = data_visits['updrs1'][data_visits['updrs1']['EVENT_ID'].isin(selection_availability['updrs1'])].drop('INFODT', axis=1).set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['updrs1pq'] = data_visits['updrs1pq'][data_visits['updrs1pq']['EVENT_ID'].isin(selection_availability['updrs1pq'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['updrs2pq'] = data_visits['updrs2pq'][data_visits['updrs2pq']['EVENT_ID'].isin(selection_availability['updrs2pq'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['updrs3'] = data_visits['updrs3'][data_visits['updrs3']['EVENT_ID'].isin(selection_availability['updrs3'])].drop(['PAG_NAME','CMEDTM','EXAMTM','PD_MED_USE','ON_OFF_DOSE','ANNUAL_TIME_BTW_DOSE_NUPDRS', 'DYSKIRAT'],axis=1, errors='ignore').set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['epworth'] = data_visits['epworth'][data_visits['epworth']['EVENT_ID'].isin(selection_availability['epworth'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['moca'] = data_visits['moca'][data_visits['moca']['EVENT_ID'].isin(selection_availability['moca'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['benton'] = data_visits['benton'][data_visits['benton']['EVENT_ID'].isin(selection_availability['benton'])].drop_duplicates(['PATNO', 'EVENT_ID'], keep='first').set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['neuro_cranial'] = data_visits['neuro_cranial'][data_visits['neuro_cranial']['EVENT_ID'].isin(selection_availability['neuro_cranial'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['geriatric'] = data_visits['geriatric'][data_visits['geriatric']['EVENT_ID'].isin(selection_availability['geriatric'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['hopkins_verbal'] = data_visits['hopkins_verbal'][data_visits['hopkins_verbal']['EVENT_ID'].isin(selection_availability['hopkins_verbal'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['letter_seq'] = data_visits['letter_seq'][data_visits['letter_seq']['EVENT_ID'].isin(selection_availability['letter_seq'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['quip'] = data_visits['quip'][data_visits['quip']['EVENT_ID'].isin(selection_availability['quip'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['rem'] = data_visits['rem'][data_visits['rem']['EVENT_ID'].isin(selection_availability['rem'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['aut'] = data_visits['aut'][data_visits['aut']['EVENT_ID'].isin(selection_availability['aut'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['semantic'] = data_visits['semantic'][data_visits['semantic']['EVENT_ID'].isin(selection_availability['semantic'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['stai'] = data_visits['stai'][data_visits['stai']['EVENT_ID'].isin(selection_availability['stai'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    data['sdm'] = data_visits['sdm'][data_visits['sdm']['EVENT_ID'].isin(selection_availability['sdm'])].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')
    return data

def create_matrix_normalize_form(data, datasets_of_interest, type='minmax', reverse_feature=[], labels=None, remove_outlier=False):
    feat_list = ['updrs1', 'updrs1pq', 'updrs2pq', 'updrs3', 'epworth', 'moca', 'benton', 'neuro_cranial', 'geriatric'
                 ,'hopkins_verbal', 'letter_seq', 'quip', 'rem', 'aut', 'semantic', 'stai', 'sdm']
    if type=='minmax':
        data_visits_minmax = {}
        data_visits_minmax_cleaned = {}
        data_visits_orig_outlier = {}
        data_visits_zs = {}
        for en, i in enumerate(data):
            if not feat_list[en] in datasets_of_interest:
                print(feat_list[en], i)
                continue
            stack_data = data[i].stack(level=0)
            mean_value = stack_data.mean(0)
            std_value = stack_data.std(0)
            min_cols = stack_data.min(0)
            max_cols = stack_data.max(0)
            data_visits_minmax[i] = stack_data
            data_visits_zs[i] = (stack_data - mean_value).div(std_value)
            # import pdb; pdb.set_trace()
            data_visits_minmax_cleaned[i] = (stack_data).unstack()
            data_visits_orig_outlier[i] = stack_data
            for col_re in data_visits_minmax[i].columns:
                if remove_outlier:
                    temp_data = np.array( data_visits_minmax[i].loc[:, col_re] ).flatten()
                    upper_quartile = np.percentile(temp_data, 98)
                    lower_quartile = np.percentile(temp_data, 2)
                    IQR = (upper_quartile - lower_quartile) * 0
                    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
                    val_low = quartileSet[0]
                    val_high = quartileSet[1]
                    if val_high == val_low:
                        del data_visits_minmax[i][col_re]
                        continue
                    data_visits_minmax[i].loc[:, col_re] = data_visits_minmax[i].loc[:, col_re].apply(lambda x: val_high if x >= val_high else x)
                    data_visits_minmax[i].loc[:, col_re] = data_visits_minmax[i].loc[:, col_re].apply(lambda x: val_low if x <= val_low else x)
            data_visits_orig_outlier[i] = data_visits_minmax[i].unstack().copy()
            min_cols = data_visits_minmax[i].min(0)
            max_cols = data_visits_minmax[i].max(0)
            data_visits_minmax[i] = (data_visits_minmax[i] - min_cols).div(max_cols - min_cols)
            for col_re in data_visits_minmax[i].columns:
                if col_re in list(set(reverse_feature + ['JLO_TOTRAW', 'naming'])):
                    data_visits_minmax[i].loc[:, col_re] = 1 - data_visits_minmax[i].loc[:, col_re]
            data_visits_minmax[i] = data_visits_minmax[i].unstack()
            data_visits_zs[i] = data_visits_zs[i].unstack()
        M_zs = pd.concat([data_visits_zs[i].astype('float') for i in datasets_of_interest], axis=1, join='inner')
        M_orig = pd.concat([data_visits_minmax_cleaned[i].astype('float') for i in datasets_of_interest], axis=1, join='inner')
        M_orig_outlier = pd.concat([data_visits_orig_outlier[i].astype('float') for i in datasets_of_interest], axis=1, join='inner')
        M_minmax = pd.concat([data_visits_minmax[i].astype('float') for i in datasets_of_interest], axis=1, join='inner')
        return M_minmax, M_orig, M_orig_outlier, M_zs