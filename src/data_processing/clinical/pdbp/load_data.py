import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

# loading libraries and settings
import pandas as pd
import numpy as np


def read_from_raw_files_old(m, merge_screen):
    ## loading data and selecting the necessary columns
    cols = {}  # column names
    cols["epworth"] = ["EpworthSleepinessScale.Required Fields.GUID",
                       "EpworthSleepinessScale.Required Fields.VisitTypPDBP",
                       "EpworthSleepinessScale.ESS.ESS_SittingReading", "EpworthSleepinessScale.ESS.ESS_WatchingTV",
                       "EpworthSleepinessScale.ESS.ESS_SittingInactive",
                       "EpworthSleepinessScale.ESS.ESS_PassengerInCar",
                       "EpworthSleepinessScale.ESS.ESS_LyingDownToRest",
                       "EpworthSleepinessScale.ESS.ESS_SittingTalking",
                       "EpworthSleepinessScale.ESS.ESS_SittingLunchNoAlc",
                       "EpworthSleepinessScale.ESS.ESS_DozingInTraffc", "EpworthSleepinessScale.ESS.ESS_TotalScore"]
    cols["HDRS"] = ["HDRS.Required Fields.VisitTypPDBP", "HDRS.Required Fields.GUID", "HDRS.HDRS.HDRSDeprsdMdInd",
                    "HDRS.HDRS.HDRSGltMndInd", "HDRS.HDRS.HDRSucdInd", "HDRS.HDRS.HDRSErlyNgtInsmnInd",
                    "HDRS.HDRS.HDRSMddlNgtInsmnInd", "HDRS.HDRS.HDRSErlyMornInsmnInd", "HDRS.HDRS.HDRSWrkActDifcltInd",
                    "HDRS.HDRS.HDRSRetrdtnInd", "HDRS.HDRS.HDRSAgttnInd", "HDRS.HDRS.HDRSAnxPsycDifcltInd",
                    "HDRS.HDRS.HDRSAnxSomtcInd", "HDRS.HDRS.HDRSSomtcSymptmInd", "HDRS.HDRS.HDRSGenrlSomtcSymptmInd",
                    "HDRS.HDRS.HDRSGentlSymptmInd", "HDRS.HDRS.HDRSHypchdssInd", "HDRS.HDRS.HDRSWgtLosPatInd",
                    "HDRS.HDRS.HDRSWgtLosMeasrInd", "HDRS.HDRS.HDRSInsgtInd", "HDRS.HDRS.HDRSTotScore"]
    cols["updrs"] = ["MDS_UPDRS.Required Fields.VisitTypPDBP", "MDS_UPDRS.Required Fields.GUID",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSRcntCogImprmntScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSHallucPsychosScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSDrpssMoodScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSAnxsMoodScore", "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSApathyScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSDopmnDysregSyndScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSQstnnreInfoProvdrTyp",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSSleepProbScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSDaytmSleepScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSPainOthrSensScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSUrnryProbScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSConstipProbScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSLiteHeadStndngScore",
                     "MDS_UPDRS.Part I: nM-EDL.MDSUPDRSFatigueScore", "MDS_UPDRS.Part I: nM-EDL.MDSUPDRS_PartIScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSSpeechScore", "MDS_UPDRS.Part II: M-EDL.MDSUPDRSSlivaDroolScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSChwngSwllwngScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSEatingTskScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSDressingScore", "MDS_UPDRS.Part II: M-EDL.MDSUPDRSHygieneScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSHandwritingScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSHobbieOthrActScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSTurngBedScore", "MDS_UPDRS.Part II: M-EDL.MDSUPDRSTremorScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSGttngOutBedScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSWlkngBalanceScore",
                     "MDS_UPDRS.Part II: M-EDL.MDSUPDRSFreezingScore", "MDS_UPDRS.Part II: M-EDL.MDSUPDRS_PartIIScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSPtntPrknsnMedInd",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSPtClinStatePrknsnMdInd",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSPtntUseLDOPAInd",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSLstLDOPADoseTm",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSFreeFlowSpeechScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSFacialExprScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSNeckRigidScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRUERigidScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSLUERigidScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRLERigidScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSLLERigidScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSFingerTppngRteHndScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSFingerTppngLftHndScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRteHndScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSLftHndScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSProntSupnRtHndMvmtScr",
                     "MDS_UPDRS.Part III: Motor Examination.PronatSupinLftHndMvmntScore",
                     "MDS_UPDRS.Part III: Motor Examination.RteFtToeTppngScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSLftFtToeTppngScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSLegAgiltyRteLegScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSLegAgiltyLftLegScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSArisingFrmChrScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSGaitScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSFreezingGaitScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSPostrlStabltyScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSPostureScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSGlblSpontntyMvmntScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSPostrlTremorRtHndScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSPostrlTremrLftHndScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSKineticTremrRtHndScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSKineticTremrLftHndScr",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRestTremorAmpRUEScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRestTremorAmpLUEScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRestTremorAmpRLEScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRestTremorAmpLLEScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSRestTremrAmpLipJawScr",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSConstncyRestTremrScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSDyskChreaDystnaPrsScr",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSMvmntIntrfrnceScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRSHoehnYahrStageScore",
                     "MDS_UPDRS.Part III: Motor Examination.MDSUPDRS_PartIIIScore",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTmSpntDyskScore",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTtlHrAwkDyskNum",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTtlHrDyskNum",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSPrcntDyskVal",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSFuncImpactDyskScore",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTtlHrAwkOffStateNum",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTtlHrOffNum",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSPrcntOffVal",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTmSpntOffStateScore",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSFuncImpactFluctScore",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSComplxtyMtrFluctScore",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSPainflOffStatDystnaScr",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTtlHrOffDemnDystniaNum",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSTtlHrOffWDystniaNum",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRSPrcntOffDystniaVal",
                     "MDS_UPDRS.Part IV: Motor Complications.MDSUPDRS_PartIVScore",
                     "MDS_UPDRS.Total Score.MDSUPDRS_TotalScore"]
    if not 'del' in merge_screen:
        cols["moca"] = ["MoCA.Required Fields.VisitTypPDBP", "MoCA.Required Fields.GUID",
                        "MoCA.MoCA.MOCA_VisuospatialExec", "MoCA.MoCA.MOCA_Naming", "MoCA.MoCA.MOCA_Digits",
                        "MoCA.MoCA.MOCA_Letters", "MoCA.MoCA.MOCA_Serial7", "MoCA.MoCA.MOCA_LangRepeat",
                        "MoCA.MoCA.MOCA_LangFluency", "MoCA.MoCA.MOCA_Abstraction", "MoCA.MoCA.MOCA_DelydRecall",
                        "MoCA.MoCA.MOCA_DelydRecallOptnlCatCue", "MoCA.MoCA.MOCA_DelydRecalOptnlMultChoice",
                        "MoCA.MoCA.MOCA_Orient", "MoCA.MoCA.MOCA_ImageResponse", "MoCA.MoCA.MOCA_EduInd",
                        "MoCA.MoCA.MOCA_Total"]
    else:
        cols["moca"] = ["MoCA.Required Fields.VisitTypPDBP", "MoCA.Required Fields.GUID",
                        "MoCA.MoCA.MOCA_VisuospatialExec", "MoCA.MoCA.MOCA_Digits", "MoCA.MoCA.MOCA_Letters",
                        "MoCA.MoCA.MOCA_Serial7", "MoCA.MoCA.MOCA_LangRepeat", "MoCA.MoCA.MOCA_LangFluency",
                        "MoCA.MoCA.MOCA_Abstraction", "MoCA.MoCA.MOCA_DelydRecall",
                        "MoCA.MoCA.MOCA_DelydRecallOptnlCatCue", "MoCA.MoCA.MOCA_DelydRecalOptnlMultChoice",
                        "MoCA.MoCA.MOCA_Orient", "MoCA.MoCA.MOCA_ImageResponse", "MoCA.MoCA.MOCA_EduInd",
                        "MoCA.MoCA.MOCA_Total"]
    cols["schwab"] = ["ModSchwabAndEnglandScale.Required Fields.VisitTypPDBP",
                      "ModSchwabAndEnglandScale.Required Fields.GUID",
                      "ModSchwabAndEnglandScale.Scale Score.UPDRStScaleSchEngDalLivScl"]
    # cols["pdq39"] = ["PDQ39.Required Fields.VisitTypPDBP", "PDQ39.Required Fields.GUID", "PDQ39.PDQ-39.PDQ_39_Leisure", "PDQ39.PDQ-39.PDQ_39_Housework", "PDQ39.PDQ-39.PDQ_39_GroceryBags", "PDQ39.PDQ-39.PDQ_39_WalkingHalfMile", "PDQ39.PDQ-39.PDQ_39_WalkingBlock", "PDQ39.PDQ-39.PDQ_39_House", "PDQ39.PDQ-39.PDQ_39_PublicPlaces", "PDQ39.PDQ-39.PDQ_39_Outside", "PDQ39.PDQ-39.PDQ_39_Falling", "PDQ39.PDQ-39.PDQ_39_Confined", "PDQ39.PDQ-39.PDQ_39_Showering", "PDQ39.PDQ-39.PDQ_39_Dressing", "PDQ39.PDQ-39.PDQ_39_Buttons", "PDQ39.PDQ-39.PDQ_39_Writing", "PDQ39.PDQ-39.PDQ_39_CuttingFood", "PDQ39.PDQ-39.PDQ_39_HoldingDrink", "PDQ39.PDQ-39.PDQ_39_Depressed", "PDQ39.PDQ-39.PDQ_39_Lonely", "PDQ39.PDQ-39.PDQ_39_Tearful", "PDQ39.PDQ-39.PDQ_39_Angry", "PDQ39.PDQ-39.PDQ_39_Anxious", "PDQ39.PDQ-39.PDQ_39_Worried", "PDQ39.PDQ-39.PDQ_39_HidePD", "PDQ39.PDQ-39.PDQ_39_EatingPublic", "PDQ39.PDQ-39.PDQ_39_EmbarrassedPublic", "PDQ39.PDQ-39.PDQ_39_WorriedReaction", "PDQ39.PDQ-39.PDQ_39_Relationships", "PDQ39.PDQ-39.PDQ_39_LackOfSuprtPrtnr", "PDQ39.PDQ-39.PDQ_39_LackOfSuprtFmly", "PDQ39.PDQ-39.PDQ_39_UnexpctdSleep", "PDQ39.PDQ-39.PDQ_39_Concentration", "PDQ39.PDQ-39.PDQ_39_Memory", "PDQ39.PDQ-39.PDQ_39_Dreams", "PDQ39.PDQ-39.PDQ_39_Speaking", "PDQ39.PDQ-39.PDQ_39_Communicate", "PDQ39.PDQ-39.PDQ_39_Ignored", "PDQ39.PDQ-39.PDQ_39_CrampsSpasms", "PDQ39.PDQ-39.PDQ_39_AchesPains", "PDQ39.PDQ-39.PDQ_39_HotCold", "PDQ39.Scale Scores.PDQ_39_TotalScore_Mobility", "PDQ39.Scale Scores.PDQ_39_TotalScore_ADL", "PDQ39.Scale Scores.PDQ_39_TotalScore_Emotional", "PDQ39.Scale Scores.PDQ_39_TotalScore_Stigma", "PDQ39.Scale Scores.PDQ_39_TotalScore_SocialSuprt", "PDQ39.Scale Scores.PDQ_39_TotalScore_CogImpairmnt", "PDQ39.Scale Scores.PDQ_39_TotalScore_Communcation", "PDQ39.Scale Scores.PDQ_39_TotalScore_BodDiscomfrt"]
    cols["pdq39"] = ["PDQ39.Required Fields.VisitTypPDBP", "PDQ39.Required Fields.GUID",
                     "PDQ39.Scale Scores.PDQ_39_TotalScore_Mobility", "PDQ39.Scale Scores.PDQ_39_TotalScore_ADL",
                     "PDQ39.Scale Scores.PDQ_39_TotalScore_Emotional", "PDQ39.Scale Scores.PDQ_39_TotalScore_Stigma",
                     "PDQ39.Scale Scores.PDQ_39_TotalScore_SocialSuprt",
                     "PDQ39.Scale Scores.PDQ_39_TotalScore_CogImpairmnt",
                     "PDQ39.Scale Scores.PDQ_39_TotalScore_Communcation",
                     "PDQ39.Scale Scores.PDQ_39_TotalScore_BodDiscomfrt"]
    cols["upsit"] = ["UnivOfPennSmellIdenTest.Required Fields.VisitTypPDBP",
                     "UnivOfPennSmellIdenTest.Required Fields.GUID", "UnivOfPennSmellIdenTest.UPSIT.UPennSITTotal"]
    cols["demographics"] = ["Demographics.Required Fields.GUID", "Demographics.Required Fields.AgeYrs","Demographics.Required Fields.VisitDate",
                            "Demographics.Demographics.GenderTypPDBP"]
    cols["biosample"] = ["BiosampleCatalogV5.Subject Information.GUID",
                         "BiosampleCatalogV5.Subject Information.InclusnXclusnCntrlInd"]
    cols["family_history_col"] = ["FamilyHistory.Required Fields.GUID",
                                  "FamilyHistory.Parkinson's disease.FamHistMedclCondInd"]

    ## data load
    epworth = pd.read_csv(m.query_result_EpworthSleepinessScale, usecols=cols["epworth"])
    epworth = epworth.rename(columns={col: col.split('.')[-1] for col in epworth.columns})
    epworth = epworth.rename(columns={'VisitTypPDBP': 'EVENT_ID', 'GUID': 'PATNO'})

    HDRS = pd.read_csv(m.query_result_HDRS, usecols=cols["HDRS"])
    HDRS = HDRS.rename(columns={col: col.split('.')[-1] for col in HDRS.columns})
    HDRS = HDRS.rename(columns={'VisitTypPDBP': 'EVENT_ID', 'GUID': 'PATNO'})

    updrs = pd.read_csv(m.query_result_MDS_UPDRS, usecols=cols["updrs"])
    updrs = updrs.rename(columns={col: col.split('.')[-1] for col in updrs.columns})
    updrs = updrs.rename(columns={'VisitTypPDBP': 'EVENT_ID', 'GUID': 'PATNO'})

    moca = pd.read_csv(m.query_result_MoCA, usecols=cols["moca"])
    moca = moca.rename(columns={col: col.split('.')[-1] for col in moca.columns})
    moca = moca.rename(columns={'VisitTypPDBP': 'EVENT_ID', 'GUID': 'PATNO'})

    schwab = pd.read_csv(m.query_result_ModSchwabAndEnglandScale, usecols=cols["schwab"])
    schwab = schwab.rename(columns={col: col.split('.')[-1] for col in schwab.columns})
    schwab = schwab.rename(columns={'VisitTypPDBP': 'EVENT_ID', 'GUID': 'PATNO'})

    pdq39 = pd.read_csv(m.query_result_PDQ39, usecols=cols["pdq39"])
    pdq39 = pdq39.rename(columns={col: col.split('.')[-1] for col in pdq39.columns})
    pdq39 = pdq39.rename(columns={'VisitTypPDBP': 'EVENT_ID', 'GUID': 'PATNO'})

    upsit = pd.read_csv(m.query_result_UnivOfPennSmellIdenTest, usecols=cols["upsit"])
    upsit = upsit.rename(columns={col: col.split('.')[-1] for col in upsit.columns})
    upsit = upsit.rename(columns={'VisitTypPDBP': 'EVENT_ID', 'GUID': 'PATNO'})

    # Patient info
    demographics = pd.read_csv(m.query_result_Demographics, usecols=cols["demographics"])
    demographics = demographics.rename(columns={col: col.split('.')[-1] for col in demographics.columns})
    demographics = demographics.rename(columns={'GUID': 'PATNO', 'AgeYrs': 'Age', 'GenderTypPDBP': 'Gender'})
    demographics = demographics.drop_duplicates(subset='PATNO', keep='last')

    pdbp_info = demographics.copy()
    biosample = pd.read_csv(m.query_result_BiosampleCatalogV5, usecols=cols["biosample"])
    biosample = biosample.rename(columns={col: col.split('.')[-1] for col in biosample.columns})
    biosample = biosample.rename(columns={'GUID': 'PATNO', 'InclusnXclusnCntrlInd': 'ENROLL_CAT'})
    biosample = biosample.drop_duplicates(subset='PATNO', keep='last')

    pdbp_info = pd.concat([pdbp_info.set_index('PATNO'), biosample.set_index('PATNO')], axis=1, join='inner')

    family_history_pd = pd.read_csv(m.query_result_FamilyHistory, usecols=cols["family_history_col"])
    family_history_pd = family_history_pd.rename(columns={col: col.split('.')[-1] for col in family_history_pd.columns})
    family_history_pd = family_history_pd.rename(columns={'GUID': 'PATNO'})
    family_history_pd = family_history_pd.dropna(axis=0).drop_duplicates(subset='PATNO', keep='last')

    data = {}
    data['epworth'] = epworth
    data['HDRS'] = HDRS
    data['updrs'] = updrs
    data['moca'] = moca
    data['schwab'] = schwab
    data['pdq39'] = pdq39
    data['upsit'] = upsit
    data['demographics'] = demographics
    data['pdbp_info'] = pdbp_info
    data['biosample'] = biosample
    data['family_history_pd'] = family_history_pd
    return data

def test_measure(l_hc, l_pd, fil=0):
    n_hc = len(l_hc)
    n_pd = len(l_pd)
    s_hc = np.std(l_hc)
    s_pd = np.std(l_pd)
    m_hc = np.mean(l_hc)
    m_pd = np.mean(l_pd)
    m_diff = m_hc - m_pd
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
    if fil == 0:
        if p_val < 0.05 or (p_val > 0.15 and p_val < 0.25):
            return ('PASS', m_hc, m_pd, p_val, '[{},{}]'.format(L_L, L_U))
        else:
            return ('FAIL', m_hc, m_pd, p_val, '[{},{}]'.format(L_L, L_U))
    elif fil==1:
        if p_val < 0.25: # or (p_val > 0.15 and p_val < 0.25) or (p_val < 0.25):
            return ('PASS', m_hc, m_pd, p_val, '[{},{}]'.format(L_L, L_U))
        else:
            return ('FAIL', m_hc, m_pd, p_val, '[{},{}]'.format(L_L, L_U))

import copy
import copy
def flipping_data(data, ECAT, files_of_interest, getP=False, visit_id='V12', fil=0):
    copy_data = copy.deepcopy(data)
    ignore = ['index', 'MDSUPDRSLstLDOPADoseTm', 'HDRSWgtLosMeasrInd', 'MDSUPDRSPtntUseLDOPAInd', 'MDSUPDRSPtClinStatePrknsnMdInd',
              'MDSUPDRSPtntPrknsnMedInd', 'MDSUPDRSDyskChreaDystnaPrsScr', 'MDSUPDRSTtlHrAwkDyskNum', 'MDSUPDRSMvmntIntrfrnceScore',
              'MDSUPDRSPrcntOffVal', 'MDSUPDRSTtlHrOffNum', 'MDSUPDRSTtlHrAwkOffStateNum', 'MDSUPDRSPrcntDyskVal',
              'MDSUPDRSTtlHrDyskNum', 'MDSUPDRSPrcntOffDystniaVal', 'MDSUPDRSTtlHrOffWDystniaNum','MDSUPDRSTtlHrOffDemnDystniaNum',
              'MDSUPDRSQstnnreInfoProvdrTyp','MOCA_DelydRecalOptnlMultChoice', 'MOCA_DelydRecallOptnlCatCue', 'MOCA_ImageResponse']

    test_df = {i: {} for i in files_of_interest}
    for feat in files_of_interest:
        KP = data[feat].set_index(['PATNO'])
        DF = pd.merge(KP, ECAT, left_index=True, right_index=True)
        DF2 = DF[DF['EVENT_ID']==visit_id]
        DF2 = DF2.reset_index().set_index(['PATNO', 'EVENT_ID'])
        for col in list(DF2.columns):
            if col in ignore + ['ENROLL_CAT']:
                continue
            l_hc = list(DF2[DF2['ENROLL_CAT'] == 'Control'][col].dropna())
            l_pd = list(DF2[DF2['ENROLL_CAT'] == 'Case'][col].dropna())
            try:
                decision, m_hc, m_pd, p_val, ci = test_measure(l_hc, l_pd, fil)
            except:
                import pdb; pdb.set_trace()
            if decision == 'PASS':# or feat in ['hopkins_verbal', 'moca']:
                # if not getP:
                #     copy_data[feat][col] = list(-data[feat][col])
                test_df[feat][col] = str(np.round(p_val, 3)) + '**' + ' ' + ci
            else:
                test_df[feat][col] = str(np.round(p_val, 3)) + ' ' + ci
    from collections import defaultdict
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
    P = pd.DataFrame(test_df_df)
    P = P.sort_values(by=['p value', 'feature']).set_index(pd.Index(range(len(P))))
    if fil == 0:
        P['reverse'] = P['p value'].map(lambda x: '**anant**' in x)
    else:
        P['reverse'] = P['p value'].map(lambda x: '**' in x)
    
    return copy_data, P

def load_required_files(data, datasets_of_interest, visits_of_interest):

    # visits_of_interest = ['BL', 'V02', 'V03', 'V04', 'V06']#, 'V08']#, 'V08']#, 'V09', 'V10', 'V11', 'V12'] #'V01', 'V05', 'V07'
    last_visit = visits_of_interest[-1]
    # selecting participants with data from BL to last_visit
    dataset_first_noindx = data[datasets_of_interest[0]].reset_index()
    patno_filtered_visited = dataset_first_noindx[ dataset_first_noindx.EVENT_ID == last_visit ]['PATNO']
    for dataset in datasets_of_interest[1:]:
        dataset_noindx = data[dataset].reset_index()
        temp_patno = dataset_noindx[ dataset_noindx.EVENT_ID == last_visit ]['PATNO']
        patno_filtered_visited = patno_filtered_visited[ patno_filtered_visited.isin(temp_patno) ]

    data_visits = {}
    data_visits["info"] = data['pdbp_info'][data['pdbp_info'].index.isin(patno_filtered_visited)]

    for dataset in datasets_of_interest:
        dataset_noindx = data[dataset].reset_index()
        data_visits[dataset] = dataset_noindx[ dataset_noindx['PATNO'].isin(patno_filtered_visited) & dataset_noindx['EVENT_ID'].isin(visits_of_interest) ]

    return data_visits, patno_filtered_visited

def convert_to_timeseq(data_visits, selection_availability=None):
    data = {}
    for i,j in data_visits.items():
        if i in selection_availability:
            # import pdb;pdb.set_trace()
            data_visits[i] = j[j['EVENT_ID'].isin(selection_availability[i])]
    t1 = data_visits["epworth"].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t1 = t1.loc[~t1.index.duplicated(keep='last')]
    data["epworth"] = t1.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1)
    t2 = data_visits['HDRS'].drop(['index', 'HDRSWgtLosMeasrInd'],axis=1).set_index(['PATNO','EVENT_ID'])
    t2 = t2.loc[~t2.index.duplicated(keep='last')]
    data["HDRS"] = t2.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1)
    t3 = data_visits['updrs'].drop(['index', 'MDSUPDRSLstLDOPADoseTm', 'MDSUPDRSPtntUseLDOPAInd', 'MDSUPDRSPtClinStatePrknsnMdInd', 'MDSUPDRSPtntPrknsnMedInd', 'MDSUPDRSDyskChreaDystnaPrsScr', 'MDSUPDRSTtlHrAwkDyskNum', 'MDSUPDRSMvmntIntrfrnceScore', 'MDSUPDRSPrcntOffVal', 'MDSUPDRSTtlHrOffNum', 'MDSUPDRSTtlHrAwkOffStateNum', 'MDSUPDRSPrcntDyskVal', 'MDSUPDRSTtlHrDyskNum', 'MDSUPDRSPrcntOffDystniaVal','MDSUPDRSTtlHrOffWDystniaNum','MDSUPDRSTtlHrOffDemnDystniaNum', 'MDSUPDRSQstnnreInfoProvdrTyp'],axis=1).set_index(['PATNO','EVENT_ID'])
    t3 = t3.loc[~t3.index.duplicated(keep='last')]
    data["updrs"] = t3.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    t4 = data_visits['moca'].drop(['index', 'MOCA_DelydRecalOptnlMultChoice', 'MOCA_DelydRecallOptnlCatCue', 'MOCA_ImageResponse'], axis=1).set_index(['PATNO','EVENT_ID'])
    t4 = t4.loc[~t4.index.duplicated(keep='last')]
    data["moca"] = t4.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1)
    t5 = data_visits['schwab'].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t5 = t5.loc[~t5.index.duplicated(keep='last')]
    data["schwab"] = t5.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1)
    t6 = data_visits['pdq39'].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t6 = t6.loc[~t6.index.duplicated(keep='last')]
    data["pdq39"] = t6.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1)
    t7 = data_visits['upsit'].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t7 = t7.loc[~t7.index.duplicated(keep='last')]
    data["upsit"] = t7.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1)
    return data





def convert_to_timeseq_old(data_visits):
    data = {}
    t1 = data_visits["epworth"].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t1 = t1.loc[~t1.index.duplicated(keep='last')]
    data["epworth"] = t1.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t2 = data_visits['HDRS'].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t2 = t2.loc[~t2.index.duplicated(keep='last')]
    data["HDRS"] = t2.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t3 = data_visits['updrs'].drop(['index', 'MDSUPDRSLstLDOPADoseTm', 'MDSUPDRSPtntUseLDOPAInd', 'MDSUPDRSPtClinStatePrknsnMdInd', 'MDSUPDRSPtntPrknsnMedInd', 'MDSUPDRSDyskChreaDystnaPrsScr', 'MDSUPDRSTtlHrAwkDyskNum', 'MDSUPDRSMvmntIntrfrnceScore', 'MDSUPDRSPrcntOffVal', 'MDSUPDRSTtlHrOffNum', 'MDSUPDRSTtlHrAwkOffStateNum', 'MDSUPDRSPrcntDyskVal', 'MDSUPDRSTtlHrDyskNum', 'MDSUPDRSPrcntOffDystniaVal','MDSUPDRSTtlHrOffWDystniaNum','MDSUPDRSTtlHrOffDemnDystniaNum', 'MDSUPDRSQstnnreInfoProvdrTyp'],axis=1).set_index(['PATNO','EVENT_ID'])
    t3 = t3.loc[~t3.index.duplicated(keep='last')]
    data["updrs"] = t3.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    t4 = data_visits['moca'].drop(['MOCA_DelydRecalOptnlMultChoice', 'MOCA_DelydRecallOptnlCatCue', 'MOCA_ImageResponse'], axis=1).set_index(['PATNO','EVENT_ID'])
    t4 = t4.loc[~t4.index.duplicated(keep='last')]
    data["moca"] = t4.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t5 = data_visits['schwab'].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t5 = t5.loc[~t5.index.duplicated(keep='last')]
    data["schwab"] = t5.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t6 = data_visits['pdq39'].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t6 = t6.loc[~t6.index.duplicated(keep='last')]
    data["pdq39"] = t6.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')#.interpolate(method='linear', axis=1)
    t7 = data_visits['upsit'].drop(['index'],axis=1).set_index(['PATNO','EVENT_ID'])
    t7 = t7.loc[~t7.index.duplicated(keep='last')]
    data["upsit"] = t7.sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    return data


def create_matrix_normalize_form(data, datasets_of_interest, type='minmax', reverse_feature=[], labels=None, remove_outlier=False):
    feat_list = ['updrs1', 'updrs1pq', 'updrs2pq', 'updrs3', 'epworth', 'moca', 'benton', 'neuro_cranial', 'geriatric'
                 ,'hopkins_verbal', 'letter_seq', 'quip', 'rem', 'aut', 'semantic', 'stai', 'sdm']
    feat_list = ['epworth', 'HDRS', 'updrs', 'moca', 'schwab', 'pdq39', 'upsit']
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


def create_matrix_nddffffffffffformalize_form(data, datasets_of_interest, type='minmax', remove_outlier=False):
    # .stack(level=0).interpolate(method='linear', axis=1, limit_direction='both').unstack()
    if type=='minmax':
        data_visits_minmax = {}
        data_visits_minmax_cleaned = {}
        for en, i in enumerate(data):
            # if not feat_list[en] in datasets_of_interest:
            #    print (feat_list[en], i)
            #    continue
            stack_data = data[i].stack(level=0)
            min_cols = stack_data.min(0)
            max_cols = stack_data.max(0)
            data_visits_minmax[i] = (stack_data - min_cols).div(max_cols - min_cols)
            data_visits_minmax_cleaned[i] = (stack_data).unstack()
            # for col_re in data_visits_minmax[i].columns:
            #     if col_re in reverse_feature:
                    # print(col_re)
                    # import pdb; pdb.set_trace()
            #         data_visits_minmax[i] .loc[:, col_re] = -data_visits_minmax[i] .loc[:, col_re] + 1
                    # import pdb; pdb.set_trace()
            data_visits_minmax[i] = data_visits_minmax[i].unstack()
        M_orig = pd.concat([data_visits_minmax_cleaned[i].astype('float') for i in datasets_of_interest], axis=1, join='inner')
        M_minmax = pd.concat([data_visits_minmax[i].astype('float') for i in datasets_of_interest], axis=1, join='inner')
        return M_minmax, M_orig

    # if type=='minmax':
    #     data_visits_minmax = {}
    #     minmax_min = {}
    #     minmax_max = {}
    #     for i in data:
    #         min_cols = data[i].stack(level=0).min(0)
    #         max_cols = data[i].stack(level=0).max(0)
    #         data_visits_minmax[i] = (data[i].stack(level=0) - min_cols).div(max_cols - min_cols).unstack()
    #     M_minmax = pd.concat([data_visits_minmax[i].astype('float') for i in datasets_of_interest], axis=1, join='inner')
    #     return M_minmax
    data_visits_zs = {}
    for i in range(len(datasets_of_interest)):
        dataset = datasets_of_interest[i]
        dataset_columns = data[dataset].columns.levels[0][0:-1]
        data_visits_zs[dataset] = pd.DataFrame(index=data[dataset].index, columns=data[dataset].columns)
        for col in dataset_columns:
            data_visits_zs[dataset][col] = (data[dataset][col] - data[dataset][col].mean().mean()) / data[dataset][
                col].stack().std()
    M_zs = pd.concat([data_visits_zs[i].stack(level=0).interpolate(method='linear', axis=1, limit_direction='both').unstack() for i in datasets_of_interest], axis=1, join='inner')
    return M_zs

def create_matrix_form_old(data_visits):
    data = {}
    t1 = data_visits['updrs1'].drop('INFODT', axis=1).set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t2 = data_visits['updrs1pq'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t3 = data_visits['updrs2pq'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t4 = data_visits['updrs3'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index(
        'PATNO').interpolate(method='linear', axis=1)
    t5 = data_visits['epworth'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1)
    t6 = data_visits['moca'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index(
        'PATNO').interpolate(method='linear', axis=1)
    t7 = data_visits['benton'].drop_duplicates(['PATNO', 'EVENT_ID'], keep='first').set_index(
        ['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(
        method='linear', axis=1)
    t8 = data_visits['neuro_cranial'][data_visits['neuro_cranial'].EVENT_ID != 'V03'].set_index(
        ['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(
        method='linear', axis=1)
    t9 = data_visits['geriatric'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10,
                                                                              limit_direction='both')
    t10 = data_visits['hopkins_verbal'][data_visits['hopkins_verbal'].EVENT_ID != 'V03'].set_index(
        ['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(
        method='linear', axis=1, limit=10, limit_direction='both')
    t11 = data_visits['letter_seq'][data_visits['letter_seq'].EVENT_ID != 'V03'].set_index(
        ['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(
        method='linear', axis=1, limit=10, limit_direction='both')
    t12 = data_visits['quip'][data_visits['quip'].EVENT_ID != 'V03'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10,
                                                                              limit_direction='both')
    t13 = data_visits['rem'][data_visits['rem'].EVENT_ID != 'V03'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10,
                                                                              limit_direction='both')
    t14 = data_visits['aut'][data_visits['aut'].EVENT_ID != 'V03'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10,
                                                                              limit_direction='both')
    t15 = data_visits['semantic'][data_visits['semantic'].EVENT_ID != 'V03'].set_index(
        ['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10,
                                                                              limit_direction='both')
    t16 = data_visits['stai'][data_visits['stai'].EVENT_ID != 'V03'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10,
                                                                                            limit_direction='both')
    t17 = data_visits['sdm'][data_visits['sdm'].EVENT_ID != 'V03'].set_index(['PATNO', 'EVENT_ID']).sort_index(
        level='PATNO').unstack().reset_index().set_index('PATNO').interpolate(method='linear', axis=1, limit=10,limit_direction='both')

    # B = None
    # if len(biospecimen_dataset):
    #     b1 = data_visits['total_tau'].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')# .interpolate(method='linear', axis=1, limit=10,limit_direction='both')
    #     b2 = data_visits['alpha_syn'].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')# .interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    #     b3 = data_visits['abeta_42'].set_index(['PATNO', 'EVENT_ID']).sort_index(level='PATNO').unstack().reset_index().set_index('PATNO')# .interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    #     b4 = data_visits['p_tau181p'].set_index(['PATNO', 'EVENT_ID']).sort_index( level='PATNO').unstack().reset_index().set_index('PATNO')# .interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    #     B = pd.concat([b1, b2, b3, b4], axis=1)# .interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    #     for i in range(1, 5):
    #         data['b' + str(i)] = eval('b' + str(i))
    M = pd.concat([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17], axis=1).interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    for i in range(1, 18):
        data['t'+str(i)] = eval('t'+str(i))

    return data, M

def create_matrix_normalize_form_old(data, datasets_of_interest):
    data_visits_zs = {}
    for i in range(1, len(datasets_of_interest) + 1):
        dataset = 't' + str(i)
        dataset_columns = data[dataset].columns.levels[0][0:-1]
        data_visits_zs[dataset] = pd.DataFrame(index=data[dataset].index, columns=data[dataset].columns)
        for col in dataset_columns:
            data_visits_zs[dataset][col] = (data[dataset][col] - data[dataset][col].mean().mean()) / data[dataset][
                col].stack().std()
    M_zs = pd.concat([data_visits_zs['t1'], data_visits_zs['t2'], data_visits_zs['t3'], data_visits_zs['t4'],
                  data_visits_zs['t5'], data_visits_zs['t6'], data_visits_zs['t7'], data_visits_zs['t8'],
                  data_visits_zs['t9'], data_visits_zs['t10'], data_visits_zs['t11'], data_visits_zs['t12'],
                  data_visits_zs['t13'], data_visits_zs['t14'], data_visits_zs['t15'], data_visits_zs['t16'],
                  data_visits_zs['t17']], axis=1).interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    data_visits_minmax = {}
    minmax_min = {}
    minmax_max = {}
    for i in range(1, len(datasets_of_interest) + 1):
        dataset = 't' + str(i)
        dataset_columns = data[dataset].columns.levels[0][0:-1]
        data_visits_minmax[dataset] = pd.DataFrame(index=data[dataset].index, columns=data[dataset].columns)
        minmax_min[dataset] = pd.DataFrame(index=[1], columns=data[dataset].columns)
        minmax_max[dataset] = pd.DataFrame(index=[1], columns=data[dataset].columns)
        for col in dataset_columns:
            data_visits_minmax[dataset][col] = (data[dataset][col] - data[dataset][col].min().min()) / (
                        data[dataset][col].max().max() - data[dataset][col].min().min())
            minmax_min[dataset][col] = data[dataset][col].min().min()
            minmax_max[dataset][col] = data[dataset][col].max().max()
    M_minmax = pd.concat(
        [data_visits_minmax['t1'], data_visits_minmax['t2'], data_visits_minmax['t3'], data_visits_minmax['t4'],
         data_visits_minmax['t5'], data_visits_minmax['t6'], data_visits_minmax['t7'], data_visits_minmax['t8'],
         data_visits_minmax['t9'], data_visits_minmax['t10'], data_visits_minmax['t11'], data_visits_minmax['t12'],
         data_visits_minmax['t13'], data_visits_minmax['t14'], data_visits_minmax['t15'], data_visits_minmax['t16'],
         data_visits_minmax['t17']], axis=1).interpolate(method='linear', axis=1, limit=10, limit_direction='both')
    M_minmax_min = pd.concat([minmax_min['t1'], minmax_min['t2'], minmax_min['t3'], minmax_min['t4'],
                              minmax_min['t5'], minmax_min['t6'], minmax_min['t7'], minmax_min['t8'],
                              minmax_min['t9'], minmax_min['t10'], minmax_min['t11'], minmax_min['t12'],
                              minmax_min['t13'], minmax_min['t14'], minmax_min['t15'], minmax_min['t16'],
                              minmax_min['t17']], axis=1)
    M_minmax_max = pd.concat([minmax_max['t1'], minmax_max['t2'], minmax_max['t3'], minmax_max['t4'],
                              minmax_max['t5'], minmax_max['t6'], minmax_max['t7'], minmax_max['t8'],
                              minmax_max['t9'], minmax_max['t10'], minmax_max['t11'], minmax_max['t12'],
                              minmax_max['t13'], minmax_max['t14'], minmax_max['t15'], minmax_max['t16'],
                              minmax_max['t17']], axis=1)
    return M_zs, M_minmax, M_minmax_max, M_minmax_min
