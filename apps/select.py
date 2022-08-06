import streamlit as st

def app():
    st.title("Identification and prediction of Parkinson disease subtypes and progression using machine learning in two cohorts")
    st.write("Anant Dadu, Vipul K Satone, Rachneet Kaur, Sayed Hadi Hashemi, Hampton Leonard, Hirotaka Iwaki, Mary B Makarious, Kimberley J Billingsley, Sara Bandres-Ciga, Lana Sargent, Alastair Noyce, Ali Daneshmand, Cornelis Blauwendraat, Ken Marek, Sonja W. Scholz, Andrew Singleton, Mike A Nalls, Roy Campbell, Faraz Faghri")
    st.write("bioRxiv 2022.08.04.502846; doi: https://doi.org/10.1101/2022.08.04.502846")
    st.write("## Summary")
    st.write("### Background")
    st.write("The clinical manifestations of Parkinson’s disease (PD) are characterized by heterogeneity in age at onset, disease duration, rate of progression, and the constellation of motor versus non-motor features. There is an unmet need for the characterization of distinct disease subtypes as well as improved, individualized predictions of the disease course. The emergence of machine learning to detect hidden patterns in complex, multi-dimensional datasets provides unparalleled opportunities to address this critical need.")
    st.write("### Methods and Findings")
    st.write("""We used unsupervised and supervised machine learning methods on
comprehensive, longitudinal clinical data from the Parkinson’s Disease Progression Marker
Initiative (PPMI) (n = 294 cases) to identify patient subtypes and to predict disease progression.
The resulting models were validated in an independent, clinically well-characterized cohort
from the Parkinson’s Disease Biomarker Program (PDBP) (n = 263 cases). Our analysis
distinguished three distinct disease subtypes with highly predictable progression rates,
corresponding to slow, moderate, and fast disease progression. We achieved highly accurate
projections of disease progression five years after initial diagnosis with an average area under
the curve (AUC) of 0.92 (95% CI: 0.95 ± 0.01 for the slower progressing group (PDvec1), 0.87 ±
0.03 for moderate progressors, and 0.95 ± 0.02 for the fast progressing group (PDvec3). We
identified serum neurofilament light (Nfl) as a significant indicator of fast disease progression
among other key biomarkers of interest. We replicated these findings in an independent
validation cohort, released the analytical code, and developed models in an open science
manner.""")
    st.write("### Conclusions")
    st.write("""Our data-driven study provides insights to deconstruct PD heterogeneity. This
approach could have immediate implications for clinical trials by improving the detection of
significant clinical outcomes that might have been masked by cohort heterogeneity. We
anticipate that machine learning models will improve patient counseling, clinical trial design,
allocation of healthcare resources, and ultimately individualized patient care.""")
    # st.write("### Funding")
    # st.write("US National Institute on Aging, US National Institutes of Health, Italian Ministry of Health, European Commission, University of Torino Rita Levi Montalcini Department of Neurosciences, Emilia Romagna Regional Health Authority, and Italian Ministry of Education, University, and Research.")
    # st.write("### Translations")
    # st.write("For the Italian and German translations of the abstract see Supplementary Materials section.")
    # st.write("For the Italian and German translations of the abstract see Supplementary Materials section.")
    st.write("## Citation")
    st.write(""" "Identification and prediction of Parkinson disease subtypes and progression using machine learning in two cohorts.". \[[article](https://www.biorxiv.org/content/10.1101/2022.08.04.502846v1)\]\[[supplementary materials](https://www.biorxiv.org/content/10.1101/2022.08.04.502846v1.supplementary-material)\]\[[github](https://github.com/anant-dadu/PD_progression_subtypes)\]\[[website](https://anant-dadu-pdprogressionsubtypes-streamlit-app-aaah95.streamlitapp.com/)\]""")
    # st.write("## Summary")
    # st.write("### Background")
    # st.write("The disease entity known as amyotrophic lateral sclerosis (ALS) is now known to represent a collection of overlapping syndromes. A better understanding of this heterogeneity and the ability to distinguish ALS subtypes would improve the clinical care of patients and enhance our understanding of the disease. Subtype profiles could be incorporated into the clinical trial design to improve our ability to detect a therapeutic effect. A variety of classification systems have been proposed over the years based on empirical observations, but it is unclear to what extent they genuinely reflect ALS population substructure.")
    # st.write("### Methods")
    # st.write("We applied machine learning algorithms to a prospective, population-based cohort consisting of 2,858 Italian patients diagnosed with ALS for whom detailed clinical phenotype data were available. We replicated our findings in an independent population-based cohort of 1,097 Italian ALS patients.")
    # st.write()

# app()
