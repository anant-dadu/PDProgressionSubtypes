import pandas as pd
from sklearn import mixture
def apply_GMM(M_PD_gmm_chosen, n_clusters=3, trained_model=None):
    if trained_model is None:
        model_gmm = mixture.GaussianMixture(n_components=3, covariance_type='tied', random_state=42)
        model_gmm.fit(M_PD_gmm_chosen)  # print(gmm.means_)
        bic_score = []
        components = []
        for n_components in [1, 2,3,4,5,6,10]:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='tied', random_state=42)
            gmm.fit(M_PD_gmm_chosen)
            bic_score.append(gmm.bic(M_PD_gmm_chosen))
            components.append(n_components)
        predictions = model_gmm.predict(M_PD_gmm_chosen)
        return model_gmm, predictions, pd.DataFrame({'bic_score':bic_score, 'n_components': components})
    else:
        model_gmm = trained_model
        # PLEASE BEWARE: CHECK FOR VARIANCE along dimensions
        if M_PD_gmm_chosen.shape[1] == 2:
            M_PD_gmm_chosen = M_PD_gmm_chosen[M_PD_gmm_chosen.var().sort_values().index].copy()
            M_PD_gmm_chosen.columns = ['latent_weight1', 'latent_weight2']

        print (M_PD_gmm_chosen.var())
        print (model_gmm.means_.var(0))
        print ('='*50)
        # import pdb; pdb.set_trace()

    predictions = model_gmm.predict(M_PD_gmm_chosen)
    return model_gmm, predictions, None