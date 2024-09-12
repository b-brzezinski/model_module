import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pymc.sampling.jax as pmjax
import pytensor


def model_fitext(data, sigs, clusters_num,clustering_prior=np.empty(0), additional_signatures_num=1, model_type='Multinomial', tune=2000, draws=2000 ):
    '''
    Function that created and infers utilizing ** model, it's variables are:
    data - pandas dataframe or numpy array of mutational catalogues
    sigs - pandas dataframe with the prior on signatures to be inferred on
    clusters_num - integer number of clusters in the model
    clustering_prior - numpy array with the information of clustering in the data, defaultly an uniform distribution.
    additional_signatures_num - number of additional signatures to be found in the model
    model_type - different types of random variables to model mutation counts:
        'Multinomial'
        'Poisson'
        'Normal'
        'NegativeBinomial'
    tune - number of tuning draws of the sampler 
    draws - number of draws to be performed after tuning process
    
    it returns a tuple:
    mymodel_fit - pymc model object that can be used for further analysis
    infer_data - arviz dataset object with the inference data
    '''
    genomes = list(data.index)
    mutations = list(sigs.index)
    signatures = sigs.columns
    clusters = np.arange(clusters_num)
    if mutations != list(data.columns):
        print('mutation types in sigs and data are not compatible')
        raise
    if clustering_prior.size==0:
        clustering_prior = np.ones(clusters_num)/2
        
    coords = {
        "signatures_const":signatures,
        "signatures_additional":np.arange(additional_signatures_num),
        "signatures_total":list(signatures)+list(np.arange(additional_signatures_num)),
        "mutations":mutations,
        "genomes":genomes,
        "clusters":clusters
    }
    with pm.Model(coords=coords) as mymodel_fit:
    
        pi = pm.Dirichlet('pi', a=np.ones(len(coords["signatures_total"]))/2, dims=("clusters",'signatures_total'))
        Sigs_ext = pm.Dirichlet('Sigs_ext', a=np.ones(len(coords['mutations']))/2, dims = ('signatures_additional','mutations'))
        omega_g = pm.HalfCauchy('omega_g',beta = 2.5, dims= "genomes")
        omega_c = pm.HalfCauchy('omega_c',beta = 2.5, dims= 'clusters')
        sigma_c = pm.HalfCauchy('sigma',beta = 2.5, dims= "clusters")
        e_c = pm.Dirichlet('e_c', a= np.ones(len(coords['clusters']))/2, dims=('genomes','clusters'))
        
        eta = pm.math.dot(e_c.T, data.to_numpy())
        c_mut = pm.model.core.Potential('c_mut',pm.logp(pm.Normal('c_mut_n',mu = pm.math.dot(pt.diag(omega_c),pm.math.dot(pi[:,:len(coords['signatures_const'])],sigs.values.T))+pm.math.dot(pi[:,len(coords['signatures_const']):],Sigs_ext), sigma=sigma_c.reshape((len(coords['clusters']),1)) @ np.ones(len(coords['mutations'])).reshape(1,len(coords['mutations'])), dims=('clusters', 'mutations')),eta))
        
        mu = pm.math.dot(pm.math.dot(e_c,pi[:,:len(coords['signatures_const'])]),sigs.values.T)+pm.math.dot(pm.math.dot(e_c,pi[:,len(coords['signatures_const']):]),Sigs_ext)
        if model_type == 'Multinomial':
            g_mut= pm.Multinomial('g_mut',n=data.sum(axis=1),p=mu, observed=data.to_numpy(), dims=('genomes','mutations'))
        if model_type == 'Poisson':    
            g_mut = pm.Poisson('g_mut', pm.math.dot(pt.diag(omega_g),mu), observed=data.to_numpy(), dims=('genomes','mutations'))
        if model_type == 'Normal':
            sigma_g = pm.HalfCauchy('sigma',beta = 2.5, dims= "genomes")
            g_mut = pm.Normal('g_mut', mu =  pm.math.dot(np.diag(omega_g),mu), sigma = sigma_g.reshape((len(coords['genomes']),1)) @ np.ones(len(coords['mutations'])).reshape(1,96) , observed=data.to_numpy(),dims=('genomes','mutations'))
        if model_type == 'NegativeBinomial':
            psi = pm.HalfCauchy('psi',beta = 2.5, dims= "mutations")
            g_mut = pm.NegativeBinomial('g_mut', mu =  pm.math.dot(pt.diag(omega_g),mu), alpha = np.ones(len(coords['genomes'])).reshape((len(coords['genomes']),1)) @ psi.reshape((1,96)) , observed=data.to_numpy(),dims=('genomes','mutations'))
        infer_data = pmjax.sample_numpyro_nuts(draws=draws, tune=tune)

    return mymodel_fit, infer_data
