import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pymc.sampling.jax as pmjax
import pytensor

def model_fit(data, sigs, clusters_num,clustering_prior=np.empty(0), model_type='Multinomial', tune=2000, draws=2000):
    '''
    Function that created and infers utilizing ** model, it's variables are:
    data - pandas dataframe of mutational catalogues
    sigs - pandas dataframe with the prior on signatures to be inferred on
    clusters_num - integer number of clusters in the model
    clustering_prior - numpy array with the information of clustering in the data, defaultly an uniform distribution.
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
    if mutations != list(data.columns):
        print('mutation types in sigs and data are not compatible')
        raise
    clusters = np.arange(clusters_num)

    coords = {
        "signatures":signatures,
        "mutations":mutations,
        "genomes":genomes,
        "clusters":clusters
    }
    if clustering_prior.size==0:
        clustering_prior = np.ones(clusters_num)/2
    ones_genomes = pytensor.shared(np.ones(len(coords['genomes'])).reshape((len(coords['genomes']),1)))
    ones_mutations = pytensor.shared(np.ones(len(coords['mutations'])).reshape(1,len(coords['mutations'])))

    with pm.Model(coords=coords) as mymodel_fit:
    
        #constant variables
        pi = pm.Dirichlet('pi', a=np.ones(len(coords["signatures"]))/2, dims=("clusters",'signatures'))
        upsilon = pm.HalfCauchy('upsilon',beta = 2.5, dims= 'clusters')
        tau = pm.HalfCauchy('sigma',beta = 2.5, dims= "clusters")
        e_c = pm.Dirichlet('e_c', a= clustering_prior, dims=('genomes','clusters'))
        eta = pm.math.dot(e_c.T, data.to_numpy())
        mu = pm.math.dot(pm.math.dot(e_c,pi),sigs.values.T)

        c_mut = pm.model.core.Potential('c_mut',pm.logp(pm.Normal('c_mut_n',mu = pm.math.dot(pt.diag(upsilon),pm.math.dot(pi,sigs.values.T)), sigma=tau.reshape((len(coords['clusters']),1)) @ ones_mutations, dims=('clusters', 'mutations')),eta))
        if model_type == 'Multinomial':
            g_mut= pm.Multinomial('g_mut',n=data.sum(axis=1),p=mu, observed=data.values, dims=('genomes','mutations'))
        elif model_type== 'Poisson':
            omega_g = pm.HalfCauchy('omega_g',beta = 2.5, dims= "genomes")
            g_mut = pm.Poisson('g_mut', pm.math.dot(pt.diag(omega_g),mu), observed=data.values, dims=('genomes','mutations'))
        elif model_type == 'Normal':
            omega_g = pm.HalfCauchy('omega_g',beta = 2.5, dims= "genomes")
            sigma_g = pm.HalfCauchy('sigma_g',beta = 2.5, dims= "genomes")
            g_mut = pm.Normal('g_mut', mu =  pm.math.dot(pt.diag(omega_g),mu), sigma = sigma_g.reshape((len(coords['genomes']),1)) @ ones_mutations , observed=data.values,dims=('genomes','mutations'))
        elif model_type == 'NegativeBinomial':
            omega_g = pm.HalfCauchy('omega_g',beta = 2.5, dims= "genomes")
            psi = pm.HalfCauchy('psi',beta = 2.5, dims= "mutations")
            g_mut = pm.NegativeBinomial('g_mut', mu =  pm.math.dot(pt.diag(omega_g),mu), alpha = ones_genomes @ psi.reshape((1,96)) , observed=data.values,dims=('genomes','mutations'))
        else:
            print('stated model is incorrect. Please check spelling and capital letters')
            raise
        infer_data = pmjax.sample_numpyro_nuts(draws=draws, tune=tune)

    return mymodel_fit, infer_data
