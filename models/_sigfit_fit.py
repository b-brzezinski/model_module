import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pymc.sampling.jax as pmjax
import pytensor


def sigfit_fit(data, sigs, model_type='Multinomial', tune=2000, draws=2000 ):
    '''
    Function that creates and infers utilizing sigfit model, it's variables are:
    data - pandas dataframe or numpy array of mutational catalogues
    sigs - pandas dataframe with the prior on signatures to be inferred on
    model_type - different types of random variables to model mutation counts:
        'Multinomial'
        'Poisson'
        'Normal'
        'NegativeBinomial'
    tune - number of tuning draws of the sampler 
    draws - number of draws to be performed after tuning process
    
    it returns a tuple:
    sigfit_fit - pymc model object that can be used for further analysis
    infer_data - arviz dataset object with the inference data
    '''
    genomes = list(data.index)
    mutations = list(sigs.index)
    if mutations != list(data.columns):
        print('mutation types in sigs and data are not compatible')
        raise
    signatures = sigs.columns
    

    coords = {
        "signatures":signatures,
        "mutations":mutations,
        "genomes":genomes
    }
    with pm.Model(coords=coords) as sigfit_fit:
        
        e = pm.Dirichlet('e', a=np.ones(len(coords["signatures"]))/2, dims=("genomes",'signatures'))
        omega = pm.HalfCauchy('omega',beta = 2.5, dims= "genomes")
        if model_type == "Multinomial":
            g_mut= pm.Multinomial('g_mut',n=data.sum(axis=1),p=pm.math.dot(e,sigs.to_numpy().T), observed=data.to_numpy(), dims=('genomes','mutations'))
        if model_type == "Poisson":    
            g_mut = pm.Poisson('g_mut', pm.math.dot(pt.diag(omega),pm.math.dot(e,sigs.to_numpy().T)), observed=data.to_numpy(), dims=('genomes','mutations'))
        if model_type == "Normal":
            sigma = pm.HalfCauchy('sigma',beta = 2.5, dims= "genomes")
            g_mut = pm.Normal('g_mut', mu =  pm.math.dot(pt.diag(omega),pm.math.dot(e,sigs.to_numpy().T)), sigma = sigma.reshape((16,1)) @ np.ones(len(coords['mutations'])).reshape(1,96) , observed=data.to_numpy(),dims=('genomes','mutations'))
        if model_type == "NegativeBinomial":
            psi = pm.HalfCauchy('psi',beta = 2.5, dims= "mutations")
            g_mut = pm.NegativeBinomial('g_mut', mu =  pm.math.dot(pt.diag(omega),pm.math.dot(e,sigs.to_numpy().T)), alpha = np.ones(len(coords['genomes'])).reshape((len(coords['genomes']),1)) @ psi.reshape((1,96)) , observed=data.to_numpy(),dims=('genomes','mutations'))
        infer_data = pmjax.sample_numpyro_nuts(draws, tune=tune)

    return sigfit_fit, infer_data
