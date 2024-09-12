import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pymc.sampling.jax as pmjax
import pytensor


def sigfit_fitext(data, sigs,additional_signatures_num=1,model_type='Multinomial', tune=2000, draws=2000 ):
    '''
    Function that creates and infers utilizing sigfit model, it's variables are:
    data - pandas dataframe or numpy array of mutational catalogues
    sigs - pandas dataframe with the prior on signatures to be inferred on
    signatures_new - number of additional signatures to be inferred from the data
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
    signatures = sigs.columns
    if mutations != list(data.columns):
        print('mutation types in sigs and data are not compatible')
        raise
    coords = {
        "signatures":signatures,
        'signatures_new':np.ones(signatures_new),
        'signatures_total':list(signatures)+list(np.ones(signatures_new)),
        "mutations":mutations,
        "genomes":genomes
    }
    with pm.Model(coords=coords) as sigfit_fitext:
    #variables
        sigs_new = pm.Dirichlet("sigs_new", a = np.ones(len(coords['mutations']))/2, dims = ('signatures_new', "mutations"))
        e = pm.Dirichlet('e', a=np.ones(len(coords["signatures_total"]))/2, dims=("genomes",'signatures_total'))
        omega = pm.HalfCauchy('omega',beta = 2.5, dims= "genomes")
        if model_type == 'Multinomial':
            mu= pm.Multinomial('mu',n=data.sum(axis=1),p=pm.math.dot(e[:,len(coords['signatures']):],sigs_new)+pm.math.dot(e[:,:len(coords['signatures'])],sigs.values.T), observed=data.to_numpy(), dims=('genomes','mutations'))
        if model_type== 'Poisson':    
            mu = pm.Poisson('mu', pm.math.dot(pt.diag(omega),pm.math.dot(e[:,len(coords['signatures']):],sigs_new))+pm.math.dot(e[:,:len(coords['signatures'])],sigs.values.T), observed=data.to_numpy(), dims=('genomes','mutations'))
        if model_type == 'Normal':
            sigma = pm.HalfCauchy('sigma',beta = 2.5, dims= "genomes")
            mu = pm.Normal('mu', mu =  pm.math.dot(pt.diag(omega),pm.math.dot(e[:,len(coords['signatures']):],sigs_new))+pm.math.dot(e[:,:len(coords['signatures'])],sigs.values.T), sigma = sigma.reshape((coord_len["sig"],1)) @ np.ones(len(coords['mutations'])).reshape(1,96) , observed=data.to_numpy(),dims=('genomes','mutations'))
        if model_type == "NegativeBinomial":
            psi = pm.HalfCauchy('psi',beta = 2.5, dims= "mutations")
            mu = pm.NegativeBinomial('mu', mu =  pm.math.dot(pt.diag(omega),pm.math.dot(e[::,len(coords['signatures']):],sigs_new)+pm.math.dot(e[::,:len(coords["signatures"])],signatures.values.T)), alpha = np.ones(len(coords['genomes'])).reshape((len(coords["genomes"]),1)) @ psi.reshape((1,96)) , observed=data.to_numpy(),dims=('genomes','mutations'))
        infer_data = pmjax.sample_numpyro_nuts(draws, tune=tune)

    return sigfit_fitext, infer_data
