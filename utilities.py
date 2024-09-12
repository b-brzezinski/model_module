import numpy as np
import pandas as pd
import scipy.spatial as spatial
import pytensor.tensor as pt
import pymc as pm
import matplotlib.ticker as ticker
import seaborn as sns
from models._mymodel_fit import model_fit
from models._mymodel_fitext import model_fitext
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
def exposures_reconstruction(infer_data, chain=0):
    '''
    using:
    infer_data - arviz dataset object
    chain - chain from which data will be taken
    it calculates signature exposures per patient.
    returns a pandas dataframe with genomes as rows and sginatures as columns.
    '''
    e_c = infer_data.posterior['e_c']
    pi = infer_data.posterior['pi']
    exposures = e_c.dot(pi,['clusters']).mean(dim=['draw'])[chain].to_dataframe(name='exposures')
    exposures = exposures.drop(columns=['chain'])
    exposures = exposures.reset_index()
    exposures = pd.pivot_table(exposures,values='exposures',index='genomes',columns='signatures')
    return exposures

def exposures_reconstruction_sigfit(infer_data, chain=0):
    '''
    using:
    infer_data - arviz dataset object
    chain - chain from which data will be taken
    it calculates signature exposures per patient.
    returns a pandas dataframe with genomes as rows and sginatures as columns.
    '''
    exposures = infer_data.posterior['e'].mean(dim=['draw'])[chain].to_dataframe(name='exposures')
    exposures = exposures.drop(columns=['chain'])
    exposures = exposures.reset_index()
    exposures = pd.pivot_table(exposures,values='exposures',index='genomes',columns='signatures')
    return exposures

def cluster_mix_reconstruction(infer_data, chain=0):
    '''
    using:
    infer_data - arviz dataset object
    chain - chain from which data will be taken
    it calculates cluster exposures per patient.
    returns a pandas dataframe with genomes as rows and clusters as columns.
    '''

    mix = infer_data.posterior['e_c'].mean(dim=['draw'])[chain].to_dataframe(name='mix')
    mix = mix.drop(columns=['chain'])
    mix = mix.unstack()
    mix = mix.droplevel(0, axis=1)
    
    return mix

def cluster_exposures_reconstruction(infer_data,chain):
    '''
    using:
    infer_data - arviz dataset object
    it calculates signature exposures per inferred cluster.
    returns a pandas dataframe with clusters as rows and signatures as columns.
    '''
    pi = infer_data.posterior['pi'].mean(dim=['draw'])[chain].to_dataframe(name='pi')
    pi = pi.drop(columns=['chain'])
    pi = pi.unstack()
    pi = pi.droplevel(0, axis=1)
    
    return pi

def newsigs_reconstruction(infer_data,chain=0):
    '''
    using:
    infer_data - arviz dataset object
    chain - chain from which data will be taken
    it calculates new signatures found in the mutational catalogues.
    '''
    recon_data = infer_data.posterior['Sigs_ext'].mean(dim=['draw'])[chain].to_dataframe()
    recon_data = recon_data.drop(columns='chain')
    recon_data = recon_data.stack()
    recon_data = recon_data.droplevel(2)
    recon_data = recon_data.unstack()
    return recon_data

def newsig_plot(signature,title):
    '''
    Given:
    signature - pandas dataframe with a SBS signature
    title - string
    it creates a plot of given signature.
    '''
    mutations = []
    palette = {}
    bases = ['A','C','G','T']
    pos_mutations = ['[C>A]','[C>G]','[C>T]','[T>A]','[T>C]','[T>G]']
    mut_color = {'[C>A]':'#02BCED','[C>G]':'#010101','[C>T]':'#E22926','[T>A]':'#CAC8C9','[T>C]':'#A0CE62','[T>G]':'#ECC6C5'}
    for mutation in pos_mutations:
        for base_3 in bases:
            for base_5 in bases:

                mutations.append(f'{base_3}{mutation}{base_5}')
                palette[base_3+mutation+base_5]=mut_color[mutation]
    def sorter(column):
        cat = pd.Categorical(column, categories=mutations, ordered=True)
        return pd.Series(cat)
    signature = signature.sort_index(key=sorter)
    plot_sig7 = sns.barplot(x=signature.index,y=signature*100,palette=palette)
    plot_sig7.figure.set_size_inches((4*4.42,4))
    plot_sig7.yaxis.set_major_formatter(ticker.PercentFormatter())
    plot_sig7.tick_params(axis='x',rotation=90)
    plot_sig7.figure.dpi= 350
    plot_sig7.set_title(title)
    plot_sig7.set_ylabel('percentage of SBS')
    plot_sig7.set_xlabel('type of mutation')

def similarity(original_data, reconstructed_data,chain=0):
    '''
    given:
    original_data - a pandas dataframe
    reconstructed - a pandas dataframe
    it returns cosine similarity score
    '''
    return 1 - spatial.distance.cosine(original_data.values.ravel(), reconstructed_data.values.ravel())

def data_reconstruction(model, infer_data,chain=0):
    '''
    given:
    model - pymc model object
    infer_data - arviz dataset object
    chain - chain from which data will be taken
    the function creates a reconstruction of the dataset using mean of the inferred posterior predictives of mutation counts.
    '''
    with model:
        pm.sample_posterior_predictive(infer_data, extend_inferencedata=True)
    recon_data = infer_data.posterior_predictive['g_mut'].mean(dim=['draw'])[chain].to_dataframe()
    recon_data = recon_data.drop(columns='chain')
    recon_data = recon_data.stack()
    recon_data = recon_data.droplevel(2)
    recon_data = recon_data.unstack()
    return recon_data

def calculate_logposterior(infer_data,chain=0):
    '''
    given:
    infer_data - arviz dataset object
    chain - chain from which data will be taken
    '''
    return infer_data.sample_stats['lp'].values[chain]

def calculate_loglikelihood(model,inferdata,chain=0):
    with model:
        pm.compute_log_likelihood(inferdata)
    return inferdata.log_likelihood['g_mut'][chain].sum(dim='genomes').values

def best_chain(infer_data):
    '''
    given:
    infer_data - arviz dataset object
    function returns index of the simulation with the best logposterior probability
    '''    
    temp_dict = {}
    for chain in np.arange(len(infer_data.sample_stats['lp'])):
        temp_dict[chain]=calculate_logposterior(infer_data,chain).mean()
    return max(temp_dict,key= lambda key: temp_dict[key])

def BIC_clusters_fit(data,signatures,min_clusters=1,max_clusters=5,model_type='Multinomial',tune=2000,draws=2000,prior='unif'):
    '''
    Given:
    data - pandas Dataframe with mutational catalogues
    signatures - pandas Dataframe with mutational signatures
    min_clusters, max_clusters - integers assigning the range to be tested
    model_type - string with type of model to be used, from set {'Multinomial','Poisson','NegativeBinomial','Normal'}
    tune, draws - specification of length of sampling
    prior - string assigning what kind of prior to use for data. from set {'unif','NMF','k-means'}
    it returns a list of log likelihoods of the models trained on given number of clusters.
    '''
    best_BIC='first'
    best_model = None
    best_data = None
    n = len(data)
    logliks =  []
    if prior not in {'unif','NMF','k-means'}:
        print('wrong prior, please use \'unif\', \'NMF\' or \'k-means\'')
    for cluster_num in range(min_clusters,max_clusters+1):
        if prior == 'unif':
            clust_prior = np.empty(0)
        elif prior == 'NMF':
            nmf = NMF(cluster_num)
            nmf.fit(data.transpose().values)
            clust_prior = nmf.components_.T+0.05*np.ones((len(data),cluster_num))
        elif prior == 'k-means':
            kmeans = KMeans(cluster_num)
            kmeans.fit(data.values)
            clusters = 0.05*np.ones((len(data),cluster_num))
            for i in range(len(kmeans.labels_)):
                clusters[i,kmeans.labels_[i]]+=1

        curr_model, curr_data = model_fit(data = data,sigs = signatures,clusters_num = cluster_num,model_type=model_type,tune= tune,draws=draws,clustering_prior=clust_prior)
        b_chain = best_chain(curr_data)
        loglik = np.mean(calculate_loglikelihood(curr_model,curr_data,b_chain))
        logliks.append(loglik)
        if best_BIC=='first':
            best_BIC= loglik
            best_model = curr_model
            best_data = curr_data
            best_c = cluster_num
        else:
            if loglik > best_BIC:
                best_BIC = loglik
                best_model = curr_model
                best_data = curr_data
                best_c = cluster_num
            
    return logliks


def BIC_clusters_fitext(data,signatures,new_signatures=1,min_clusters=1,max_clusters=5,model_type='Multinomial',tune=2000,draws=2000,prior='unif'):
    '''
    Given:
    data - pandas Dataframe with mutational catalogues
    signatures - pandas Dataframe with mutational signatures
    new_signatures - number of signatures to be inferred from the data
    min_clusters, max_clusters - integers assigning the range to be tested
    model_type - string with type of model to be used, from set {'Multinomial','Poisson','NegativeBinomial','Normal'}
    tune, draws - specification of length of sampling
    prior - string assigning what kind of prior to use for data. from set {'unif','NMF','k-means'}
    it returns a list of log likelihoods of the models trained on given number of clusters.

    '''
    best_BIC='first'
    best_model = None
    best_data = None
    n = len(data)
    if prior not in {'unif','NMF','k-means'}:
        print('wrong prior, please use \'unif\', \'NMF\' or \'k-means\'')
    for cluster_num in range(min_clusters,max_clusters+1):
        if prior == 'unif':
            clust_prior = np.empty(0)
        elif prior == 'NMF':
            nmf = NMF(cluster_num)
            nmf.fit(data.transpose().values)
            clust_prior = nmf.components_.T+0.05*np.ones((len(data),cluster_num))
        elif prior == 'k-means':
            kmeans = KMeans(cluster_num)
            kmeans.fit(data.values)
            clusters = 0.05*np.ones((len(data),cluster_num))
            for i in range(len(kmeans.labels_)):
                clusters[i,kmeans.labels_[i]]+=1

        curr_model, curr_data = model_fitext(data = data,sigs = signatures,new_signatures=new_signatures,clusters_num = cluster_num,model_type=model_type,tune= tune,draws=draws,clustering_prior=clust_prior)
        b_chain = best_chain(curr_data)
        loglik = np.mean(calculate_loglikelihood(curr_model,curr_data,b_chain))
        curr_BIC = cluster_num*(len(signatures.columns)-1) * np.log(n) - 2*loglik
        if best_BIC=='first':
            best_BIC= curr_BIC
            best_model = curr_model
            best_data = curr_data
            best_c = cluster_num
        else:
            if curr_BIC < best_BIC:
                best_BIC = curr_BIC
                best_model = curr_model
                best_data = curr_data
                best_c = cluster_num
            
    return best_c
