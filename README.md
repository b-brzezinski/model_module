# Models
The models section of the library consists of four Python files, each consisting of the source code for either the proposed combined model or the sigfit model without mutational opportunities.
The latter has been previously implemented using Stan and its R-based interface~\cite{sigfit}, but here we provide a Python re-implementation as well.
To perform inference on the sigfit model, the user needs to use sigfit_fit or sigfit_fitext. The first function performs only refitting to preexisting signatures, while the latter allows for both signature extraction and fitting.
The sigfit_fit function has five parameters: \texttt{data, sigs, model_type, tune, draws}, where the two first parameters are mandatory. 
    * data and sigs are pandas dataframe objects. The former contains mutational catalogues in rows and columns named after mutation types. The latter is indexed by mutations from the chosen alphabet and contains mutational signatures in its columns. Comparing to the formal description of the model, the function takes the signature matrix in its transposed form. This representation has been chosen because it allows the user to utilize COSMIC signatures with the least amount of additional operations. All inference functions check the correctness of the input data, i.e. whether both selected data and signatures share the same mutation alphabet and order. 
    * model_type parameters] is used to specify which of four model types should be used: Multinomial, Normal, Poisson, or Negative Binomial. If not specified by the user, the multinomial mode will be used.
    * tune and draws parameters are used to specify the length of sampling, where the sampling will do tune+draws steps, with discarding first tune steps. 

The sigfit_fitext function takes one additional parameter: additional_signatures_num, which specifies the number of signatures to be extracted from the data.


Implementation of the proposed combined model is in mymodel_fit and mymodel_fitext functions. Their inputs are mostly similar to those in sigfit, only adding clusters_num and optional clustering_prior. The clustering_prior parameter allows for passing some prior knowledge about the clustering properties of catalogues.

All of the functions in models utilize PyMC library to build suitable models and perform sampling. PyMC is a probabilistic programming library for Python that allows users to build Bayesian models with a simple Python API and fit them using Markov chain Monte Carlo (MCMC) methods. It accepts data in the form of numpy arrays. The flexibility of the pymc library allows for a multitude of samplers to be used in models from Metropolis-Hastings sampling and Gibbs sampling to No-U-Turns sampling. PyMC also allows for third-party samplers utilizing the Jax library. Jax-based samplers offer hardware acceleration by utilizing the CUDA capabilities of GPUs.

Each function returns a tuple of a trained PyMC model object and an ArviZ InferenceData object. Both of those resulting objects can used in reconstructing latent variables.
# Utilities
Utilities submodule contains functions that help the user to process and summarize the results. InferenceData objects hold information from every non-discarded step of sampling, and the package allows for the recreation of any of the matrices from the following list:
    * exposure matrix $e=\varepsilon\times\pi$,
    * cluster exposure over genomes matrix $\epsilon$,
    * signature exposures over clusters matrix $\pi$,
    * part of matrix $S$ with newly inferred signatures,
    * data matrix reconstructed from sampling posterior predictive distribution.
The utilities submodule also contains functions for calculating the log posterior probability of the model, finding the chain with the best log posterior probability, cosine similarity between matrices utilizing scipy library, and plotting the signature. I also included a function that allows the user to create a list of loglikelihoods of the model across different numbers of clusters to be used in the elbow method.

# libraries used
pandas,
numpy,
scipy,
matplotlib,
seaborn,
scikit learn,
pymc,
pytensor,
arviz
