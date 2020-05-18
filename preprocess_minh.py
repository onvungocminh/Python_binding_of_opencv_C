import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.spatial import distance
import os
import re
from sklearn import mixture
from scipy.special import logsumexp
import sklearn.preprocessing
from sklearn.neighbors import KernelDensity


# Compute the log of likelihood ratio

def preprocess(ima_lab, list_fg: np.array, list_bg: np.array):


	h, w, nchannel = ima_lab.shape
	n_comp = 4

	gmm_bg = mixture.GaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_fg = mixture.GaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_bg.fit(list_bg)
	gmm_fg.fit(list_fg)
	print("bg:", gmm_bg.weights_)
	print("fg:", gmm_fg.weights_)

	## Bayesian inference
	## Compute the unnormalized posterior log probability of X
	log_prior = np.log(0.5)
	scores = np.empty((2, w*h))
	scores[0,:] = log_prior + gmm_bg.score_samples(ima_lab.reshape((w*h, nchannel)))
	scores[1,:] = log_prior + gmm_fg.score_samples(ima_lab.reshape((w*h, nchannel)))
	## Normalize by P(X): P(X|0) / P(X) and P(X|1) / P(X)                (actually log P(x))
	P = logsumexp(scores, axis=0)
	print(P)
	log_proba = scores - P[np.newaxis,:]
	proba = np.exp(log_proba)

	score = proba[1,:] #- score_bg # log of likelihood ratio
	score = sklearn.preprocessing.minmax_scale(score)
	score = score.reshape((h,w))
	#print("max:", score.max())
	#print("min:", score.min())

	score1 = proba[0,:] #- score_fg # log of likelihood ratio
	score1 = sklearn.preprocessing.minmax_scale(score1)
	score1 = score1.reshape((h,w))


	return score, score1


# compute the log of the probability

def preprocess_logpro(ima_lab, list_fg: np.array, list_bg: np.array):


	h, w, nchannel = ima_lab.shape
	n_comp = 5

	gmm_bg = mixture.GaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_fg = mixture.GaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_bg.fit(list_bg)
	gmm_fg.fit(list_fg)
	print("bg:", gmm_bg.weights_)
	print("fg:", gmm_fg.weights_)


	proba_fg =  gmm_fg.score_samples(ima_lab.reshape((w*h, nchannel)))
	proba_bg =  gmm_bg.score_samples(ima_lab.reshape((w*h, nchannel)))
	## Normalize by P(X): P(X|0) / P(X) and P(X|1) / P(X)                (actually log P(x))


	proba = proba_fg/ ( proba_fg +proba_bg + 0.0001)
	proba = proba.reshape((h,w))
	proba1 = proba_bg/ ( proba_fg +proba_bg + 0.0001)
	proba1 = proba1.reshape((h,w))


	return proba, proba1





# Predict posterior probability of each component given the data.



def preprocess_postproba(ima_lab, list_fg: np.array, list_bg: np.array):


	h, w, nchannel = ima_lab.shape
	n_comp = 5

	gmm_bg = mixture.BayesianGaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_fg = mixture.BayesianGaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_bg.fit(list_bg)
	gmm_fg.fit(list_fg)
	print("bg:", gmm_bg.weights_)
	print("fg:", gmm_fg.weights_)


	proba_fg =  gmm_fg.score_samples(ima_lab.reshape((w*h, nchannel)))
	proba_bg =  gmm_bg.score_samples(ima_lab.reshape((w*h, nchannel)))
	## Normalize by P(X): P(X|0) / P(X) and P(X|1) / P(X)                (actually log P(x))


	proba = proba_fg/ ( proba_fg +proba_bg+ 0.0001)
	proba = proba.reshape((h,w))
	proba1 = proba_bg/ ( proba_fg +proba_bg+0.0001)
	proba1 = proba1.reshape((h,w))




	return proba, proba1



# Predict posterior probability of each component given the data using the fast kernal density estimation

def preprocess_kde(ima_lab, list_fg: np.array, list_bg: np.array):

	h, w, nchannel = ima_lab.shape
	n_comp = 5

	# instantiate and fit the KDE model
	kde_bg = KernelDensity(bandwidth=1.0, kernel='gaussian')
	kde_fg = KernelDensity(bandwidth=1.0, kernel='gaussian')

	kde_bg.fit(list_bg)
	kde_fg.fit(list_fg)

	# score_samples returns the log of the probability density
	proba_fg = kde_fg.score_samples(ima_lab.reshape((w*h, nchannel)))
	proba_bg = kde_bg.score_samples(ima_lab.reshape((w*h, nchannel)))

	proba = proba_fg/ ( proba_fg +proba_bg)
	proba = proba.reshape((h,w))
	proba1 = proba_bg/ ( proba_fg +proba_bg)
	proba1 = proba1.reshape((h,w))

	return proba, proba1



# Raw segmentation


def preprocess_raw(ima_lab, list_fg: np.array, list_bg: np.array):


	h, w, nchannel = ima_lab.shape
	n_comp = 5

	gmm_bg = mixture.GaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_fg = mixture.GaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
	gmm_bg.fit(list_bg)
	gmm_fg.fit(list_fg)
	print("bg:", gmm_bg.weights_)
	print("fg:", gmm_fg.weights_)


	proba_fg =  gmm_fg.score_samples(ima_lab.reshape((w*h, nchannel)))
	proba_bg =  gmm_bg.score_samples(ima_lab.reshape((w*h, nchannel)))
	## Normalize by P(X): P(X|0) / P(X) and P(X|1) / P(X)                (actually log P(x))


	proba = proba_fg/ ( proba_fg +proba_bg)
	proba = proba.reshape((h,w))
	proba1 = proba_bg/ ( proba_fg +proba_bg)
	proba1 = proba1.reshape((h,w))


	return proba, proba1