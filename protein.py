import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd 
from tqdm import tqdm 
from numba import njit, objmode 
from math import comb 
from collections import defaultdict
import matplotlib.pyplot as plt 
import seaborn as sns 

def log_lik(inclusion, X, y, g=96, n=96, T=1):
    model = LinearRegression()
    
    X_incl = X[:, inclusion == 1]
    p = X_incl.shape[1]

    if p < 1:
        return 0
    else:
        model.fit(X_incl, y)
        r_sq = model.score(X_incl, y)

    log_marg_lik = 0.5 * ((n - p - 1) * np.log(1 + g) - (n - 1) * np.log(1 + g * (1 - r_sq)))
    return log_marg_lik / T

def tempered_log_lik(log_lik, T):
    return log_lik / T

def renormalized_model(models, x, y, p=88, T=1):
    n_models = len(models)
    likelihoods = np.zeros(n_models)
     
    for i in range(n_models):
        likelihoods[i] = log_lik(models[i], x, y, T=T)
    
    likelihoods *= T 
    posteriors = np.exp(likelihoods - likelihoods.max())
    #posteriors *= T
    posteriors /= posteriors.sum()
    pi_j = np.zeros(p)

    for j in range(p):
        pi_j[j] = np.sum(posteriors[models[:, j] == 1])
        #print(posteriors[models[:, j]])
    return pi_j 


def local_optim(model, s, X, y, T=1):
    all_indices = np.arange(88)
    # Keep the large jump indices s unchanged 
    rem_idx = np.setdiff1d(all_indices, s)
    # take subset for computational reasons 
    subset = np.random.choice(rem_idx, size=20, replace=False)
    best_lik = log_lik(model, X, y, T=T)
    best_model = model.copy()
    

    cur = best_model.copy() 
    for i in range(0, len(subset)):
        for j in range(i + 1, len(subset)):
            cur = best_model.copy()
            # Flip two covariates
            cur[subset[i]] = 1 - cur[subset[i]]
            cur[subset[j]] = 1 - cur[subset[j]]
            cur_lik = log_lik(cur, X, y, T=T)
            if cur_lik > best_lik:
                best_lik = cur_lik
                best_model = cur.copy()
        
    return best_model

def randomization(model, p_swap=0.1):
    new_model = model.copy()
    for j in range(len(model)):
        if np.random.rand() < p_swap:
            new_model[j] = 1 - model[j]
    return new_model 


def mjmcmc(num_iterations, X, y, p, large_prob=0.05, T=1):
    print("Large jump probability: ", large_prob)
    sample = np.zeros((num_iterations, p), dtype=int)
    posteriors = np.zeros(num_iterations)
    count = 0

    init_model = np.random.choice([0, 1], size=p)
    sample[0] = init_model 
   
    for i in range(1, num_iterations):
        current = sample[i-1].copy()
        
        if np.random.rand() <= large_prob:
            proposed = current.copy()
            # Generate large jump indices  
            s = np.random.choice(p, 4, replace=False)
            proposed[s] = 1 - proposed[s]
            # Perform local optimization keeping the large swap indices unchanged 
            #pre_opt_lik = log_lik(proposed, X, y)
            proposed = local_optim(proposed, s, X, y, T=T)
            #post_opt_lik = log_lik(proposed, data)
            #print(f"Local opt change: {post_opt_lik - pre_opt_lik}")
            chi_k_star = proposed.copy() 

            # Perform randomization
            proposed = randomization(proposed)
            # Backwards large jump swapping the components swapped in the large jump  
            reverse = proposed.copy()
            reverse[s] = 1 - reverse[s]

            # Backwards local optimimization 
            reverse = local_optim(reverse, s, X, y, T=T)
            
            # Calculate likelihoods 
            prop_lik = log_lik(proposed, X, y, T=T)
            current_lik = log_lik(current, X, y, T=T)

            # Calculate model probabilities
            swaps_cur = np.where(reverse != proposed)[0]
            current_prob = np.log(0.1**len(swaps_cur) /  comb(p, len(swaps_cur)))

            swaps_prop = np.where(proposed != chi_k_star)[0]
            prop_prob = np.log(0.1**len(swaps_prop) / comb(p, len(swaps_prop)))
        else:
            # Perform swap 
            s = np.random.choice(p, 1, replace=False)
            proposed = current.copy()
            proposed[s] = 1 - proposed[s]
            
            # Calculate likelihoods 
            prop_lik = log_lik(proposed, X, y, T=T)
            current_lik = log_lik(current, X, y, T=T)
            
            #print("proposed: ", proposed, "log_lik: ", prop_lik)
            #print("current: ", current, "log_lik: ", current_lik)

            # Calculate model probabilities
            # set to 0 since uniform priors cancel out 
            current_prob = 0 
            prop_prob = 0 
            
        mh_ratio = min(0, (prop_lik + current_prob) - (current_lik + prop_prob))
        u = np.random.rand()
        if np.log(u) < mh_ratio:
            sample[i] = proposed
            count += 1 
        else:
            sample[i] = current
        
    print("Temperature: ", T) 
    print(f"Acceptance rate: {count/num_iterations:.2f}")
    unique_vals, counts = np.unique(sample, axis=0, return_counts=True)
    print(f"Unique models visited: {len(unique_vals)}")
    return sample

df = pd.read_csv("data/proteincen.txt", header=None, sep=" ")
y = df.iloc[:, 0]
x = df.iloc[:, 1:]
y = y.to_numpy()
x = x.to_numpy()

sample = mjmcmc(30000, x, y, 88, large_prob=0.05, T=2)
pi_j = renormalized_model(sample, x, y)
unique_samples = np.unique(sample, axis=0)
no_unique = len(unique_samples)
#posteriors_mj = np.zeros(no_unique)
likelihoods_mj = np.zeros(no_unique) 
for j in range(no_unique):
    likelihoods_mj[j] = log_lik(unique_samples[j], x, y)
posteriors_mj = np.exp(likelihoods_mj)
print("Large jumps enabled: ")
print("Total posterior: ", posteriors_mj.sum())

sample2 = mjmcmc(30000, x, y, 88, large_prob=0.00, T=2)

pi_j = renormalized_model(sample2, x, y)
unique_samples = np.unique(sample2, axis=0)
no_unique = len(unique_samples)
#posteriors_mj = np.zeros(no_unique)
likelihoods_mc = np.zeros(no_unique) 
for j in range(no_unique):
    likelihoods_mc[j] = log_lik(unique_samples[j], x, y)
posteriors_mc = np.exp(likelihoods_mc)
print("No large jumps: ")
print("Total posterior: ", posteriors_mc.sum())

