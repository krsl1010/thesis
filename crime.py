import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd 
from tqdm import tqdm 
from numba import njit, objmode 
from math import comb 
from collections import defaultdict
import matplotlib.pyplot as plt 


total_posterior = 29723043.900832776

def log_lik(inclusion, data, g=47, n=47, T=1):
    model = LinearRegression()
    
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    X = X.to_numpy()
    y = y.to_numpy()
    
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

def renormalized_model(models, p=15, T=1):
    n_models = len(models)
    likelihoods = np.zeros(n_models)
     
    for i in range(n_models):
        likelihoods[i] = log_lik(models[i], df, T=T)
    
    likelihoods *= T 
    posteriors = np.exp(likelihoods - likelihoods.max())
    #posteriors *= T
    posteriors /= posteriors.sum()
    pi_j = np.zeros(p)

    for j in range(p):
        pi_j[j] = np.sum(posteriors[models[:, j] == 1])
        #print(posteriors[models[:, j]])
    return pi_j 


df = pd.read_csv('data/crime.csv', header=None)

def test_full_model():
    incl = np.ones(15, dtype=int)
    print(log_lik(incl, df))

test_full_model()
def test_full_space():
    full_model_space = np.array([list(format(i, 'b').zfill(15)) for i in range(2**15)], dtype=int)
    #print(np.sort(renormalized_model(full_model_space)))
    return renormalized_model(full_model_space)
    

"""
def local_optim(model, s, data, T=1):
    all_indices = np.arange(15)
    #keep the large jump indices s unchaged 
    rem_idx = np.setdiff1d(all_indices, s)
    #print(s, rem_idx)
    best_lik = log_lik(model, data)
    best_model = model.copy()
    cur = model.copy()
    for i in range(0, len(rem_idx)):
        for j in range(i, len(rem_idx)):
            #print(rem_idx[i], rem_idx[j])
            cur[rem_idx[i]] = 1 - cur[rem_idx[i]]
            cur[rem_idx[j]] = 1 - cur[rem_idx[j]]
            cur_lik = log_lik(cur, data, T=T)
            if cur_lik > best_lik:
                best_lik = cur_lik  
                best_model = cur.copy()

    return best_model
"""

def local_optim(model, s, data, T=1):
    all_indices = np.arange(15)  # Updated to 11 covariates
    # Keep the large jump indices s unchanged 
    rem_idx = np.setdiff1d(all_indices, s)
    best_lik = log_lik(model, data, T=T)  # Fixed T usage
    best_model = model.copy()
    
    while True:
        cur = best_model.copy()  # Start from best_model each iteration
        improved = False
        
        for i in range(0, len(rem_idx)):
            for j in range(i + 1, len(rem_idx)):  # Exclude i == j
                # Create fresh copy for each pair
                cur = best_model.copy()
                # Flip two covariates
                cur[rem_idx[i]] = 1 - cur[rem_idx[i]]
                cur[rem_idx[j]] = 1 - cur[rem_idx[j]]
                cur_lik = log_lik(cur, data, T=T)
                if cur_lik > best_lik:
                    best_lik = cur_lik
                    best_model = cur.copy()
                    improved = True
        
        # If no improvement, stop
        if not improved:
            break
    
    return best_model

def model_prob(proposal, current):
    pass

def randomization(model, p_swap=0.1):
    new_model = model.copy()
    for j in range(len(model)):
        if np.random.rand() < p_swap:
            new_model[j] = 1 - model[j]
    return new_model 


def mjmcmc(num_iterations, data, p, large_prob=0.05, T=1):
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
            #print(np.where(proposed != current)[0])
            # Perform local optimization keeping the large swap indices unchanged 
            pre_opt_lik = log_lik(proposed, data)
            proposed = local_optim(proposed, s, data, T=T)
            post_opt_lik = log_lik(proposed, data)
            #print(f"Local opt change: {post_opt_lik - pre_opt_lik}")
            chi_k_star = proposed.copy() 

            # Perform randomization
            proposed = randomization(proposed)

            # Backwards large jump swapping the components swapped in the large jump  
            reverse = proposed.copy()
            reverse[s] = 1 - reverse[s]

            # Backwards local optimimization 
            reverse = local_optim(reverse, s, data, T=T)
            
            # Calculate likelihoods 
            prop_lik = log_lik(proposed, data, T=T)
            current_lik = log_lik(current, data, T=T)

            # Calculate model probabilities
            swaps_cur = np.where(reverse != current)[0]
            #k = len(swaps_cur)
            current_prob = np.log(0.1**len(swaps_cur) /  comb(p, len(swaps_cur)))
            #current_prob = k * np.log(0.1) + (p - k) * np.log(0.9) + np.log(comb(p, k))

            swaps_prop = np.where(proposed != chi_k_star)[0]
            #k = len(swaps_prop)
            prop_prob = np.log(0.1**len(swaps_prop) / comb(p, len(swaps_prop)))
            #prop_prob = k * np.log(0.1) + (p - k) * np.log(0.9) + np.log(comb(p, k))
        else:
            # Perform swap 
            s = np.random.choice(p, 1, replace=False)
            proposed = current.copy()
            proposed[s] = 1 - proposed[s]
            
            # Calculate likelihoods 
            prop_lik = log_lik(proposed, data, T=T)
            current_lik = log_lik(current, data, T=T)
            
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
             
        posteriors[i] = log_lik(current, data)
    print(f"Acceptance rate: {count/num_iterations:.2f}")
    unique_vals, counts = np.unique(sample, axis=0, return_counts=True)
    print(f"Unique models visited: {len(unique_vals)}")
    return sample

#sample = mjmcmc(30000, df, 15)

#truth = test_full_space()
truth = np.array([0.34202705, 0.56770511, 0.39102621, 0.30456034, 0.22654125, 0.33104482, 
                  0.28621796, 0.1612566, 0.23391989, 0.76960068, 0.58884742, 0.21559302,
                  0.16313296, 0.19439907, 0.8178239])
    
def hundred_runs(n_runs=100, n_iter=30000, T=1):
     
    estimates = np.zeros((n_runs, 15))
    estimates_mc = np.zeros((n_runs, 15))
    posterior_masses = np.zeros(n_runs)
    models_visited = np.zeros(n_runs)

    for i in tqdm(range(n_runs)):
        sample = mjmcmc(n_iter, df, 15, large_prob=0.05, T=T)
        unique_samples = np.unique(sample, axis=0)
        models_visited[i] = len(unique_samples)
        pi_j = renormalized_model(unique_samples, p=15, T=1)
        print(pi_j)
        estimates[i] = pi_j
            
        likelihoods = np.zeros(len(unique_samples))
        for j in range(len(unique_samples)):
            likelihoods[j] = log_lik(unique_samples[j], df, T=1)
         
        log_posteriors = likelihoods
        posteriors = np.exp(log_posteriors)
        print("Total posterior: ", posteriors.sum())
        posterior_masses[i] = posteriors.sum()
        #posteriors /= posteriors.sum()
        
        print(np.sum(sample, axis=0)/n_iter)
        estimates_mc[i] = np.sum(sample, axis=0)
        estimates_mc[i] /= n_iter
        """
        #plt.scatter(truth, estimates)
        #plt.plot(truth, truth)
        #plt.title(f"Temperature {T}")
        ##plt.show()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot([np.sum(state) for state in sample])
        plt.xlabel("Iteration")
        plt.ylabel("Number of included covariates")
        plt.title("Trace plot: Inclusions: ")
        plt.subplot(1, 2, 2)
        plt.plot([log_lik(state, df, T=1) for state in sample])
        plt.xlabel("Iteration")
        plt.ylabel("Log-likelihood")
        plt.title("Trace plot: Log-likelihood")
        plt.tight_layout()
        plt.show()
        """
    squared_diffs = (estimates[:,] - truth)**2
    mse = np.mean(squared_diffs, axis=0)
    rmse = np.sqrt(mse)
     
    squared_diffs_mc = (estimates_mc[:,] - truth)**2
    mse_mc = np.mean(squared_diffs_mc, axis=0)
    rmse_mc = np.sqrt(mse_mc)
    
    print("Temperature: ", T)
    print("pi_j | RM | MC")
    for i in range(len(rmse)):
        print(f"pi_{i+1} | {rmse[i]*100:.2f} | {rmse_mc[i]*100:.2f}")
    
    avg_posterior_mass = np.mean(posterior_masses) / total_posterior
    avg_models_visited = np.mean(models_visited)
    print(f"Average number of unique models visited: {avg_models_visited:.2f}")
    print(f"Average posterior mass: {avg_posterior_mass:.2f}")
    np.savetxt(f"rmse_{T}.txt", rmse*100)

if __name__ == "__main__":
    print("Enter number of runs: ")
    n_runs = int(input())
    hundred_runs(n_runs, n_iter=3276, T=2)
