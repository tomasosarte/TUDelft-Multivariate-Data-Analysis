import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt

def posterior(t, y):

    # Priors
    p_mean = np.zeros(3).reshape(3,1)
    p_cov = 100*np.eye(3)

    # Init vectors
    post_means = [p_mean]
    post_covs = [p_cov]
    
    # Random noise variance
    var = 1

    # First iteration
    h_0 = np.array([1, t[0], np.sin(2*np.pi*t[0])]).reshape(1, 3)
    # post_covs.append(np.linalg.inv(h_0.T * var * h_0 + np.linalg.inv(post_covs[0])))
    new = post_covs[0] - post_covs[0] @ h_0.T * 1/(h_0 @ post_covs[0] @ h_0.T + var) @ h_0 @ post_covs[0]
    post_covs.append(new)
    new = post_covs[-1] @ (h_0.T * var * y[0] + np.linalg.inv(post_covs[0]) @ post_means[0])
    post_means.append(new)


    for k in range(1, len(t)):

        # Get H_k
        h_k = np.array([1,t[k],np.sin(2*np.pi*t[k])]).reshape(1, 3)

        # Get posterior covariance
        # post_covs.append(np.linalg.inv(h_k.T * var * h_k + np.linalg.inv(post_covs[k-1])))
        new = post_covs[k-1] - post_covs[k-1] @ h_k.T * 1/(h_k @ post_covs[k-1] @ h_k.T + var) @ h_k @ post_covs[k-1]
        post_covs.append(new)

        # Get posterior mean
        new = post_covs[-1] @ (h_k.T * 1 * y[k] + np.linalg.inv(post_covs[k-1]) @ post_means[k-1])
        post_means.append(new)
    
    return post_means, post_covs

def predictions(t, theta):
    # Generate predictions with theta parameters
    return theta[0] + theta[1] * t + theta[2] * np.sin(2 * np.pi * t)

if __name__ == "__main__":

    # Get data
    dataset = pd.read_csv("periodic.csv")
    t = dataset["t"].to_numpy()
    y = dataset["y"].to_numpy()

    post_means, post_covs = posterior(t, y)
    print("---------- Posterior mean with all data ------------")
    print(post_means[-1])
    print("--------------- Posterior covariance with all data -------------------")
    print(post_covs[-1])
    print("-------------- Posterior mean with 5 datapoints ------------------")
    print(post_means[5])
    print("-------------- Posterior covariance with 5 datapoints ----------------")
    print(post_covs[5])

    # Generate prediction for a linspace
    t_linspace = np.linspace(min(t), max(t), 1000)
    pred = predictions(t_linspace, post_means[-1].flatten())

    # Plot observations
    plt.scatter(t, y, label="Observations", color='red')

    # Plot predictions
    plt.plot(t_linspace, pred, label="Predictions")

    plt.title('Fitted Curve with All Observations')
    plt.ylabel('y')
    plt.xlabel('t')
    plt.legend()
    plt.show()