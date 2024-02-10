import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma

def plot_theta_evolution(thetas):
    #Plot theta evolution
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('Newton\'s method', fontsize=16)
    axs[0,0].plot(range(thetas.shape[0]), thetas[:, 0, 0], color='black')
    axs[0,1].plot(range(thetas.shape[0]), thetas[:, 1, 0], color='black')
    axs[1,0].plot(range(thetas.shape[0]), thetas[:, 2, 0], color='black')
    axs[1,1].plot(range(thetas.shape[0]), thetas[:, 3, 0], color='black')
    axs[0,0].set_title("theta 1")
    axs[0,1].set_title("theta 2")
    axs[1,0].set_title("theta 3")
    axs[1,1].set_title("theta 4")
    plt.show()

def Laplace_approximation(theta, std, X):
    mean = theta
    cov = - np.linalg.inv(MAP_Hessian(theta, std, X))
    return mean, cov

def random_walk_Metropolis_Hastings(\
        theta, sigma=1.0, n_iter = 1000, burn_in = 100, prior_std = 1.0):
    print("Running random walk Metropolis-Hastings algorithm...")

    # Initialize
    theta = theta.flatten()
    var = prior_std**2
    samples = [theta]
    num_accepteds_proposals = 0
    for _ in range(1, n_iter):
        
        # Sample from proposal distribution
        theta_prime = theta + sigma * np.random.normal(0, prior_std, theta.shape[0])
        
        # Compute prior of theta
        prior_theta = np.exp(-np.dot(theta.T, theta) / (2*var**2))
        prior_theta_prime = np.exp(-np.dot(theta_prime.T, theta_prime) / (2*var**2))

        ratio = (prior_theta_prime) / (prior_theta)

        # Accept or reject
        if np.random.uniform() <= ratio:
            theta = theta_prime
            num_accepteds_proposals += 1
        
        # Save sample
        samples.append(theta)

    # Compute posterior mean
    samples = np.array(samples)
    post_mean = np.mean(samples[burn_in:, :], axis=0)
    post_cov = np.cov(samples[burn_in:, :].T)

    # Important data
    print("Acceptance rate: ", num_accepteds_proposals / n_iter)
    print("Posterior mean: \n", post_mean)
    print("Posterior covariance: \n", post_cov)

    # Plot
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('Random walk Metropolis-Hastings algorithm', fontsize=16)
    axs[0].scatter(samples[burn_in:, 0], \
                   samples[burn_in:, 1], c=range(n_iter - burn_in))
    axs[1].scatter(samples[:, 0], samples[:, 1], c=range(n_iter))
    axs[0].set_title("Burn-in samples")
    axs[1].set_title("All samples")
    plt.show()

    # Plot all thetas in diff plots in function of iteration
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('Theta convergence', fontsize=16)
    axs[0,0].plot(range(n_iter), samples[:, 0], color='black')
    axs[0,1].plot(range(n_iter), samples[:, 1], color='black')
    axs[1,0].plot(range(n_iter), samples[:, 2], color='black')
    axs[1,1].plot(range(n_iter), samples[:, 3], color='black')
    axs[0,0].set_title("theta 1")
    axs[0,1].set_title("theta 2")
    axs[1,0].set_title("theta 3")
    axs[1,1].set_title("theta 4")
    plt.show()

def inverse_gamma_pdf(x, A, B, k, theta):
    if x <= 0:
        return 0
    else:
        coefficient = np.power(x, -(A + k / 2) - 1)
        exponent = np.exp(-((B + k * np.power(theta, 2)) / (2 * x)))
        normalization = np.power(2 * np.pi, -0.5)
        return coefficient * exponent * normalization
    
def gibbs_sampling(theta, sigma=1.0, \
                   alpha=0.2, beta=0.2, n_iter = 1000, burn_in = 100):

    print("Running Gibbs sampling algorithm...")
    # Initialize
    theta = theta.flatten()
    samples = [theta]
    sigmas = [sigma]
    k = theta.shape[0]
    for _ in range(1, n_iter):
        
        # Sample theta
        theta = np.random.multivariate_normal(\
            np.zeros(theta.shape[0]), sigma**2 * np.eye(theta.shape[0]))
        samples.append(theta)

        # Sample sigma^2
        sigma = np.sqrt(invgamma.rvs(\
            alpha + k/2, scale=beta + np.dot(theta.T, theta)/2))
        sigmas.append(sigma)

    # Compute posterior mean
    samples = np.array(samples)
    post_mean = np.mean(samples[burn_in:, :], axis=0)
    post_cov = np.cov(samples[burn_in:, :].T)

    # Important data
    print("Posterior mean: \n", post_mean)
    print("Posterior covariance: \n", post_cov)

    # Plot
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('Gibbs sampling algorithm', fontsize=16)
    axs[0].scatter(samples[burn_in:, 0], \
                   samples[burn_in:, 1], c=range(n_iter - burn_in))
    axs[1].scatter(samples[:, 0], samples[:, 1], c=range(n_iter))
    axs[0].set_title("Burn-in samples")
    axs[1].set_title("All samples")
    plt.show()

    # Plot all thetas in diff plots in function of iteration
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('Theta convergence', fontsize=16)
    axs[0,0].plot(range(n_iter), samples[:, 0], color='black')
    axs[0,1].plot(range(n_iter), samples[:, 1], color='black')
    axs[1,0].plot(range(n_iter), samples[:, 2], color='black')
    axs[1,1].plot(range(n_iter), samples[:, 3], color='black')
    axs[0,0].set_title("theta 1")
    axs[0,1].set_title("theta 2")
    axs[1,0].set_title("theta 3")
    axs[1,1].set_title("theta 4")
    plt.show()

    # Plot sigma^2
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('Sigma^2 convergence', fontsize=16)
    axs.plot(range(n_iter), sigmas, color='black')
    plt.show() 

def MAP_gradient(theta, std, X, Y):
    mu = np.exp(np.dot(X, theta))
    return - np.dot(X.T, mu) + np.dot(X.T, Y) - (1/std**2) * theta

def MAP_Hessian(theta, std, X):
    mu = np.exp(np.dot(X, theta))
    W = np.diag(mu.flatten())
    return - np.dot(X.T, W).dot(X) -(1/std**2) * np.eye(theta.shape[0])

def newton(X, Y, theta, std=1.0, max_iter=100, tol=1e-4): 

    print("Running Newton's method...")

    # Initialize
    thetas = [theta]

    # Run Newton's method
    i = 0
    while i < max_iter:
        
        # Update theta with Newton's method
        map_gradient = MAP_gradient(thetas[-1], std, X, Y)
        map_hessian = MAP_Hessian(thetas[-1], std, X)
        theta_prime = thetas[-1] - np.linalg.inv(map_hessian).dot(map_gradient)

        # Save theta
        thetas.append(theta_prime)

        # Check stopping criterion
        if np.linalg.norm(theta_prime - thetas[-2]) < tol:
            print("Converged after {} iterations".format(i+1))
            # plot_theta_evolution(np.array(thetas))
            return theta_prime
        
        # Update
        i += 1
    
    print("Did not converge after {} iterations".format(i))
    # plot_theta_evolution(np.array(thetas))
    return thetas[-1]

if __name__ == "__main__":

    # Get data
    dataset = pd.read_csv("dataexercise2.csv")
    x1 = dataset["x1"].to_numpy()
    x2 = dataset["x2"].to_numpy()
    x3 = dataset["x3"].to_numpy()
    x4 = dataset["x4"].to_numpy()
    Y = dataset["y"].to_numpy().reshape(-1, 1).astype(float)

    # Transform data
    X = np.array([x1, x2, x3, x4]).T

    # Initialize theta
    theta = np.zeros((4, 1)).astype(float)

    # Run Newton's method for laplace approximation
    theta = newton(X, Y, theta, max_iter=100, std=4.0)
    mean, covariance = Laplace_approximation(theta, 4.0, X)

    # Print results
    print("theta: \n ", theta)
    print("mean: \n ", mean)
    print("covariance: \n", covariance)
    print("\n")

    # random walk Metropolis-Hastings
    random_walk_Metropolis_Hastings(theta, sigma=4.0, n_iter=10000, burn_in=5000, prior_std=4.0)
    print("\n")

    # Gibbs sampling
    gibbs_sampling(theta, sigma=1.0, alpha=0.2, beta=0.2, n_iter=10000, burn_in=5000)