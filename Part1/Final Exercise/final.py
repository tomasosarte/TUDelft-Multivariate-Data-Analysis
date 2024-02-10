import scipy.io as scio
import matplotlib.pyplot as plt

# We consider that the position is a wide sense stationary process (WSS):
#
#   - The mean is constant on time
#   - The autocorrelation depends only on the time difference
#
# The alpha value oscilates between -1 and 1.
#

class RandomSignalProcessigGPS:

    def __init__(self, X: list, alpha: float, expected_value: float) -> None:
        """
        Initialize the class with the positions, alpha value and expected value.
        """
        self.positions = X
        self.alpha = alpha
        self.expected_value = expected_value

    def position_predicition(self, n: int) -> float:
        """
        Predict the position of an object on time n + 1 given his last position at time n. The prediction is based on
        this formula:
            X[n + 1] = X[n] + alpha(X[n] - X[n - 1])
        Args:
            n: Time
        Returns:    
            Float with the predicted position of the object at time n + 1
        """
        return self.positions[n] + self.alpha * (self.positions[n] - self.positions[n - 1])
    
    def estimation_error(self, estimation: float, real_value: float) -> float:
        """
        Calculate the estimation error given the estimation and the real value.
        Args:
            estimation: Estimated value
            real_value: Real value  
        Returns:
            Float with the estimation error
        """
        return real_value - estimation
    
    def variance(self, rand_signal: list) -> float:
        """
        Calculate the variance.
        Args:
            rand_dignal: List with the random signal
        Returns:
            Float with the estimation random signal
        """
        return sum([var ** 2 for var in rand_signal]) / len(rand_signal)
    
    def calculate_estimation_errors(self, init: int, end: int) -> list:
        """
        Calculate the estimation errors for the given range.
        Args:
            init: Initial time
            end: End time
        Returns:
            List with the estimation errors
        """
        estimation_errors = []
        for n in range(init, end):
            estimation = self.position_predicition(n)
            estimation_errors.append(self.estimation_error(estimation, self.positions[n + 1]))
        return estimation_errors
    
    def autocorrelation_function(self, k: int, rho: float) -> float:
        """
        Calculate the autocorrelation function for the given k with 
        the formula: Rx(k) = variance(X) * rho^abs(k). abs(rho) < 1
        Args:
            k: Time difference
            rho: rho value
        Returns:
            Float with the autocorrelation function
        """
        return self.autocorrelation(0) * rho ** abs(k)
    
    def autocorrelation(self, k: int) -> list:
        """
        Calculate autocorrelation for the given k.
        Args:
            k: Time difference
        Returns:
            List of autocorrelation values
        """
        sum = 0
        if k >= 0:   
            for i in range(len(self.positions) - k):    
                sum += (self.positions[i] - self.expected_value) * (self.positions[i + k] - self.expected_value)        
        else:
            for i in range(abs(k), len(self.positions)):    
                sum += (self.positions[i] - self.expected_value) * (self.positions[i + k] - self.expected_value)

        return (sum / (len(self.positions) - k))
    
    def optimal_alpha_value(self, rho) -> float:
        """
        Calculate the optimal alpha value given the rho value with this
        formula: alpha = (-1 + 2*rho - rho^2) / (2 - 2*rho)
        Args:
            rho: rho value
        Returns:
            Float with the optimal alpha value
        """
        return (-1 + 2*rho - rho**2) / (2 - 2*rho)
    
    def optimal_alpha_value_with_autocorrelations(self) -> float:
        """
        Calculate the optimal alpha value given the rho value with this
        formula: alpha = [- 2RX(0) + 4RX(1) - 2RX(2)]  / [4RX(0) -4RX(1)]
        Returns:
            Float with the optimal alpha value
        """
        return (-2 * self.autocorrelation(0) + 4 * self.autocorrelation(1) - 2 * self.autocorrelation(2)) / (4 * self.autocorrelation(0) - 4 * self.autocorrelation(1))
    
    def optimal_alpha_value_with_autocorrelations_rho(self, rho) -> float:
        """
        Calculate the optimal alpha value given the rho value with this
        formula: alpha = [- 2RX(0) + 4RX(1) - 2RX(2)]  / [4RX(0) -4RX(1)]
        Args:
            rho: rho value
        Returns:
            Float with the optimal alpha value
        """
        return (-2 * self.autocorrelation_function(0, rho) + 4 * self.autocorrelation_function(1, rho) - 2 * self.autocorrelation_function(2, rho)) / (4 * self.autocorrelation_function(0, rho) - 4 * self.autocorrelation_function(1, rho))
    
    def variance_mean_squared_error_alpha_function(self, alpha: float) -> float:
        """
        Calculate the mean squared error given the alpha value with the 
        fomula: (2 + 2*alpha + 2*alpha^2)RX(0) + (-2 - 4*alpha - 2*alpha^2)RX(1) + (2*alpha)RX(2)
        Args:
            alpha: Alpha value
        Returns:
            Float with the mean squared error for the estimation error
        """
        return (2 + 2*alpha + 2*alpha**2) * self.autocorrelation(0) + (-2 - 4*alpha - 2*alpha**2) * self.autocorrelation(1) + (2*alpha) * self.autocorrelation(2)
    
if "__main__" == __name__:

    # Load data
    positions = scio.loadmat('positions1.mat')['data'][0]
    N = len(positions)
    expected_value = 0

    # A: 
    alpha = 0.1
    rsp = RandomSignalProcessigGPS(positions, alpha, expected_value)
    estimation_errors = rsp.calculate_estimation_errors(1, N - 2)
    plt.plot(estimation_errors)
    plt.title('Estimation errors with alpha = 0.1')
    plt.xlabel('Time')
    plt.ylabel('Estimation error')
    plt.savefig('estimation_errors_alpha_0.1.png')

    # B:
    relation_alpha_mean_square_error = {'alpha': [], 'mean_square_error': []}
    alpha_values = [x / 100 for x in range(-100, 101)]
    for alpha_value in alpha_values:
        rsp.alpha = alpha_value
        estimation_errors = rsp.calculate_estimation_errors(1, N - 2)
        mean_square_error = rsp.variance(estimation_errors)
        relation_alpha_mean_square_error['alpha'].append(alpha_value)
        relation_alpha_mean_square_error['mean_square_error'].append(mean_square_error)
    
    plt.figure()
    plt.plot(relation_alpha_mean_square_error['alpha'], relation_alpha_mean_square_error['mean_square_error'])
    plt.title('Relation between alpha and mean square error')
    plt.xlabel('Alpha')
    plt.ylabel('Mean square error')
    plt.savefig('relation_alpha_mean_square_error.png')
    
    min_mean_square_error = min(relation_alpha_mean_square_error['mean_square_error'])
    min_mean_square_error_alpha = relation_alpha_mean_square_error['alpha'][relation_alpha_mean_square_error['mean_square_error'].index(min_mean_square_error)]

    print(f'Min variance estimation error: {min_mean_square_error}')
    print(f'Alpha: {min_mean_square_error_alpha}')

    # E: 
    alpha_optimality_given_rho = {'rho': [], 'optimal_alpha': []}
    rho_values = [x / 100 for x in range(-99, 100)]
    for rho in rho_values:
        alpha_optimality_given_rho['rho'].append(rho)
        alpha_optimality_given_rho['optimal_alpha'].append(rsp.optimal_alpha_value(rho))

    plt.figure()
    plt.plot(alpha_optimality_given_rho['rho'], alpha_optimality_given_rho['optimal_alpha'])
    plt.title('Optimal alpha value given rho')
    plt.xlabel('Rho')
    plt.ylabel('Optimal alpha')
    plt.savefig('optimal_alpha_given_rho.png')

    # F:
    autocorrelation_k1 = rsp.autocorrelation(1)
    variance = rsp.autocorrelation(0)
    rho = autocorrelation_k1 / variance
    print(f'Autocorrelation with k = 1: {autocorrelation_k1}')
    print(f'Variance: {variance}')
    print(f'Rho: {rho}')
    print(f'Optimal alpha: {rsp.optimal_alpha_value(rho)}')
    

    # G:
    autocorrelation_function = {'k': [], 'autocorrelation_function': [], 'autocorrelation': []}
    k_values = [x for x in range(-100, 101)]
    for k in k_values:
        autocorrelation_function['k'].append(k)
        autocorrelation_function['autocorrelation_function'].append(rsp.autocorrelation_function(k, rho))
        autocorrelation_function['autocorrelation'].append(rsp.autocorrelation(k))

    plt.figure()
    plt.plot(autocorrelation_function['k'], autocorrelation_function['autocorrelation_function'])
    plt.title('Autocorrelation function')
    plt.xlabel('k')
    plt.ylabel('Autocorrelation function')
    plt.savefig('autocorrelation_function.png')

    plt.figure()
    plt.plot(autocorrelation_function['k'], autocorrelation_function['autocorrelation'])
    plt.title('Autocorrelation')
    plt.xlabel('k')
    plt.ylabel('Autocorrelation')
    plt.savefig('autocorrelation.png')
    
    # H: variance estimation error in function of alpha
    variance_estimation_error = {'alpha': [], 'variance_estimation error': []}
    for alpha in alpha_values:
        variance_estimation_error['alpha'].append(alpha)
        variance_estimation_error['variance_estimation error'].append(rsp.variance_mean_squared_error_alpha_function(alpha))

    plt.figure()
    plt.plot(variance_estimation_error['alpha'], variance_estimation_error['variance_estimation error'])
    plt.title('Variance estimation error in function of alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Variance estimation error')
    plt.savefig('variance_estimation_error_alpha.png')

    # I: Find the optimal alpha value with autocorrelations
    print(f'Optimal alpha value with autocorrelations: {rsp.optimal_alpha_value_with_autocorrelations()}')
    print(f'Optimal alpha value with autocorrelations rho: {rsp.optimal_alpha_value_with_autocorrelations_rho(rho)}')

    