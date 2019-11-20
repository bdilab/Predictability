'''
    Random, Shannon, and time-correlated entropy approximation,
    and maximum predictability calculation as described in 
    K Zhao, D Khryashchev, H Vo "Predicting Taxi and Uber Demand in Cities:
    Approaching the Limit of Predictability" IEEE transactions on
    knowledge and data engineering (TKDE). IEEE, 2019.
'''


import numpy as np

def Random_Entropy(timeseries):
    """ Calculates Random entropy
    Input: 
        taxi demand series (list or 1-D array), e.g. [1, 2, 3]
    Output: 
        value of Random Entropy, e.g. 1.59
    """
    return np.log2(len(set(timeseries)))

    
def Uncorrelated_Entropy(timeseries):
    """ Computes Shannon entropy
    Input:
        timeseries: list of strings or numbers,
            e.g. ['1', '2', '3'] or [1, 2, 3]
    Output:
        value of time-uncorrelated entropy, e.g. 1.58
    """
    timeseries = np.asarray(timeseries) + abs(np.min(timeseries))
    counts = np.bincount(timeseries)
    counts = counts[counts > 0]
    if len(counts) == 1:
        return 0.
    P = 1.0 * counts / timeseries.size
    return -np.sum(P * np.log2(P))


def Real_Entropy(timeseries):
    """ Calculates an approximation of the time-correlated entropy
    Input:
        timeseries: list of strings or numbers,
            e.g. ['1', '2', '3'] or [1, 2, 3]
    Output:
        approximation of Real Entropy (time-correlated entropy), e.g. 1.09
    """

    def is_sublist(alist, asublist):
        """ Turns string lists into strings and checks if the sublist is in the list
        Input: 
            list_ : list of strings, e.g. ['1', '2', '3']
            sublist : list of strings, ['1', '2']
        Output:
            True if asublist is in alist, False otherwise
        """
        alist = "".join(map(str, alist))
        asublist = "".join(map(str, asublist))
        if asublist in alist:
            return True
        return False
    
    def shortest_subsequence(timeseries, i):
        """ Calculates length of the shortest subsequence 
            at time step i that has not appeared before
        Input: 
            timeseries: list of strings or numbers,
                e.g. ['1', '2', '3'] or [1, 2, 3]
            i: time step index, integer starting from 0
        Output:
            length of the shortest subsequence 
        """
        sequences = [timeseries[i]]  
        count = 1
        while is_sublist(timeseries[:i], sequences) and i + count <= len(timeseries) - 1:
            sequences = sequences + [timeseries[i+count]]
            count +=1
        return len(sequences)

    timeseries = list(map(str, timeseries))
    substring_length_gen = (shortest_subsequence(timeseries, i) for i in range(1, len(timeseries)))
    shortest_substring_lengths = [1] + list(map(lambda length: length, substring_length_gen))
    return np.log(len(timeseries)) * len(timeseries) / np.sum(shortest_substring_lengths)

def maximum_predictability(N, S):
    """ The Maximum Predictability function
    Input:
        N: number of unique values in time series, integer
        S: value of entropy, float
    Output:
        value of the Maximum Predictability 
        or 'No solutions' if it cannot be determined
    """
    
    def Function(x, N, S):
        """ The Maximum Predictability function
        """
        return 1.0*(-x*np.log(x)-(1-x)*np.log(1-x)+(1-x)*np.log(N-1)-S*np.log(2))

    def FirstDerivative(x, N):
        """ First derivative of the function
        """
        return 1.0*(np.log(1-x)-np.log(x)-np.log(N-1))

    def SecondDerivative(x):
        """ Second derivative of the function
        """
        return 1.0/((x-1)*x)
    
    def CalculateNewApproximation(x, N, S):
        """ One iteration of Householder's method
            of second order
        """
        function = Function(x, N, S)
        first_derivative = FirstDerivative(x, N)
        second_derivative = SecondDerivative(x)
        return 1.0*function/(first_derivative-function*second_derivative/(2*first_derivative))
 
    S = round(S, 9)
    if S > round(np.log2(N), 9):
        return "No solutions"
    else:
        if S <= 0.01:
            return 0.999
        else:
            x = (N + 1) / (2 * N)
            while abs(Function(x, N, S)) > 1e-8:
                x = x - CalculateNewApproximation(x, N, S)
    return round(x, 10)

def get_predictability(timeseries, entropy_type = 'real'):
    """ Calculates Maximum Predictability of a time series
    Input: 
        timeseries: list of strings or numbers,
            e.g. ['1', '2', '3'] or [1, 2, 3]
        entropy_type: 'real', 'random' or 'shannon'
    Output:
        value of the Maximum Predictability 
        or 'No solutions' if it cannot be determined
    """
    N = len(set(timeseries))
    if entropy_type == 'random':
        return maximum_predictability(N, Random_Entropy(timeseries))
    if entropy_type == 'shannon':
        return maximum_predictability(N, Uncorrelated_Entropy(timeseries))
    return maximum_predictability(N, Real_Entropy(timeseries))

""" Examples of usage
# Entropy
Real_Entropy([1]*5)
Real_Entropy([1]*50)
Real_Entropy(list(np.random.randint(0, 10, 10)))

Random_Entropy([1]*5)
Random_Entropy([1]*50)
Random_Entropy(list(np.random.randint(0, 10, 10)))

Uncorrelated_Entropy([1]*5)
Uncorrelated_Entropy([1]*50)
Uncorrelated_Entropy(list(np.random.randint(0, 10, 10)))

# Predictability
timeseries = list(np.random.randint(0, 10, 30))
print (timeseries)
N = len(np.unique(timeseries))
S = Real_Entropy(timeseries)
print ("N = %s, S = %s"%(N, S))
print ("Pi^max = %s"% maximum_predictability(N, S))

timeseries = [1, 2, 3]*10
print (timeseries)
N = len(np.unique(timeseries))
S = Real_Entropy(timeseries)
print ("N = %s, S = %s"%(N, S))
print ("Pi^max = %s"% get_predictability(timeseries))

 """