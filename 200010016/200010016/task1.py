"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
def binary_search_solver(pa, a, b, ua, t, C):
    rhs = (math.log(t) + C*math.log(math.log(t))) / ua
    mid = (a+b)/2
    while abs(a-b) >= 1e-4:
        mid = (a+b)/2
        lhs = (pa+1e-9)*math.log((1e-9+pa)/(1e-9+mid)) + (1-pa+1e-9)*math.log((1-pa+1e-9)/(1-mid+1e-9))
        if abs(lhs-rhs) <= 1e-4:
            return mid
        elif lhs - rhs < 0:
            a = mid
        else:
            b = mid
    return mid
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)
        self.ucb = np.zeros(self.num_arms)
        self.t = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # Implementing Round Robin till all the arms are pulled atleast once
        if self.t < self.num_arms:#:
            return self.t
        else:
            return np.argmax(self.ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        # Updating the counts
        self.counts[arm_index] += 1
        self.t += 1

        # Updating the emperical means
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

        # Ensuring that each arm is pulled once before we start comparing the ucb's
        if self.t < self.num_arms:
            self.ucb[arm_index] = self.values[arm_index] + np.sqrt(2*math.log(self.t)/self.counts[arm_index])
        else:
            # Updating ucb values for each arm
            for arm in range(self.num_arms): # Doing this loop as math.log faster than np.log and total number of arms is small 
                self.ucb[arm] = self.values[arm] + np.sqrt(2*math.log(self.t)/self.counts[arm])
        # END EDITING HERE



class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(self.num_arms) 
        self.values = np.zeros(self.num_arms)
        self.kl_ucb = np.zeros(self.num_arms)
        self.C = 3
        self.t = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # Implementing Round Robin till all the arms are pulled atleast once
        if self.t < self.num_arms:
            return self.t
        return np.argmax(self.kl_ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # Updating the counts
        self.counts[arm_index] += 1
        self.t += 1

        # Updating the emperical mean
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

        if self.t >= self.num_arms:
            for arm in range(self.num_arms):
                self.kl_ucb[arm] = binary_search_solver(pa=self.values[arm], a=self.values[arm], b=1-1e-9, ua=self.counts[arm], t=self.t, C=self.C)

        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)
        self.successes = np.zeros(self.num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(np.array([np.random.beta(self.successes[arm]+1,1+self.counts[arm]-self.successes[arm]) for arm in range(self.num_arms)]))
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        # Updating the counts
        self.counts[arm_index] += 1

        # Updating the emperical means
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

        self.successes[arm_index] += reward
        # END EDITING HERE

