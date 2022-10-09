"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import numpy as np
from task1 import Thompson_Sampling

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.zeta = int(np.sqrt(self.num_arms))
        self.curr_arms = np.random.choice([i for i in range(self.num_arms)],self.zeta)
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)
        self.successes = np.zeros(self.num_arms)


        # Horizon is same as number of arms
    
    def give_pull(self):
        # START EDITING HERE  
        return self.curr_arms[np.argmax(np.array([np.random.beta(self.successes[arm]+1,1+self.counts[arm]-self.successes[arm]) for arm in self.curr_arms]))]
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1

        # Updating the emperical means
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

        self.successes[arm_index] += reward        
        # END EDITING HERE
