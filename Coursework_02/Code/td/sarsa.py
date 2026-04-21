'''
Created on 8 Mar 2023

@author: ucacsjj
'''

import numpy as np

from .td_controller import TDController

# Simplified version of the predictor from S+B

class SARSA(TDController):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDController.__init__(self, environment)

    def initialize(self):
        
        TDController.initialize(self)
        
        self._v.set_name("SARSA Expected Value Function")
        self._pi.set_name("SARSA Greedy Policy")
                    
    def _update_action_and_value_functions_from_episode(self, episode):
        
        # This calls a method in the TDController which will update the
        # Q value estimate in the base class and will update
        # the greedy policy and estimated state value function
        
        # Handle everything up to the last state transition to the terminal state
        s = episode.state(0)
        coords = s.coords()
        reward = episode.reward(0)
        a = episode.action(0)
        
        for step_count in range(1, episode.number_of_steps()):

            # Get the next state in the episode
            s_prime = episode.state(step_count)
            coords_prime = s_prime.coords()

            # Get the next action A' from the episode 
            a_prime = episode.action(step_count)

            # Get current Q(S, A)
            current_q = self._Q[coords[0], coords[1], a]

            # Updat SARSA
            new_q = current_q + self._alpha * (reward + self._gamma * self._Q[coords_prime[0], coords_prime[1], a_prime] - current_q)
                    
            # Update the grid
            self._update_q_and_policy(coords, a, new_q)

            # Move to the next step in the episode
            reward = episode.reward(step_count)
            s = s_prime
            coords = coords_prime
            a = episode.action(step_count)

        # Final value
        new_q = reward
        self._update_q_and_policy(coords, a, new_q)

