'''
Created on 8 Mar 2023

@author: ucacsjj
'''

from monte_carlo.episode_sampler import EpisodeSampler

from .td_algorithm_base import TDAlgorithmBase

class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment):
        
        TDAlgorithmBase.__init__(self, environment)
        
        self._minibatch_buffer= [None]
                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            
    def _update_value_function_from_episode(self, episode):

        # Q1e:
        # Complete implementation of this method
        # Each time you update the state value function, you will need to make a
        # call of the form:
        #
        # self._v.set_value(x_cell_coord, y_cell_coord, new_v)

        num_steps = episode.number_of_steps()

        for t in range(num_steps):

            # Current state S_t
            state_t = episode.state(t)
            coords_t = state_t.coords()
            x_t, y_t = coords_t[0], coords_t[1]

            # Immediate reward from taking action at S_t
            reward_tp1 = episode.reward(t)

            # Current estimate V(S_t)
            v_t = self._v.value(x_t, y_t)

            # If this is the last stored transition, the next state was terminal
            if t == num_steps - 1:
                td_target = reward_tp1
            else:
                # Next state S_{t+1}
                state_tp1 = episode.state(t + 1)
                coords_tp1 = state_tp1.coords()
                x_tp1, y_tp1 = coords_tp1[0], coords_tp1[1]

                v_tp1 = self._v.value(x_tp1, y_tp1)
                td_target = reward_tp1 + self._gamma * v_tp1

            # TD(0) update
            new_v = v_t + self._alpha * (td_target - v_t)

            self._v.set_value(x_t, y_t, new_v)

