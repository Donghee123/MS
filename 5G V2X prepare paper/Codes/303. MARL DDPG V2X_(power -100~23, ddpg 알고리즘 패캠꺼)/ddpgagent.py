# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 23:04:11 2021

@author: DH
"""

class Agent(object):
    def __init__(self, DDPGAgent):
        self.DDPGAgent = DDPGAgent
        self.memory = DDPGAgent.memory
        self.select_step = 0
        
    def select_action(self, s_t, decay_epsilon=True):
        self.select_step += 1
        return self.DDPGAgent.select_action(s_t, decay_epsilon=True)
    
    def random_action(self):
        self.select_step += 1
        return self.DDPGAgent.random_action()
    
    def load_model(self, actor_path, critic_path):
        return self.DDPGAgent.load_weights(actor_path, critic_path)
    
    def observe(self, r_t, s_t):
        self.DDPGAgent.observe(r_t, s_t, False)
    
    def update_policy(self, isHardupdate = False):
        return self.DDPGAgent.update_policy(isHardupdate)
        
    def reset_state(self, s_t):
        self.DDPGAgent.reset_state(s_t)
    
    def reset(self):
        self.DDPGAgent.reset_random_process()
        
    def save_model(self, model_path, saveFileName):
    #def 
        self.DDPGAgent.save_model(model_path, saveFileName)
