import numpy as np
import matplotlib.pyplot as plt
import sys

class bandits():
	def __init__(self,no_bandit = 4, eps = 0, no_iter = 1000, 
						seed = 123456, stationary = True, starting_reward = 5
						, weighted_aver = False, alpha = 0.1):
        
		np.random.seed(seed)

		self.no_bandit = no_bandit
		self.eps = eps
		self.no_iter = no_iter
		self.weighted_aver = weighted_aver
		self.alpha = alpha

		self.total_reward = 0
		self.individual_mean = np.zeros(self.no_bandit) + starting_reward
		self.steps = 0
		self.individual_steps = np.zeros(self.no_bandit)
		self.reward_after_each_step = []
		self.stationary = stationary
         


# Prob/Mean of winning "1" for each bandit
	def individual_prob(self,sigma=0):
		return np.random.uniform(sigma,1,size = self.no_bandit)

 # True Reward at a particular time step or "1" and "0" at each time step or Distribution dependent reward
	def distri_depen_rewa(self,prob):
		out = []
		for i in prob:
			out.append(np.random.normal(i,1))

		return out # list of rewards for each arm/bamdit at a particular time step

# Which action to take
	def action(self):
		prob = np.random.rand()
		if self.steps == 0 :
		    return np.random.choice(self.no_bandit)

		elif prob > self.eps or self.eps == 0:
			# print("greedy")
			return np.argmax(self.individual_mean)

		elif prob < self.eps:
			# print("epsilon-greddy")
			return np.random.choice(self.no_bandit)

		else: raise NotImplementedError

# Forward function to implement the logic
	def forward(self):
		optimal_action = 0
		optimal_action_total = []
		prob = self.individual_prob()
		print("True mean of each arm", prob)
		for i in range(self.no_iter):

			action = self.action()
			reward = self.distri_depen_rewa(prob)
			reward_a = reward[action]

			self.steps += 1
			self.individual_steps[action] +=1 

			self.reward_after_each_step.append((self.total_reward))

			if self.weighted_aver == True:
				self.total_reward += (reward_a - (self.total_reward))*(self.alpha)
				self.individual_mean[action] += (reward_a - self.individual_mean[action])/(self.individual_steps[action])

			else: 
				self.total_reward += (reward_a - (self.total_reward))/(self.steps)
				self.individual_mean[action] += (reward_a - self.individual_mean[action])/(self.individual_steps[action])

			if reward_a >= max(reward):
				# print("optimal_action")
				optimal_action +=1
		    
			optimal_action_total.append(optimal_action/self.steps)
		    

			if self.stationary == False:
				prob = self.individual_prob()

		return self.reward_after_each_step, self.individual_mean, optimal_action_total
        
