import numpy as np
import matplotlib.pyplot as plt
import sys

class bandits():
	def __init__(self,no_bandit = 4, eps = 1, no_iter = 1000, 
						seed = 123456, stationary = True, random = False ):
        
		np.random.seed(seed)

		self.no_bandit = no_bandit
		self.eps = eps
		self.no_iter = no_iter
		# self.initial_reward = initial_reward
		self.random = random

		self.total_reward = 0
		self.individual_mean = np.zeros(self.no_bandit)
		self.steps = 0
		self.individual_steps = np.zeros(self.no_bandit)
		self.reward_after_each_step = []
		self.stationary = stationary
         


# Prob/Mean of winning "1" for each bandit
	def individual_prob(self):
		return np.random.uniform(0.2,.8,size = self.no_bandit)

 # True Reward at a particular time step or "1" and "0" at each time step or Distribution dependent reward
	def distri_depen_rewa(self,prob):
		out = []
		for i in prob:
		    out.append(np.random.choice([1,0],p=[abs(i),abs(1-i)]))

		return out # list of rewards for each arm/bamdit at a particular time step

# Which action to take
	def action(self):
		prob = np.random.rand()
		if self.steps == 0 or self.random == True:
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
	    
		if self.stationary == True:
			prob = self.individual_prob()
			# print("True probability distribution for the stationary case",prob)

			for i in range(self.no_iter):

				action = self.action()
				reward = self.distri_depen_rewa(prob)[action]
				# print(action,reward)

				# if i == 1:
					# print("first action taken is ", action)

				self.steps += 1
				self.individual_steps[action] += 1
				
				self.total_reward += (reward )
				# print(self.individual_mean[action],self.individual_mean)

				self.individual_mean[action] += (reward - self.individual_mean[action])/(self.individual_steps[action])

				# print(self.individual_mean[action],self.individual_mean)
				# sys.exit()
				

				self.reward_after_each_step.append(int(self.total_reward)/self.steps)
			    
		        
			return self.reward_after_each_step, self.individual_mean

		else:

			
			for i in range(self.no_iter):

				prob = self.individual_prob()
				action = self.action()
				reward = self.distri_depen_rewa(prob)[action]

				self.steps += 1
				self.individual_steps[action] += 1
				
				self.total_reward += (reward )
				# print(self.individual_mean[action],self.individual_mean)

				self.individual_mean[action] += (reward - self.individual_mean[action])/(self.individual_steps[action])

				# print(self.individual_mean[action],self.individual_mean)
				# sys.exit()
				

				self.reward_after_each_step.append(int(self.total_reward)/self.steps)
			    
		        
			return self.reward_after_each_step, self.individual_mean
			
		    
