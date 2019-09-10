
from main import bandits
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = "train file")
parser.add_argument("--seed", type = int, default = 123456789, help = "enter the value of seed")
parser.add_argument("--bandit", type=int, default = 10, help = "enter the no of bandits")
parser.add_argument("--iter", type = int, default = 100, help = "enter the no. of iterations")
parser.add_argument("--reward", type = float, default = 5, help = "enter the starting reward")
parser.add_argument("--eps", type = float, default = 0, help = "enter the value of epsilon")
parser.add_argument("--stat", action = 'store_false', help = "enter the environment variable")
parser.add_argument("--alpha", type = float, default = 0.1, help = "enter the value of alpha for weighted_aver")
parser.add_argument("--weighted", action = 'store_true', help = "enter if weighted_aver is true")
args = parser.parse_args()



def print_():
	print("Environment is stationary :", args.stat)
	print("Number of bandits are :", args.bandit)
	print( "epsilon chosen is ", args.eps)
	print("weighted_aver state", args.weighted)
	print("value of alpha for weighted_over ", args.alpha)


if __name__ == '__main__':
	print_()
	prob = bandits(no_bandit = args.bandit, eps = args.eps, no_iter = args.iter, 
							seed = args.seed, stationary = args.stat, starting_reward = args.reward
							, weighted_aver = args.weighted, alpha = args.alpha)

	reward_at_Each_step, indiv_prob_dist , optim_action_at_each_step= prob.forward()

	print("predicted mean of each arm", indiv_prob_dist)


	print("Average total rewardafter "+str(args.iter)+" steps is : ",reward_at_Each_step[-1])


	print("percentage of optimal action after "+str(args.iter)+" steps is : ", optim_action_at_each_step[-1])
	# plt.plot(reward_at_Each_step)
	plt.plot(optim_action_at_each_step)
	plt.show() 