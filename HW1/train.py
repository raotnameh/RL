
from main import bandits
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = "train file")
parser.add_argument("--seed", type = int, default = 123456789, help = "enter the value of seed")
parser.add_argument("--bandit", type=int, default = 10, help = "enter the no of bandits")
parser.add_argument("--iter", type = int, default = 1000, help = "enter the no. of iterations")
parser.add_argument("--reward_0", type = float, default = 0, help = "enter the starting reward")
parser.add_argument("--eps", type = float, default = 0.1, help = "enter the value of epsilon")
parser.add_argument("--stat", action = 'store_false', help = "enter the environment variable")
parser.add_argument("--alpha", type = float, default = 0.1, help = "enter the value of alpha for weighted_aver")
parser.add_argument("--weighted_aver", action = 'store_true', help = "enter if weighted_aver is true")
parser.add_argument("--random", action='store_true',help='choose actions randomly or no')
args = parser.parse_args()



def print_():
	print("Environment is stationary :", args.stat)
	print("Number of bandits are :", args.bandit)
	print( "epsilon chosen is ", args.eps)
	print("weighted_aver state", args.weighted_aver)
	print("value of alpha for weighted_aver ", args.alpha)


if __name__ == '__main__':
	print_()
	prob = bandits(no_bandit = args.bandit, eps = args.eps, no_iter = args.iter, 
							seed = args.seed, stationary = args.stat, random = False, starting_reward = args.reward_0
							, weighted_aver = args.weighted_aver, alpha = args.alpha)

	reward, mean, optim_action = prob.forward()


	print("Average total reward ",reward)