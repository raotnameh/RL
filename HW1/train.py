
from main import bandits
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = "train file")
parser.add_argument("--seed", type = int, default = 123456789, help = "enter the value of seed")
parser.add_argument("--bandit", type=int, default = 1, help = "enter the no of bandits")
parser.add_argument("--iter", type = int, default = 1000, help = "enter the no. of iterations")
parser.add_argument("--eps", type = float, default = 0.1, help = "enter the value of epsilon")
parser.add_argument("--stationary", action = 'store_false', help = "enter the environment variable")
parser.add_argument("--random", action='store_true',help='choose actions randomly or no')
args = parser.parse_args()



print("Environment is :", args.stationary)
print("Number of bandits are :", args.bandit)
print( "epsilon chosen is ", args.eps)


prob = bandits(no_bandit = args.bandit, eps = args.eps, no_iter = args.iter, 
						seed = args.seed, stationary = args.stationary, random = False)

reward, mean = prob.forward()


print("Average total reward ",reward[-1])