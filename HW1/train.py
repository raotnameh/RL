
from main import bandits
import numpy as np

no_bandit = 4
no_iter = 100
eps = 0
seed = np.random.randint(1,10036)

prob = bandits(no_bandit,eps,no_iter,seed)
reward, mean = prob.forward()

print(mean)
print(reward)


