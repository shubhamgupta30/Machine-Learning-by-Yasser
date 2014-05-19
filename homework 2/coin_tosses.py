from numpy import average
from random import choice

def single_experiment(num_coins=1000, num_times=10):
  random_coin = choice(range(1000))
  num_heads = [0]*num_coins
  for coin in xrange(num_coins):
    num_heads[coin] = sum([choice([0,1]) for i in xrange(num_times)])
  return num_heads[0]/10.0, num_heads[random_coin]/10.0, min(num_heads)/10.0


def repeat_experiment(num_iterations=100000):
  (first, rand, minimum) = (0.0, 0.0, 0.0)
  for i in xrange(num_iterations):
    if i%10000 == 0: print "%d iterations done"%(i)
    (x,y,z) = single_experiment()
    first += x
    rand += y
    minimum += z
  return first/num_iterations, rand/num_iterations, minimum/num_iterations
