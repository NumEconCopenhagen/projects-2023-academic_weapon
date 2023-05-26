import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import itertools


#Define functions 

#Cost 

def cost_dentist(par, d, x_t):
    "Cost of going to the dentist"
    if d == 0: 
        return  0 #Not going is always free
    if d == 1: 
        return par.FC + par.MC*x_t # Cost of going depends on the fixed cost, marginal cost and current level of usage of ones teeth 
    

#Decay


def teeth_decay(par, d, x_t, age, extension = False): #How your teeth are transitioning from one time period to the next
    if extension == True: #Introducing age dependence
        if d == 0: #Not going 
            x_1 = (age/4)*x_t**par.exponent + par.natural_punishment + age  
        else:
            x_1 = max(x_t - par.boost + age, age) #boost is the effect of going to the dentist
    else: 
        if d == 0:
            x_1 = x_t**par.exponent + par.natural_punishment  #Natural punishment avoids that the teeth decay is too small is zero for low values of x
        else:
            x_1 = max(x_t - par.boost, 0) #Implies that one is going to the dentist
        
    return x_1


#Pain

def disutil_ache(par, x_t): 
    disutil = np.sqrt(x_t) * par.gamma + x_t #Disutility of toothache increases with current level of usage of your teeth and some constant (gamma)
    return disutil


#Total utility

def total_util(par, d, x_t): #The function calculates the total utility by subtracting the cost of the dentist visit and the disutility associated with toothache.
    return -cost_dentist(par, d, x_t) - disutil_ache(par, x_t)


#Possible paths of the decision tree with one binary decision for the entire lifetime. 

def n_combinations(t):
    digits = (0, 1)
    return list(itertools.product(digits, repeat=t-1))



def solve_dentist(par, extension = False):
    possible_paths = n_combinations(par.life_span) #Create all decision combinations, excluding the last time period (d==0)
    n_paths = len(possible_paths) #Game starts in t1 with a t0 value. 
    x = np.zeros((n_paths, par.life_span)) #decay_path_matrix
    x[:,0] = par.x_start #Change the first column, hence the starting value
    v = np.zeros((n_paths,par.life_span+1))#value_matrix
    
    age = 0
    #Loop through all paths
    for index, val in enumerate(possible_paths):
        #print('Decision path: ', index,val)
        age = 0
        for t, d in enumerate(val): 
            age += 1
            x_next = teeth_decay(par, d=d, x_t=x[index,t], age=age, extension= extension) #Transitioning from one time period to the next
            x[index, (t+1)] = x_next
            v[index, (t+1)] = (par.beta**t)*total_util(par, d=d, x_t=x[index, t])
            #print('Dental usage in period', t, ' = ', x[index, t])
        v[index, -1] = (par.beta**(par.life_span+1))*total_util(par, d=0, x_t=x[index, -1])
        best_path = v.sum(axis=1).argmax() #The decision path maximizing total utility 
        best_decisions = possible_paths[best_path]

    return best_decisions, x[best_path] #Return results: decision path to choose and how the teeth decay

