import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from numpy.random import normal 

#Question1

#1.1



def optimal_labor_supply(kappa = 1, alpha = 0.5, v = 1/(2*(16*16)), w = 1.0, tau = 0.3):
    wt = (1-tau)*w #wage tilde
    nominator = -kappa + np.sqrt(kappa*kappa + 4*(alpha/v)*wt*wt)
    denominator = 2*wt
    return nominator/denominator



def solve_Q1(x0= 12, bound=(0,24), alpha = 0.5, kappa = 1, v = 1/(2*(16*16)), w = 1.0, tau = 0.3, G= 1, extension = False):
    #extension for Q1.3 to check numerical solution. 
    if extension == True:
        Ls = optimal_labor_supply(kappa= kappa, alpha=alpha, v=v, w=w, tau=tau)
        G = tau*w*Ls

    def objective(L): 
        C = kappa + (1-tau)*w*L
        obj = np.log((C**alpha)*(G*(1-alpha))) - 0.5*v*(L**2)
        return - obj
    
    result = optimize.minimize(objective, x0, method='SLSQP', bounds= [bound], options={'disp': False})
    

    return result, G



#1.3
def analytical_Ls_and_G(alpha = 0.5, kappa = 1, v = 1/(2*(16*16)), w = 1.0, tau = 0.3): 

    Ls = optimal_labor_supply(kappa= kappa, alpha=alpha, v=v, w=w, tau=tau) #Get the optimal labor supply with given parameters
    G = tau*w*Ls #Calculate G with the optimal labor supply
    Cs = kappa + (1-tau)*w*Ls #optimal consumption is pinned down by the optimal labor supply
    util = np.log((Cs**alpha)*(G*(1-alpha))) - 0.5*v*(Ls**2)
    return Ls, G, util

#1.4
def objective_function_tau(tau):
    # Calculate the worker utility using the analytical_Ls_and_G function with one input
    _, _, utility = analytical_Ls_and_G(tau=tau)
    # We max worker utility -->  return the negative for the minimizer
    return -utility


#1.5
def solve_Q5_G(x0= 12, bound=(0,24), alpha = 0.5, kappa = 1, v = 1/(2*(16*16)), w = 1.0, tau = 0.6632, G= 1.0, sigma= 1.001, rho= 1.001, eps= 1.0):
    #Default value for tau was taken from Q1.4, note that we first use G=1 to find the optimal labor supply and then use this to find the optimal G
   
    def objective(L): 
        C = kappa + (1-tau)*w*L
        exp1 = (sigma-1)/sigma #Exponent from the inner soft bracket
        exp2 = 1/exp1 #Exponent from the outer soft bracket
        exp3 =1-rho #Exponent from the outermost hard bracket
        exp4 = 1 + eps
        #First term
        nom1 = (((alpha*C**exp1 + (1-alpha)*G**exp1)**exp2)**exp3)-1 
        denom1 = exp3
        nom2 = v*L**exp4
        #Second term
        denom2 = exp4
        obj = nom1/denom1 - nom2/denom2
        return - obj
    
    result = optimize.minimize(objective, x0, method='SLSQP', bounds= [bound], options={'disp': False})
    Ls = result.x[0] #Optimal labor supply
    Gs = tau*w*Ls #Optimal G
    if isinstance(result.fun, float):
        utility = np.array([-result.fun])  # Convert utility to a single-element array
    else: 
        utility = -result.fun[0] #Utility
    return Ls, Gs, utility


#1.5b To confirm results

def util_plot(sigma, rho, L,tau, G, eps=1.0,kappa=1.0, v=1/(2*(16*16)), alpha=0.5, w=1.0): 
    C = kappa + (1-tau)*w*L
    exp1 = (sigma-1)/sigma #Exponent from the inner soft bracket
    exp2 = 1/exp1 #Exponent from the outer soft bracket
    exp3 =1-rho #Exponent from the outermost hard bracket
    exp4 = 1 + eps
    #First term
    nom1 = (((alpha*C**exp1 + (1-alpha)*G**exp1)**exp2)**exp3)-1 
    denom1 = exp3
    nom2 = v*L**exp4
    #Second term
    denom2 = exp4
    util = nom1/denom1 - nom2/denom2
    return util


#1.6

def two_step(sigma, rho, tau, G, eps=1):
    Ls_1, Gs_1, util_1 = solve_Q5_G(sigma=sigma, rho=rho, tau=tau, G=G, eps=eps) #Initial solver for the worker problem 
    Ls_2, Gs_2, util_2 = solve_Q5_G(sigma=sigma, rho=rho, tau=tau, G=Gs_1, eps=eps) #Now with updated G
    return Ls_2, Gs_2, util_2

#2.1

#Note that we have set the baseline parameters as default

#Define the profit function
def profit_function(k, l, eta=0.5, w=1.0): 
    profit = k * (l**(1-eta))-w*l
    return profit

#Use profite function to find the optimal l for a given k
def profit_maximization(k, x0=0):
    obj_ = lambda x: -profit_function(k=k, l=x) #Objective function
    result = optimize.minimize_scalar(obj_, method='bounded', bounds=(0, 1e10), bracket=(x0,)) #Very high upper bound
    return result.x, -result.fun

def analytical_profit(k, eta=0.5, w=1.0):
    sol = ((1-eta)*k/w)**(1/eta)
    return sol


#2.2 + 2.3


# We set a random seed used in class for the rest of our results 
np.random.seed(0)



# The demand-shock as a AR(1) model in logs is first defined:
def dynamic_function(k_t, epsilon, rho=0.9):
    log_k = rho * np.log(k_t) + epsilon
    k = np.exp(log_k)
    return k

# We then define the objective function for the dynamic optimization
def dynamic_objective(extension, Delta=0.05, T=119, sigma_epsilon=0.1, eta = 0.5, w=1.0, iota = 0.01, R= (1+0.01)**(1/12)):
    k_array = np.zeros(T + 1)
    k_array[0] = 1.0  # The initial demand shock
    h_value = 0.0
    l_prev = 0.0 # Initially we do not have any employees

    for t in range(T):
        #Draw a random error term
        epsilon_t = normal(loc=-0.5 * sigma_epsilon**2, scale=sigma_epsilon)

        #Calculate Demand shock
        k_array[t+1] = dynamic_function(k_array[t],epsilon_t)

        #Calculate optimal demand of labor:
        if extension == False: #See function below for the extension 
            l_current = analytical_profit(k=k_array[t]) 
        else: 
            l_current = policy_function(k_t=k_array[t], l_prev=l_prev, Delta=Delta)

        # "h" ex post value is conditional on the shock series:

        #Add value to the sum
        h_value += R**(-t) * (k_array[t] * l_current**(1 - eta) - w * l_current - int(l_current != l_prev) * iota) 
        
        #Update the previous labor demand
        l_prev=l_current

    return h_value


# An approximation of the ex ante expected value is setted
def ex_ante_value(K= int(1e4),sigma_epsilon=0.1, T=119, extension= False, Delta=0.05, iota = 0.01):
    h_sum = 0.0

    for k in range(K):
        epsilon_series = normal(loc=-0.5 * sigma_epsilon**2, scale=sigma_epsilon, size=T)
        h_sum += dynamic_objective(extension= extension, Delta=Delta, iota=iota) #Default values are used
    h_approx = h_sum / K
    return h_approx


#We build an extension into our ex ante function and dynamic objective function

# Policy function to determine l_t based on the given policy
def policy_function(k_t, l_prev, Delta = 0.05):
    l_star = analytical_profit(k=k_t)
    if abs(l_prev - l_star) > Delta:
        l_t = l_star
    else:
        l_t = l_prev
    return l_t

#2.4




#3


def griewank(x):
    return griewank_(x[0],x[1])
    
def griewank_(x1,x2):
    A = x1**2/4000 + x2**2/4000
    B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))
    return A-B+1


def refined_global_optimizer(x0s, tau=1e-08, K_bar= 10, K_max= 1000, N = 2):
    "Multi-start optimizer with warm up and refined starting values. Returns the best solution (xopt and fopt), starting values in xk0v, local optima input and outputs in xs and fs"

    
    # Step 1: Initialize variables
    fopt = np.inf
    xopt = np.nan
    xs = np.empty((K_max, N))
    xk0v = np.empty((K_max, N))
    fs = np.empty((K_max, N))

    # Step 2: Iterate for each k
    for k, x0 in enumerate(x0s):
        #Warm up
        if k < K_bar:
            #Run optimizer with initial guess x^k
            xk0v[k, :] = x0 #Store the initial guess
            result = optimize.minimize(griewank, x0, method='BFGS', tol=tau)
            xs[k, :] = result.x #Store inputs at local optimum
            f = result.fun
            fs[k] = f  #Store the function value at local optimum


            #Update best solution if best to date
            if f < fopt:
                fopt = f
                xopt = result.x
                print(f'{k:4d}: x0 = ({x0[0]:7.2f}, {x0[1]:7.2f})', end='')
                print(f' -> converged at ({xs[k][0]:7.2f}, {xs[k][1]:7.2f}) with f = {f:12.8f}')
        
        #With refining starting values
        else:
            #Calculate chi value. Note that 0.5 and 2 cancel out
            chi = 0.50 * (2 / (1 + np.exp((k - K_bar) / 100)))
            
            #Set refined initial guess x^{k0}
            xk0 = chi * x0 + (1 - chi) * xopt
            xk0v[k, :] = xk0
            #Run optimizer with refined initial guess x^{k0}
            result = optimize.minimize(griewank, xk0, method='BFGS', tol=tau)
            xs[k, :] = result.x
            f = result.fun
            fs[k] = f

            #Update best solution if necessary
            if f < fopt:
                fopt = f
                xopt = result.x
                print(f'{k:4d}: x0 = ({x0[0]:7.2f}, {x0[1]:7.2f})', end='')
                print(f' -> converged at ({xs[k][0]:7.2f}, {xs[k][1]:7.2f}) with f = {f:12.8f}')
                
        
        # Step 3G: Check if f(x^*) is less than the tolerance tau
        if fopt < tau:
            print(f'Optimal solution found at iteration {k}')
             # Strip empty values from xs if the loop breaks
            xs = xs[:k, :]
            fs = fs[:k, :]
            xk0v = xk0v[:k, :]
            break
    
    # Step 4: Return the best solution x^*
    return xopt, fopt, xs, fs, xk0v, k




def print_path(xk0v, fs,  K_bar= 10):
    #Extract the two values of each pair
    plt.plot(xk0v[:,0], color= 'orange', label= '$x_1^{k0}$') 
    plt.plot(xk0v[:,1], color= 'green', label= '$x_2^{k0}$')
    #Add labels
    plt.xlabel('k')
    plt.ylabel('$x^{k0}$')
    k = len(fs)
    #Add visual support
    plt.xlim(0,k+10) #To show that it stops after
    plt.axhline(y=0, color='red', linestyle='--')  #horizontal 0-line
    plt.axvline(x=k, color='grey', alpha= 0.5,linestyle='--', label= '$k^*$')  #horizontal 0-line
    plt.axvline(x=K_bar, color='blue',linestyle='--', label ='Warm-up K')  #horizontal 0-line
    plt.title('Iteration path of $x^{k0}$ with Warm-up K = '+str(K_bar))
    plt.legend()
    return plt.show()