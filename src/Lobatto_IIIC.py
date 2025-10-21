#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:33:33 2025

@author: aron
"""


# =============================================================================
# comments
# =============================================================================

# file is structured as follows: 
    
# imports
# set up coefficients Lobatto IIIC
# set up solver function

# set up mass-spring-damper system
# compute and plot results


# during computation, solvers plots a label for the current computation 
#      as well as current computation time and step size 
    

#%%imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.linalg as linalg
import time
# plt.style.use('seaborn')


#%%
# =============================================================================
# mass-spring-damper system
# =============================================================================

#%% set up double mass-spring-damper system (set d_1 = d_2 = 0 for undamped system)
#    ----spring_1---- m_1 ----spring_2---- m_2 ---spring_3 ----

m_1=1
m_2=1
M=np.diag((m_1, m_2))

bs=2   #number of bodies
ds=1  #spatial dimension
dim=bs*ds  #dimension of positional variables
d=2*dim    #number of all differential variables

r_1=10    #length of springs at rest
r_2=10
r_3=10

k_1=1   #spring constants  
k_2=1 
k_3=1 

d_1 = 0 # damping constants 
d_2 = 0  # setting d_2 > 0 reduces index from 2 (d_2 = 0) to 1 for m_2 = 0 

alpha = 0 #set input force pushing m_1 to the right

l=r_1+r_2+r_3 #length of model such that equilibrium is reached for all springs at rest length

#right hand side function for ODE 
def fd(W):
    p_1, p_2, v_1, v_2 = W
    rhs = np.array([
        v_1,
        v_2,
        - k_1*(p_1 - r_1) + 2*k_2*(p_2-p_1-r_2) - d_1 *v_1 + alpha,
        - 2*k_2*(p_2-p_1-r_2) + k_3*(l-p_2-r_3)- d_2 *v_2
        ])
    return rhs

#set initial values
w_start=np.array([6,24,0,0], dtype='float64')


#%%
# =============================================================================
# coefficient matrices Lobatto IIIC
# =============================================================================

#%% lobatto-3C coefficient matrices and vectors for 2 and 3 stages



l_A2 = np.array( [ [1/2, -1/2],  [1/2, 1/2],  ] )
l_c2 = np.sum(l_A2,1)


l_A3 = np.array([ [ 1/6, -1/3, 1/6 ],
                    [ 1/6, 5/12, -1/12 ],
                    [ 1/6, 2/3, 1/6  ] ]) 
l_c3 = np.sum(l_A3,1)


#vectors with weights to compute next solution
l_B2 = l_A2[-1:]
l_B3 = l_A3[-1:]

#lists of matrices and coefficients s.t. solver can call them according to number of stages 
# first entry of 0 is just a placeholder s.t. indicies match number of stages
l_Alist = [0, 0, l_A2, l_A3]
l_Blist=[0, 0, l_B2, l_B3]
l_clist = [0, 0, l_c2, l_c3]


#%%
# =============================================================================
# solver function Lobatto IIIC
# =============================================================================

#%% lobatto IIIC wrapper function to solve until a specified integration time with stepsize control by repeatedly calling solver_LO3C

def wrapper_LO3C(w_start, time, stages, initstepsize, maxstepsize=0.1, minstepsize=1e-50, hfactor=5, chunksize=50, tol=1e-6,  max_iter=5000, label=None):
    '''
    
    calls solver_LO3C with varying stepsizes until a specified integration time or maximum number of steps is reached
    

    Parameters
    ----------
    w_start : np.array with initial values
        needs to be consistent with algebraic equations
    time : any number > 0
        end of desired integration interval
    stages : integer, either 2 or 3
        number of stages for PRK method
    initstepsize : any number > 0
        stepsize for first call to solver_LO3C
    maxstepsize : optional
        DESCRIPTION. The default is 0.1.
    minstepsize : optional
        DESCRIPTION. The default is 1e-50.
    hfactor : optional
        scaling factor for stepsize control. The default is 5.
    chunksize : optional
        maximum number of integration steps carried out by solver function until stepsize is adapted. The default is 50.
    tol : optional
        computed as 2-norm of nonlinear system (solved in solver_LO3C) evaluated at numerical solution.    
        keeping / exceeding the tolerance determines decrease / increase of stepsize. The default is 1e-6.
    max_iter : optional
        maximum of total number of integration steps. The default is 5000.

    Returns
    -------
    sols : np.array
        returns solutions as array of shape (number of vars, integration steps)
    NL_sols : np.array
        returns solutions of nonlinear system that was solved in each step. shape is (integration steps, number of vars)
    F_vals : np.array
        returns evaluations of nonlinear system at numerical solutions (used for tolerance / step size control) 
    h_vals : 
        array of all used stepsizes, needed for scaled plots 

    '''
    
    s=stages
    if not s in [2,3]:
        raise Exception('{} is an invalid stage number. Please set s=2 or s=3'.format(s))
    w_curr=w_start.copy()
    t=0
    F_vals=[]
    h_vals=np.array([])
    sols=np.reshape(w_curr,(1,d))
    NL_sols=np.zeros((1, 2*s*d))
    h=initstepsize
    while t < time:
        if label:
            print(label)
        print('{}: current integration time'.format(t))
        print('{}: final integration time'.format(time))
        if h < minstepsize:
            print('minimum stepsize of {} is reached'.format(minstepsize))
            break
        if len(sols) > max_iter:
            print('maximum number of {} iterations reached'.format(max_iter))
            break
        
        chunk, NL, F = solver_LO3C(w_curr, h, chunksize, s, tol)
        
        if (len(chunk)) == 0:  #tolerance was exceeded in first step, no solutions added and stepsize is decreased       
            h *= 1/hfactor
            print('{}: current step size, tolerance exceeded in first step'.format(h))
        elif len(chunk) < chunksize:     #tolerance was exceeded, previous solutions are added and stepsize is decreased       
            sols=np.append(sols, chunk, 0)
            NL_sols = np.append(NL_sols, NL, 0)
            F_vals += F
            w_curr=chunk[-1]
            t += h * len(chunk)
            h_vals = np.append(h_vals, h * np.ones(len(chunk)))
            h *= 1/hfactor
            print('{}: current step size, tolerance exceeded'.format(h))
        else:      #tolerance was not exceeded, stepsize is increased
            sols=np.append(sols, chunk, 0)
            NL_sols = np.append(NL_sols, NL, 0)
            F_vals += F
            w_curr=chunk[-1]
            t += h * len(chunk)
            h_vals = np.append(h_vals, h * np.ones(len(chunk)))
            h = min(hfactor*h, maxstepsize)          
            print('{}: current step size, tolerance kept'.format(h))
            
    #NL_sols has zero as first entry for computational reasons 
    return sols, NL_sols[1:], np.asarray(F_vals), h_vals    
  


#%% Lobatto 3C solver function

#     variables are in order:    [X_1, X_2, ..., X_s, 
#                                X_dot_1, ..., X_dot_s]


def solver_LO3C(w_start, stepsize, steps, stages, tol):
    '''
    integrates system with Lobatto 3C solver until either number of integration steps has reached 'steps' or 
    tolerance is exceeded. in that case, all previously computed solutions are returned

    Parameters
    ----------
    w_start : np.array
        needs to be consistent with algebraic equations if mass matrix is singular 
    stepsize : any number > 0
    steps : integer
        number of integration steps 
    s : integer, either 2 or 3
        number of stages for PRK method
    tol : 
        computed as 2-norm of nonlinear system evaluated at numerical solution of each integration step.    
        solver runs until either tolerance is exceeded or desired number of steps is reached. 
    init_guess : string, either 'n' or 'e'
        either use a naive ('n') or educated ('e') guess as the initial value for solving the nonlinear system.
        The default is 'n'.

    Returns
    -------
   sols : np.array
       returns solutions as array of shape (number of vars, integration steps)
   NL_sols : np.array
       returns solutions of nonlinear system that was solved in each step. shape is (integration steps, number of vars)
   F_vals : list
       returns evaluations of nonlinear system at numerical solutions (used for tolerance / step size control) 

    '''   
    
    
    s=stages
    w_curr=w_start
    h=stepsize
        
    sols=[w_start]
    NL_sols=[]
    F_vals=[]
    
    X_init = np.tile(w_curr, s)
    X_dot_init = np.zeros(d*s)
    
    W_init=np.concatenate((X_init, X_dot_init))            

    E=linalg.block_diag(np.identity(dim), M)
    E_block = np.kron(np.identity(s),E)
    A_block = np.kron(l_Alist[s], np.identity(d))
    B_block = np.kron(l_Blist[s], np.identity(d))


    for i in range(steps):        
        x_start=w_curr[:d]
            
        def F(W):    #create nonlinear system of equations to solve in each step
            x_start_block = np.tile(x_start, s)
            X_block = W[:d*s]
            X_dot_block = W[d*s : 2*d*s]
            OMblock_mat = np.reshape( np.split(X_block, s), (s,d))     #create matrix with all variables to apply fd 
            
            #to avoid inversion of M, X_dot's are explicitly computed 
            NLsys = np.concatenate((
                           x_start_block + h * np.dot(A_block, X_dot_block) - X_block,
                           np.dot(E_block, X_dot_block) - np.apply_along_axis(fd, 1, OMblock_mat).flatten(),
                    ))
            
            return NLsys
    

        NL = fsolve(F, W_init)
        F_val = np.linalg.norm(F(NL)) / np.linalg.norm(NL)   #compute 2-norm of numerical solution to NL system, should be close to root
        if F_val > tol:
            break
        
        NL_sols.append(NL)
        F_vals.append(F_val)

        
        w_curr += h * np.dot(B_block, NL[d*s : 2*d*s])   #update current solution
        
        sols.append(w_curr.copy())
        
        W_init = np.concatenate( ( np.tile( w_curr[:d],s), np.zeros(d*s) ) )
        
        
    #first solution is omitted as it is already stored and would be doubled 
    return np.asarray(sols)[1:], np.asarray(NL_sols), F_vals 




#%%
# =============================================================================
# compute and plot results 
# =============================================================================
# =============================================================================
# 
# =============================================================================

#%%  compute numerical solution with lobatto IIIC solver

# initial values and masses can be adapted in set up at the top or overwritten here
# change parameters in wrapper_LO3C according to documentation

start=time.time() #tracks computation time

l_msd_sols, l_msd_NLsols, l_msd_Fvals, l_msd_hvals = wrapper_LO3C(w_start, 50, 3, 1e-5, tol=1e-10, max_iter=200000, maxstepsize=0.1, label='LIIIC mass-spring-damper, generic results')

end=time.time()
print('computation took {}s'.format(round(end-start,2)))   #prints how long the computation took in seconds


#store solutions of positions and velocities
l_p_1, l_p_2 ,l_v_1 ,l_v_2 = np.split(l_msd_sols,4, 1) #save individual variables for plotting 

#compute energies
l_msd_kin = 0.5 * ( m_1 * l_v_1**2 + m_2 * l_v_2**2) 
l_msd_pot = 0.5 * (k_1*(l_p_1-r_1)**2 + 2*k_2*(l_p_2-l_p_1-r_2)**2 + k_3*(l-l_p_2-r_3)**2)
l_msd_tot = l_msd_kin + l_msd_pot

#creates values for x-axis of plots s.t. variable stepsizes are accounted for (x-axis corresponds to 'real-time')
l_x_ax = np.append(0, np.cumsum(l_msd_hvals)) 




#%%
# =============================================================================
# plot results 
# x-axis is scaled to time, independent of varying step sizes
# =========================================================================

#%% positions

fig, ax = plt.subplots( layout='constrained')
ax.plot(l_x_ax, l_p_1, label='m_1')  
ax.plot(l_x_ax, l_p_2, label='m_2')  
ax.set_xlabel('time')  
ax.set_ylabel('x')  
ax.set_title("Lobatto IIIC positions") 
ax.legend(frameon=True)  
plt.show()


#%% velocities

fig, ax = plt.subplots( layout='constrained')
ax.plot(l_x_ax, l_v_1, label='m_1')  
ax.plot(l_x_ax, l_v_2, label='m_2')  
ax.set_xlabel('time')  
ax.set_ylabel('x')  
ax.set_title("Lobatto IIIC velocities") 
ax.legend(frameon=True) 
plt.show()

#%% energies

fig, ax = plt.subplots( layout='constrained')
ax.plot(l_x_ax, l_msd_kin, label='kinetic energy')  
ax.plot(l_x_ax, l_msd_pot, label='potential energy')
ax.plot(l_x_ax, l_msd_tot, label='total energy')  
ax.set_xlabel('time')  
ax.set_ylabel('energy')  
ax.set_title("Lobatto IIIC energies")
ax.legend(frameon=True) 
plt.show()
