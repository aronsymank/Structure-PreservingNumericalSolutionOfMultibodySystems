#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:48:17 2025

@author: aron
"""


# =============================================================================
# comments
# =============================================================================

# file is structured as follows: 
    
# imports
# set up coefficients JPRK methods
# set up solver function 

# set up simple pendulum
# compute and plot results 

# during computation, solver plots a label for the current computation 
#      as well as current computation time and step size 
    


#%%imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time 
# plt.style.use('seaborn')


#%%
# =============================================================================
# coefficient matrices JPRK
# =============================================================================
#%% Lobatto 3A - 3B pair setup coefficient matrices and vectors for 2 and 3 stages

#Lobatto 3j_A
j_A2 = np.array( [ [0, 0], [0.5, 0.5] ] )
j_c2 = np.sum(j_A2,1)

j_A3 = np.array([  [0,0,0], 
                   [5/24, 1/3,-1/24],
                   [1/6, 2/3, 1/6] ])
j_c3 = np.sum(j_A3,1)

#vectors with weights to compute next solution
j_B2=np.array( [0.5, 0.5] )
j_B3=np.array( [1/6, 2/3, 1/6] )


#Lobatto 3j_B
j_A2_b = np.array( [ [0.5, 0],  [0.5, 0],  ] )
j_c2_b = np.array([0, 1])

j_A3_b = np.array([ [ 1/6, -1/6, 0 ],
                    [ 1/6, 1/3, 0 ],
                    [ 1/6, 5/6, 0 ] ]) 
j_c3_b = np.sum(j_A3_b,1)


#vectors with weights to compute next solution
j_B2_b = np.array([0.5, 0.5])
j_B3_b = np.array([1/6, 2/3, 1/6])


#lists for reference in solver
# first two 0-entries are just placeholders s.t. index corresponds to stages
j_Alist = [0, 0, j_A2, j_A3]
j_Alist_b = [0, 0, j_A2_b, j_A3_b]

j_Blist=[0, 0, j_B2, j_B3]
j_Blist_b=[0, 0, j_B2_b, j_B3_b]

clist = [0, 0, j_c2, j_c3]
clist_b = [0, 0, j_c2_b, j_c3_b]


#%%
# =============================================================================
# solver function MPRK
# =============================================================================

#%% JPRK wrapper function to solve until a specified integration time with stepsize control by repeatedly calling solver_JPRK


def wrapper_JPRK(w_start, time, stages, initstepsize, maxstepsize=0.1, minstepsize=1e-50, hfactor=5, chunksize=50, tol_1=1e-6, tol_2=1e-2,  max_iter=5000, label=None):
    '''
    calls solver_JRPK with varying stepsizes until a specified integration time or maximum number of steps is reached
    

    Parameters
    ----------
    w_start : np.array with initial values
        needs to be consistent with algebraic equations
    time : any number > 0
        end of desired integration interval
    stages : integer, either 2 or 3
        number of stages for PRK method
    initstepsize : any number > 0
        stepsize for first call to solver_JPRK
    maxstepsize : optional
        DESCRIPTION. The default is 0.1.
    minstepsize : optional
        DESCRIPTION. The default is 1e-50.
    hfactor : optional
        scaling factor for stepsize control. The default is 5.
    chunksize : optional
        maximum number of integration steps carried out by solver function until stepsize is adapted. The default is 50.
    tol_1 : optional
        computed as 2-norm of first nonlinear system (solved in solver_JPRK) evaluated at numerical solution.    
        keeping / exceeding the tolerance determines decrease / increase of stepsize. The default is 1e-6.
    tol_2 : optional
        computed as 2-norm of second nonlinear system (solved in solver_JPRK) evaluated at numerical solution.
        keeping / exceeding the tolerance determines decrease / increase of stepsize. The default is 1e-2.
    max_iter : optional
        maximum of total number of integration steps. The default is 5000.
    label : optional, string. print statement to track the current computation 
        The default is None
            

    Returns
    -------
    sols : np.array
        returns solutions as array of shape (number of vars, integration steps)
    NL_sols : np.array
        returns solutions of nonlinear system that was solved in each step. shape is (integration steps, number of vars)
    F_vals : np.array
        returns evaluations of nonlinear systems at numerical solutions (used for tolerance / step size control) 
        of shape (integration steps, 2). columns correspond to respective nonlinear system 
    h_vals : 
        array of all used stepsizes, needed for scaled plots 

    '''
    s=stages
    if not s in [2,3]:
        raise Exception('{} is an invalid stage number. Please set s=2 or s=3'.format(s))
    w_curr=w_start.copy()
    t=0
    F_vals=np.zeros((1,2))
    h_vals=np.array([])
    sols=np.reshape(w_curr,(1,dt))
    NL_sols=np.zeros((1, s*(3*dim+c)))
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
        
        chunk, NL, F = solver_JPRK(w_curr, h, chunksize, s, tol_1, tol_2)
        
        if (len(chunk)) == 0:   #tolerance was exceeded in first step, no solutions added and stepsize is decreased       
            h *= 1/hfactor
            # print('exceeded in first step, new h is {}'.format(h))
            print('{}: current step size, tolerance exceeded in first step'.format(h))
        elif len(chunk) < chunksize:    #tolerance was exceeded, previous solutions are added and stepsize is decreased       
            sols=np.append(sols, chunk, 0)
            NL_sols = np.append(NL_sols, NL, 0)
            F_vals = np.append(F_vals, F, 0)
            w_curr=chunk[-1]
            t += h * len(chunk)
            h_vals = np.append(h_vals, h * np.ones(len(chunk)))
            h *= 1/hfactor
            # print('exceeded, new h is {}'.format(h))
            print('{}: current step size, tolerance exceeded'.format(h))
        else:                   #tolerance was not exceeded, stepsize is increased
            sols=np.append(sols, chunk, 0)
            NL_sols = np.append(NL_sols, NL, 0)
            F_vals = np.append(F_vals, F, 0)
            w_curr=chunk[-1]
            t += h * len(chunk)
            h_vals = np.append(h_vals, h * np.ones(len(chunk)))
            h = min(hfactor*h, maxstepsize)             
            # print('tolerance kept, new h is {}'.format(h))
            print('{}: current step size, tolerance kept'.format(h))

    # NL_sols and F_vals start with zero entry for numpy reasons
    return sols, NL_sols[1:], F_vals[1:,:], h_vals    
  
    
    
#%% J-PRK solver function

#     variables are in order:    [X_2, ..., X_s, 
#                                Y_1, Y_2, ..., Y_s, 
#                                Y_dot_1, ..., Y_dot_{s-1},
#                                Z_1, ..., Z_{s-1}]
# X_1 = x_0, and the last stages of Y_dot and Z are computed after x_1


def solver_JPRK(w_start, stepsize, steps, stages, tol_1, tol_2, init_guess='n'):
    '''
    integrates system with J-PRK method until either number of integration steps has reached 'steps' or 
    tolerance is exceeded. in that case, all previously computed solutions are returned

    Parameters
    ----------
    w_start : np.array
        needs to be consistent with algebraic equations
    stepsize : any number > 0
    steps : integer
        number of integration steps 
    stages : integer, either 2 or 3
        number of stages for PRK method
    tol_1 : 
        computed as 2-norm of first nonlinear system evaluated at numerical solution of each integration step.    
        solver runs until either a tolerance is exceeded or desired number of steps is reached. 
    tol_2 : 
        computed as 2-norm of second nonlinear system evaluated at numerical solution of each integration step.    
        solver runs until either a tolerance is exceeded or desired number of steps is reached. 

    Returns
    -------
   sols : np.array
       returns solutions as array of shape (number of vars, integration steps)
   NL_sols : np.array
       returns solutions of nonlinear system that was solved in each step. shape is (integration steps, number of vars)
   F_vals : np.arary
       returns evaluations of nonlinear systems at numerical solutions (used for tolerance / step size control) 
       of shape (integration steps, 2). columns correspond to respective nonlinear system 

    '''   
    
    s=stages
    w_curr=w_start.copy()
    h=stepsize
    if not (s==2 or s==3):
        raise Exception('stage number s has to be 2 or 3')
        
    sols=[w_start]
    NLsols=[]
    F_vals_1=[]
    F_vals_2=[]
    
    
    #initial guess for fsolve
    W_init = np.concatenate( ( np.tile( w_curr[:dim],s-1), np.tile(w_curr[dim:2*dim],s), np.zeros(dim*(s-1)), np.zeros(c*(s-1)))) 
    
    E_block = np.kron(np.identity(s-1), M)
    A_block = np.kron(j_Alist[s][1:,:], np.identity(dim))
    A_block_b = np.kron(j_Alist_b[s][:,:-1], np.identity(dim))
    B_block = np.kron(j_Blist[s], np.identity(dim))
    B_block_b = np.kron(j_Blist_b[s], np.identity(dim))
    
    for i in range(steps):
        x_start=w_curr[:dim].copy()
        y_start=w_curr[dim:2*dim].copy()
            

        def F(W):  #create nonlinear system of equations to solve in each step
            x_start_block = np.tile(x_start, s-1)
            y_start_block = np.tile(y_start, s)
            X_block = W[:dim*(s-1)]
            Y_block = W[dim*(s-1) : dim*(2*s-1)]
            Y_dot_block = W[dim*(2*s-1): dim*(3*s-2)]
            Z_block = W[dim*(3*s-2) : dim*(3*s-2) + c*(s-1)]
            
            #matrix with all variables to evaluate fd along axis
            OMblock_mat = np.concatenate(( np.reshape (np.append(x_start, X_block[:-dim]), (s-1,dim)) , 
                                          np.reshape(Y_block[:-dim], (s-1,dim)), 
                                          np.reshape(Z_block, (s-1, c))),
                                          axis=1) 
            
            #matrix with all differential variables to evaluate fc along axis
            X_block_mat = np.reshape( X_block, (s-1,dim) )
            
            
            #to avoid inversion of M, Y_dot's are explicitly computed 
            NLsys = np.concatenate((
                           x_start_block + h * np.dot(A_block, Y_block) - X_block,
                           y_start_block + h * np.dot(A_block_b, Y_dot_block) - Y_block,
                           np.dot(E_block, Y_dot_block) - np.apply_along_axis(j_fd, 1, OMblock_mat).flatten(), 
                           np.apply_along_axis(j_fc, 1, X_block_mat).flatten()
                    ))            
            return NLsys


        
        NL_1 = fsolve(F, W_init)
        
        #check whether relative 2-norm of numerical solution exceeds tolerance when F is evaluated there (should be close to root)
        F_val_1 = np.linalg.norm(F(NL_1)) / np.linalg.norm(NL_1)
        if F_val_1 > tol_1:
            break

        #compute p_1        
        w_curr[:dim] += h * np.dot(B_block, NL_1[dim*(s-1) : dim*(2*s-1)])  
            
        #compute value for v_1 which does not yet include contribution of Z_s (and the resulting V_s)
        v_temp = y_start + h * np.dot(B_block_b[:,:-dim], NL_1[dim*(2*s-1) : dim*(3*s-2)]) 
        
        
        #set up nonlinear system to find Y_dot_s, Z_s such that index 2 constraint is satisfied 
        def find_Ls(W):
            Y_dot_s = W[:dim]  
            Z_s = W[dim : dim+c]
            X_s = NL_1[(s-2)*dim : (s-1)*dim]
            Y_s = NL_1[(2*s-2)*dim : (2*s-1)*dim]
            p_1 = w_curr[:dim]
            NLsys = np.append(  np.dot(G(p_1), v_temp) + h *  np.dot(G(p_1), np.dot(B_block_b[:,-dim:], Y_dot_s)),
                              np.dot(M, Y_dot_s) - j_fd( np.concatenate( (X_s, Y_s, Z_s) ).flatten() ) 
                              )
            return NLsys
        
        #starting value for fsolve
        Ls_init = np.append( NL_1[(3*s-3)*dim : (3*s-2)*dim], NL_1[-c:]) #Y_dot_{s-1}, Z_{s-1}
        #compute Y_dot_s, Z_s
        NL_2 = fsolve(find_Ls, Ls_init)
        Y_dot_s, L_s = NL_2[:dim], NL_2[dim:]        
        
        
        #check tolerance for second nonlinear system
        F_val_2 = np.linalg.norm(find_Ls(NL_2)) / np.linalg.norm(NL_2)
        if F_val_2  > tol_2:
            break
        
        
        #if for loop runs until here, tolerance was kept and solutions are added / updated
        
        #add values from function evaluation
        F_vals_1.append(F_val_1)
        F_vals_2.append(F_val_2)
        
        
        NL=np.insert(NL_1, (3*s-2)*dim, Y_dot_s)     #add last stage of V_dot to solution
        NL=np.append(x_start, NL)                 #add p_0 as first stage for X
        NL=np.append(NL,L_s)                  #add Lambda_s, solved by second NL system 

        # add solution of nonlinear system
        NLsols.append(NL)
    
        #update current step 
        w_curr[dim:2*dim] = v_temp + h * j_Blist_b[s][-1] * Y_dot_s      #update y
        w_curr[2*dim:2*dim+c] = L_s      #update z as last stage Z_s that was computed in find_Ls
        
        
        #add new solution 
        sols.append(w_curr.copy())
        
        #compute next initial guess for NL system
        W_init = np.concatenate( ( np.tile( w_curr[:dim],s-1), np.tile( w_curr[dim:2*dim],s), 
                                  np.zeros(dim*(s-1)), np.zeros(c*(s-1)) ) )

        
    #array with F_vals_1 and F_vals_2 
    F_vals=np.column_stack((F_vals_1, F_vals_2))
        
    #first solution is omitted as it is already stored and would be doubled 
    return np.asarray(sols)[1:], np.asarray(NLsols), F_vals


#%%
# =============================================================================
# simple pendulum
# =============================================================================
# =============================================================================
# 
# =============================================================================
#%% set up simple pendulum



l=1
m=1
g=-9.81
k=0      # viscosity term k ≥ 0


dim = 2     #spacial dimension of problem
d = 2*dim   # dimension of differential variables, twice as many due to order reduction
c = 1       # dimension (number) of constraints
dt = d + c   # total number of variables 


#mass matrix
M = m * np.identity(dim)



# define right hand side functions. make sure variables are in the right order


#differential function
def j_fd(W):
    x,y,u,v,z = W
    rhsd = np.array([
        - z*x - k*u,
        g*m - z*y - k*v,
        ])
    return rhsd


#constraint function for holonomic constraints (index 3)
def j_fc(W):
    x,y = W            #note that algebraic variables are ommitted here
    rhsc= np.array([      
        0.5 * (x**2 + y**2 - l**2)
        ])    
    return rhsc


#total derivative of holonomic constraints, needed in JPRK to compute last stage of algebraic variables
# s.t. numerical solution satisfies index 2 constraint G(p)v = 0
def G(W):            
    x,y = W         #note that algebraic variables are ommitted here
    G=np.array([
        x, y
           ])
    return G
    






#%%
# =============================================================================
# simple pendulum
# =============================================================================
# =============================================================================
# compute and plot results 
# =============================================================================

#%% JPRK compute numerical solution simple pendulum 


#adapt solver parameters in arguments of wrapper_JPRK according to documentation
#to change model parameters, go to set up above

#initial values
js_w_start = np.array([l, 0, 0 ,0, 0], dtype='float64')

# some alternative consistent initial values
# ms_w_start = np.array([l, 0, 0, 5, 0], dtype='float64')
# ms_w_start = np.array([0,-l,0,0,(-g*m)/l], dtype='float64')
# ms_w_start = np.array([0,-l, 5, 0,(-g*m)/l], dtype='float64')



start=time.time() #tracks computation time

js_sols, js_PRKsols, js_Fvals, js_hvals = wrapper_JPRK(js_w_start, 30, 3, 1e-4, 0.01, 1e-20, 5, 50, 1e-8, 1e-8, 50000, label='JPRK simple pendulum energy conservation')

end=time.time()
print('computation took {}s'.format(round(end-start,2)))   #prints how long the computation took in seconds
t_js_enCon=end-start

#store solutions
js_x, js_y, js_u, js_v, js_z = np.split(js_sols, 5, 1)

#compute constraints
js_constr_pos = js_x**2 + js_y**2 -l**2
js_constr_vel = js_x*js_u + js_y*js_v

# compute energies
js_kin = 0.5 * m * (js_u**2 + js_v**2)
js_pot = -(js_y+l)*m*g
js_tot = js_kin + js_pot


#creates values for x-axis of plots s.t. variable stepsizes are accounted for (x-axis corresponds to 'real-time')
js_x_ax = np.append(0, np.cumsum(js_hvals)) 


# =============================================================================
# plot results simple pendulum
# x-axis is scaled to time, independent of varying step sizes
# =============================================================================


#%% positions

fig, ax = plt.subplots( layout='constrained')
ax.scatter(js_x, js_y, label='x', s=10)  
ax.set_xlabel('x')  
ax.set_ylabel('y')  
ax.set_title("JPRK Simple pendulum positions") 
ax.axis('equal')
# ax.legend(frameon=True)  
plt.show()


#%% velocities

fig, ax = plt.subplots( layout='constrained')
ax.plot(js_x_ax, js_u, label='u')  
ax.plot(js_x_ax, js_v, label='v')  
ax.set_xlabel('time')  
ax.set_ylabel('velocity')  
ax.set_title("JPRK Simple pendulum velocities") 
ax.legend(frameon=True)  
plt.show()

#%% constraints


fig, ax = plt.subplots( layout='constrained')
ax.plot(js_x_ax, js_constr_pos, label='positional constraint')  
ax.plot(js_x_ax, js_constr_vel, label='velocity constraint')  
ax.set_xlabel('time')  
ax.set_title("JPRK Simple pendulum constraints") 
ax.legend(frameon=True)  
plt.show()

#%% energies


fig, ax = plt.subplots( layout='constrained')
ax.plot(js_x_ax, js_kin, label='kinetic energy')  
ax.plot(js_x_ax, js_pot, label='potential energy')
ax.plot(js_x_ax, js_tot, label='total energy')  
ax.set_xlabel('time')  
ax.set_ylabel('energy')  
ax.set_title("JPRK Simple pendulum energies")
ax.legend(frameon=True) 
plt.show()

