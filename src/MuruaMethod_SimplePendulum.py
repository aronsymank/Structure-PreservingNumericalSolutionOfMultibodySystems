#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:36:36 2025

@author: aron
"""


# =============================================================================
# comments
# =============================================================================

# file is structured as follows: 
    
# imports
# set up coefficients MPRK method
# set up MPRK solver function (optional to add GGL formulation)

# set up simple pendulum
# compute and plot results with and without GGL formulation

# during computation, solver plots a label for the current computation 
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
# coefficient matrices MPRK
# =============================================================================

#%% Murua setup coefficient matrices and vectors

#Gauss collocation coefficients
m_A1 = np.array([0.5])
m_c1 = np.array([0.5])

m_A2 = np.array( [ [1/4, 1/4 - np.sqrt(3)/6 ], [1/4 + np.sqrt(3)/6, 1/4] ] )
m_c2 = np.sum(m_A2,1)

m_A3 = np.array([  [5/36, 2/9-np.sqrt(15)/15, 5/36-np.sqrt(15)/30], 
                   [ 5/36+np.sqrt(15)/24, 2/9,  5/36-np.sqrt(15)/24 ],
                   [ 5/36+np.sqrt(15)/30, 2/9 + np.sqrt(15)/15,  5/36] ])
m_c3 = np.sum(m_A3,1)

#vectors with weights to compute next solution
m_B1=np.array([1])
m_B2=np.array( [0.5, 0.5] )
m_B3=np.array( [5/18, 4/9, 5/18] )


#second method used for 'dummy' variable with Lobatto quadrature nodes and adapted weights for this method, denoted by _l 
m_A1_l = np.array([1])
m_c1_l = np.array([1])

m_A2_l = np.array( [ [0.25+(np.sqrt(3))/8, 0.25-(np.sqrt(3))/8],  [1/2, 1/2],  ] )
m_c2_l = np.sum(m_A2_l,1)

m_A3_l = np.array([ [ (25-np.sqrt(5) + 6*np.sqrt(15))/180, (10-4*np.sqrt(5))/45 , (25-np.sqrt(5) - 6*np.sqrt(15))/180 ],
                    [ (25+np.sqrt(5) + 6*np.sqrt(15))/180, (10+4*np.sqrt(5))/45  ,(25+np.sqrt(5) - 6*np.sqrt(15))/180 ],
                    [ 5/18,  4/9,  5/18  ] ]) 
m_c3_l = np.sum(m_A3_l,1)


#vectors with weights to compute next solution of constraint variable 
# z_0 is also used as part of the weighted sum for z_1, so the length is #stages + 1
m_B1_l = np.array( [ -1, 2] )       
m_B2_l = np.array([  1,  -1.7320508075688772,  1.7320508075688772])
m_B3_l = np.array([  -1, 5/3, -4/3 , 5/3 ])


#lists for reference in solver
m_Alist = [0, m_A1, m_A2, m_A3]
m_Alist_l = [0, m_A1_l, m_A2_l, m_A3_l]

clist = [0, m_c1, m_c2, m_c3]
clist_l = [0, m_c1_l, m_c2_l, m_c3_l]

m_Blist=[0, m_B1, m_B2, m_B3]
m_Blist_l=[0, m_B1_l, m_B2_l, m_B3_l]


#%%
# =============================================================================
# solver function MPRK
# =============================================================================

#%% Murua wrapper function to solve until a specified integration time with stepsize control by repeatedly calling solver_MRK


def wrapper_Murua(w_start, time, stages, initstepsize, maxstepsize=0.1, minstepsize=1e-50, hfactor=5, chunksize=50, tol=1e-6, max_iter=5000, init_guess='n', GGL=False, label=None):
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
        stepsize for first call to solver_MPRK
    maxstepsize : optional
        DESCRIPTION. The default is 0.1.
    minstepsize : optional
        DESCRIPTION. The default is 1e-50.
    hfactor : optional
        scaling factor for stepsize control. The default is 5.
    chunksize : optional
        maximum number of integration steps carried out by solver function until stepsize is adapted. The default is 50.
    tol : optional
        computed as 2-norm ofnonlinear system (solved in solver_MPRK) evaluated at numerical solution.    
        keeping / exceeding the tolerance determines decrease / increase of stepsize. The default is 1e-6.
    max_iter : optional
        maximum of total number of integration steps. The default is 5000.
    init_guess : string, either 'n' or 'e'
        either use a naive ('n') or educated ('e') guess as the initial value for solving the nonlinear system.
        The default is 'n'.
    GGL: optional, boolean. if set to True, the GGL formulation is used
        The default is False        
    label : optional, string. print statement to track the current computation 
        The default is None
    
    Returns
    -------
    sols : np.array
        returns solutions as array of shape (number of vars, integration steps)
    NL_sols : np.array
        returns solutions of nonlinear system that was solved in each step. shape is (integration steps, number of vars)
    F_vals : list
        returns evaluations of nonlinear system at numerical solutions (used for tolerance / step size control) 
    h_vals : 
        array of all used stepsizes, needed for scaled plots 
    
    '''
    s=stages
    if not s in [1,2,3]:
        raise Exception('{} is an invalid stage number. Please set s=1, s=2, or s=3'.format(s))
    w_curr=w_start.copy()
    t=0
    F_vals=[]
    F_in_vals=[]
    h_vals=np.array([])
    
    if GGL==True:
        sols=np.reshape(w_curr,(1,g_dt))
        NL_sols=np.zeros((1, s*(3*d+g_c)))
    elif GGL==False:
        sols=np.reshape(w_curr,(1,dt))
        NL_sols=np.zeros((1, s*(3*d+c)))
    else:
        raise Exception('GGL needs to be set to True or False')
    
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
        
        #call MPRK solver with or without GGL formulation
        if GGL==True:
            chunk, NL, F, F_iv = solver_MPRK_GGL(w_curr, h, chunksize, s, tol, init_guess)
        else:
            chunk, NL, F, F_iv = solver_MPRK(w_curr, h, chunksize, s, tol, init_guess)
        
        if (len(chunk)) == 0:   #tolerance was exceeded in first step, no solutions added and stepsize is decreased       
            h *= 1/hfactor
            print('{}: current step size, tolerance exceeded in first step'.format(h))
        elif len(chunk) < chunksize:      #tolerance was exceeded, previous solutions are added and stepsize is decreased          
            sols=np.append(sols, chunk, 0)
            NL_sols = np.append(NL_sols, NL, 0)
            F_vals += F
            F_in_vals += F_iv
            w_curr=chunk[-1]
            t += h * len(chunk)
            h_vals = np.append(h_vals, h * np.ones(len(chunk)))
            h *= 1/hfactor            
            print('{}: current step size, tolerance exceeded'.format(h))
        else:                               #tolerance was not exceeded, stepsize is increased
            sols=np.append(sols, chunk, 0)
            NL_sols = np.append(NL_sols, NL, 0)
            F_vals += F
            F_in_vals += F_iv
            w_curr=chunk[-1]
            t += h * len(chunk)
            h_vals = np.append(h_vals, h * np.ones(len(chunk)))
            h = min(hfactor*h, maxstepsize)  
            print('{}: current step size, tolerance kept'.format(h))
    
    #NL_sols has zero as first entry for numpy reasons 
    return sols, NL_sols[1:], F_vals, h_vals, F_in_vals
        
    

#%% Murua solver function

#     variables are in order:    [X_1, X_2, ..., X_s, 
#                                X_dot_1, ..., X_dot_s,
#                                X_1_l, ... , X_s_l
#                                Z_1, ..., Z_s]  


def solver_MPRK(w_start, stepsize, steps, stages, tol, init_guess):
    '''
    integrates system with M-PRK method until either number of integration steps has reached 'steps' or 
    tolerance is exceeded. in that case, all previously computed solutions are returned

    Parameters
    ----------
    w_start : np.array
        needs to be consistent with algebraic equations
    stepsize : any number > 0
    steps : integer
        number of integration steps 
    stages : integer, 1 ≤ stages ≤ 3
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
   F_vals : np.arary
       returns evaluations of nonlinear systems at numerical solutions (used for tolerance / step size control) 
       of shape (integration steps, 2). columns correspond to respective nonlinear system 

    '''   
    
    s=stages
    w_curr=w_start.copy()
    h=stepsize
    cs = clist[s]
    cs_l = clist_l[s]
        
    sols=[w_start]
    NLsols=[]
    Fvals=[]
    F_in_vals=[]
    
    
    E=linalg.block_diag(np.identity(dim), M)
    E_block = np.kron(np.identity(s),E)
    A_block = np.kron(m_Alist[s], np.identity(d))
    A_block_l = np.kron(m_Alist_l[s], np.identity(d))
    B_block = np.kron(m_Blist[s], np.identity(d))
    B_block_l = np.kron(m_Blist_l[s], np.identity(c))
    
    
    for i in range(steps):
        x_start=w_curr[:d]
            

        def F(W): #create nonlinear system of equations to solve in each step
            x_start_block = np.tile(x_start, s)
            X_block = W[:d*s]
            X_dot_block = W[d*s : 2*d*s]
            X_block_l = W[2*d*s : 3*d*s]
            X_block_l_mat = np.reshape( np.split(X_block_l, s), (s,d))
            Z_block = W[3*d*s : 3*d*s+c*s]
            OMblock_mat = np.append( np.reshape( np.split(X_block, s), (s,d)),     #create matrix with all variables 
                                np.reshape ( np.split(Z_block, s), (s,c)),
                                axis=1
                                )
            
      
            #to avoid inversion of M, X_dot's are explicitly computed 
            NLsys = np.concatenate((
                           x_start_block + h * np.dot(A_block, X_dot_block) - X_block,
                           np.dot(E_block, X_dot_block) - np.apply_along_axis(fd, 1, OMblock_mat).flatten(),
                           x_start_block + h * np.dot(A_block_l, X_dot_block) - X_block_l,
                           np.apply_along_axis(fc, 1, X_block_l_mat).flatten()
                    ))
            
            
            return NLsys
    

        # compute initial guess for non linear system: X1_dot = fd (w_curr), X1 = x0 + c0 *h* X1_dot, X2_dot = fd (X1, Z1) etc     
        #                                               analogous for X_l (except that X_dot_l values are not stored)
        #                                               Z values are given by current solution
        
        if init_guess=='e':
            
            Z_init = np.tile(w_curr[d:],s)    #to avoid implicit computations for Z values, use previous step
            X_init = np.zeros(d*s)
            X_dot_init = np.zeros(d*s)
            X_init_l = np.zeros(d*s)
            X_init[:d] = w_curr[:d] + h * cs[0] * fd(w_curr)
            X_dot_init[:d] = fd(np.append(X_init[:d], Z_init[:c]))
            X_init_l[:d] = w_curr[:d] + h * cs_l[0] * fd(w_curr)
            X_dot_init_l  = fd(np.append(X_init_l[:d], Z_init[:c]))    # no need to store vector with solutions since X_dot_l values are not used for W_init
            
            if s>1:
                for k in range(s-1):
                    X_init[(k+1)*d : (k+2)*d] = X_init[k*d : (k+1)*d] + h * (cs[k+1] - cs[k]) * fd(np.append(X_dot_init[k*d : (k+1)*d], Z_init[:c]))
                    X_dot_init[(k+1)*d : (k+2)*d] = fd(np.append(X_init[(k+1)*d : (k+2)*d], Z_init[:c]))
            
                    X_init_l[(k+1)*d : (k+2)*d] = X_init_l[k*d : (k+1)*d] + h * (cs_l[k+1] - cs_l[k]) * fd(np.append(X_dot_init_l, Z_init[:c]))
                    X_dot_init_l = fd(np.append(X_init_l[(k+1)*d : (k+2)*d], Z_init[:c]))
            
            W_init=np.concatenate((X_init, X_dot_init, X_init_l, Z_init))            
        
        #for the 'naive' initial guess, we use the current solution for X, X_l, Z and zero for X_dot
        elif init_guess=='n':
            W_init = np.concatenate( ( np.tile( w_curr[:d],s), np.zeros(d*s), np.tile( w_curr[:d],s), np.tile(w_curr[d:], s)) )
        
        else:
            raise Exception('only e and n are valid arguments for initial guess')
            
        # store function evaluation for initial guesses to see if 'educated guess' is actually an improvement    
        F_in_vals.append(np.linalg.norm(F(W_init))/np.linalg.norm(W_init)) 

        NL = fsolve(F, W_init)
        F_val = np.linalg.norm(F(NL)) / np.linalg.norm(NL)
        
        if F_val > tol:
            break

        NLsols.append(NL)
        Fvals.append(F_val)
        
        w_curr[:d] += h * np.dot(B_block, NL[d*s : 2*d*s])
        w_curr[d:] = np.dot(B_block_l, np.concatenate( ( w_curr[d:] , NL[3*d*s : 3*d*s+c*s] ) ) )  # [z_0, Z_1, ..., Z_s]
        
        sols.append(w_curr.copy())
        
        
    #first solution is omitted as it is already stored and would be doubled 
    return np.asarray(sols)[1:], np.asarray(NLsols), Fvals, F_in_vals  



#%% Murua solver function for Gear-Gupta-Leimkuhler formulation

# exactly the same as solver_MPRK, only defined separately so that the two different solvers can be called 
# for the same set up to make the code cleaner in one file 


#     variables are in order:    [X_1, X_2, ..., X_s, 
#                                X_dot_1, ..., X_dot_s,
#                                X_1_l, ... , X_s_l
#                                Z_1, ..., Z_s]  


def solver_MPRK_GGL(w_start, stepsize, steps, stages, tol, init_guess):
    '''
    integrates system with M-PRK method until either number of integration steps has reached 'steps' or 
    tolerance is exceeded. in that case, all previously computed solutions are returned

    Parameters
    ----------
    w_start : np.array
        needs to be consistent with algebraic equations
    stepsize : any number > 0
    steps : integer
        number of integration steps 
    stages : integer, 1 ≤ stages ≤ 3
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
   F_vals : np.arary
       returns evaluations of nonlinear systems at numerical solutions (used for tolerance / step size control) 
       of shape (integration steps, 2). columns correspond to respective nonlinear system 

    '''   
    
    s=stages
    w_curr=w_start.copy()
    h=stepsize
    cs = clist[s]
    cs_l = clist_l[s]
        
    sols=[w_start]
    NLsols=[]
    Fvals=[]
    F_in_vals=[]
    
    E=linalg.block_diag(np.identity(dim), M)
    E_block = np.kron(np.identity(s),E)
    A_block = np.kron(m_Alist[s], np.identity(d))
    A_block_l = np.kron(m_Alist_l[s], np.identity(d))
    B_block = np.kron(m_Blist[s], np.identity(d))
    B_block_l = np.kron(m_Blist_l[s], np.identity(g_c))

    
    for i in range(steps):
        x_start=w_curr[:d]

        def F(W): #create nonlinear system of equations to solve in each step
            x_start_block = np.tile(x_start, s)
            X_block = W[:d*s]
            X_dot_block = W[d*s : 2*d*s]
            X_block_l = W[2*d*s : 3*d*s]
            X_block_l_mat = np.reshape( np.split(X_block_l, s), (s,d))
            Z_block = W[3*d*s : 3*d*s + g_c*s]
            OMblock_mat = np.append( np.reshape( np.split(X_block, s), (s,d)),     #create matrix with all variables 
                                np.reshape ( np.split(Z_block, s), (s,g_c)),
                                axis=1
                                )
            
      
            #to avoid inversion of M, X_dot's are explicitly computed 
            NLsys = np.concatenate((
                           x_start_block + h * np.dot(A_block, X_dot_block) - X_block,
                           np.dot(E_block, X_dot_block) - np.apply_along_axis(g_fd, 1, OMblock_mat).flatten(),
                           x_start_block + h * np.dot(A_block_l, X_dot_block) - X_block_l,
                           np.apply_along_axis(g_fc, 1, X_block_l_mat).flatten()
                    ))
            
            
            return NLsys
    

        # compute initial guess for non linear system: X1_dot = g_fd (w_curr), X1 = x0 + c0 *h* X1_dot, X2_dot = g_fd (X1, Z1) etc     
        #                                               analogous for X_l (except that X_dot_l values are not stored)
        #                                               Z values are given by current solution
        
        if init_guess=='e':
            
            Z_init = np.tile(w_curr[d:],s)    #to avoid implicit computations for Z values, use previous step
            X_init = np.zeros(d*s)
            X_dot_init = np.zeros(d*s)
            X_init_l = np.zeros(d*s)
            X_init[:d] = w_curr[:d] + h * cs[0] * g_fd(w_curr)
            X_dot_init[:d] = g_fd(np.append(X_init[:d], Z_init[:g_c]))
            X_init_l[:d] = w_curr[:d] + h * cs_l[0] * g_fd(w_curr)
            X_dot_init_l  = g_fd(np.append(X_init_l[:d], Z_init[:g_c]))    # no need to store vector with solutions since X_dot_l values are not used for W_init
            
            if s>1:
                for k in range(s-1):
                    X_init[(k+1)*d : (k+2)*d] = X_init[k*d : (k+1)*d] + h * (cs[k+1] - cs[k]) * g_fd(np.append(X_dot_init[k*d : (k+1)*d], Z_init[:g_c]))
                    X_dot_init[(k+1)*d : (k+2)*d] = g_fd(np.append(X_init[(k+1)*d : (k+2)*d], Z_init[:g_c]))
            
                    X_init_l[(k+1)*d : (k+2)*d] = X_init_l[k*d : (k+1)*d] + h * (cs_l[k+1] - cs_l[k]) * g_fd(np.append(X_dot_init_l, Z_init[:g_c]))
                    X_dot_init_l = g_fd(np.append(X_init_l[(k+1)*d : (k+2)*d], Z_init[:g_c]))
            
            W_init=np.concatenate((X_init, X_dot_init, X_init_l, Z_init))            
        
        #for the 'naive' initial guess, we use the current solution for X, X_l, Z and zero for X_dot
        elif init_guess=='n':
            W_init = np.concatenate( ( np.tile( w_curr[:d],s), np.zeros(d*s), np.tile( w_curr[:d],s), np.tile(w_curr[d:], s)) )
        
        else:
            raise Exception('only e and n are valid arguments for initial guess')
            
        # store function evaluation for initial guesses to see if 'educated guess' is actually an improvement    
        F_in_vals.append(np.linalg.norm(F(W_init))/np.linalg.norm(W_init)) 

        NL = fsolve(F, W_init)
        F_val = np.linalg.norm(F(NL)) / np.linalg.norm(NL)
        
        if F_val > tol:
            break

        NLsols.append(NL)
        Fvals.append(F_val)
        
        w_curr[:d] += h * np.dot(B_block, NL[d*s : 2*d*s])
        w_curr[d:] = np.dot(B_block_l, np.concatenate( ( w_curr[d:] , NL[3*d*s : 3*d*s+g_c*s] ) ) )  # [z_0, Z_1, ..., Z_s]
        
        sols.append(w_curr.copy())
        
        
    #first solution is omitted as it is already stored and would be doubled 
    return np.asarray(sols)[1:], np.asarray(NLsols), Fvals, F_in_vals  



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


#different dimensions for GGL formulation
g_c = 2*c
g_dt = d + g_c

# mass matrix
M = m * np.identity(dim)


# define right hand side functions. make sure variables are in the right order


# differential functions

# MPRK
def fd(W):
    x,y,u,v,z = W
    rhsd = np.array([
        u,
        v,
        - z*x - k*u,
        g*m - z*y - k*v,
        ])
    return rhsd


# MPRK GGL
def g_fd(W):
    x,y,u,v,z,µ = W
    rhsd = np.array([
        u - µ*x,
        v - µ*y,
        - z*x - k*u,
        g*m - z*y - k*v,
        ])
    return rhsd



# constraint functions

#define constraint function for index-2 (velocity constraints), used for MPRK (no GGL)
def fc(W):
    x,y,u,v = W            #note that algebraic variables are omitted here
    rhsc= np.array([
        x*u+y*v    
        ])    
    return rhsc


#constraint function for GGL formulation
def g_fc(W):
    x,y,u,v = W            #note that algebraic variables are omitted here
    rhsc= np.array([
        x*u+y*v,
        x**2 + y**2 - l**2
        ])    
    return rhsc




#%%
# =============================================================================
# simple pendulum
# =============================================================================
# =============================================================================
# compute and plot results 
# =============================================================================
#%% MPRK compute numerical solution simple pendulum 


#adapt solver parameters in arguments of wrapper_Murua according to documentation
#to change model parameters, go to set up above

#initial values
ms_w_start = np.array([l, 0, 0, 0, 0], dtype='float64')

# some alternative consistent initial values
# ms_w_start = np.array([l, 0, 0, 5, 0], dtype='float64')
# ms_w_start = np.array([0,-l,0,0,(-g*m)/l], dtype='float64')
# ms_w_start = np.array([0,-l, 5, 0,(-g*m)/l], dtype='float64')



start=time.time() #tracks computation time

ms_sols, ms_PRKsols, ms_Fvals, ms_hvals, ms_F_invals = wrapper_Murua(ms_w_start, 30, 3, 1e-4, 0.01, 1e-20, 5, 50, 1e-8, 50000, init_guess='n', GGL=False, label='MPRK simple pendulum')

end=time.time()
print('computation took {}s'.format(round(end-start,2)))   #prints how long the computation took in seconds
t_ms_enCon=end-start


#store solutions 
ms_x, ms_y, ms_u, ms_v, ms_z = np.split(ms_sols, 5, 1)

#compute constraints
ms_constr_pos = ms_x**2 + ms_y**2 -l**2
ms_constr_vel = ms_x*ms_u + ms_y*ms_v

# compute energies
ms_kin = 0.5 * m * (ms_u**2 + ms_v**2)
ms_pot = -(ms_y+l)*m*g
ms_tot = ms_kin + ms_pot


#creates values for x-axis of plots s.t. variable stepsizes are accounted for (x-axis corresponds to 'real-time')
ms_x_ax = np.append(0, np.cumsum(ms_hvals)) 


# =============================================================================
# plot results simple pendulum
# x-axis is scaled to time, independent of varying step sizes
# =============================================================================
#%% positions

fig, ax = plt.subplots( layout='constrained')
ax.scatter(ms_x, ms_y, label='x', s=10)   
ax.set_xlabel('x')  
ax.set_ylabel('y')  
ax.set_title("MPRK Simple pendulum positions") 
ax.axis('equal')
# ax.legend(frameon=True)  
plt.show()


#%% velocities

fig, ax = plt.subplots( layout='constrained')
ax.plot(ms_x_ax, ms_u, label='u')  
ax.plot(ms_x_ax, ms_v, label='v')  
ax.set_xlabel('time')  
ax.set_ylabel('velocity')  
ax.set_title("MPRK Simple pendulum velocities") 
ax.legend(frameon=True)  
plt.show()

#%% constraints


fig, ax = plt.subplots( layout='constrained')
ax.plot(ms_x_ax, ms_constr_pos, label='positional constraint')  
ax.plot(ms_x_ax, ms_constr_vel, label='velocity constraint')  
ax.set_xlabel('time')  
ax.set_title("MPRK Simple pendulum constraints") 
ax.legend(frameon=True)  
plt.show()

#%% energies

fig, ax = plt.subplots( layout='constrained')
ax.plot(ms_x_ax, ms_kin, label='kinetic energy')  
ax.plot(ms_x_ax, ms_pot, label='potential energy')
ax.plot(ms_x_ax, ms_tot, label='total energy')  
ax.set_xlabel('time')  
ax.set_ylabel('energy')  
ax.set_title("MPRK Simple pendulum energies")
ax.legend(frameon=True) 
plt.show()

#%% MPRK GGL compute numerical solution simple pendulum 

#adapt solver parameters in arguments of wrapper_Murua according to documentation
#to change model parameters, go to set up above

#initial values
gs_w_start = np.array([l, 0, 0 ,0, 0, 0], dtype='float64')

# some alternative consistent initial values
# gs_w_start = np.array([l, 0, 0, 5, 0, 0], dtype='float64')
# gs_w_start = np.array([0,-l,0,0,(-g*m)/l,0], dtype='float64')
# gs_w_start = np.array([0,-l, 5, 0,(-g*m)/l,0], dtype='float64')

start=time.time() #tracks computation time

gs_sols, gs_PRKsols, gs_Fvals, gs_hvals, gs_F_invals = wrapper_Murua(gs_w_start, 30, 3, 1e-4, 0.01, 1e-20, 5, 50, 1e-8, 50000, init_guess='n', GGL=True, label='MPRK GGL simple pendulum')

end=time.time()
print('computation took {}s'.format(round(end-start,2)))   #prints how long the computation took in seconds
t_gs_enCon=end-start

#store solutions (one more variable for GGL)
gs_x, gs_y, gs_u, gs_v, gs_z, gs_µ = np.split(gs_sols, 6, 1)

#compute constraints
gs_constr_pos = gs_x**2 + gs_y**2 -l**2
gs_constr_vel = gs_x*gs_u + gs_y*gs_v

# compute energies
gs_kin = 0.5 * m * (gs_u**2 + gs_v**2)
gs_pot = -(gs_y+l)*m*g
gs_tot = gs_kin + gs_pot


#creates values for x-axis of plots s.t. variable stepsizes are accounted for (x-axis corresponds to 'real-time')
gs_x_ax = np.append(0, np.cumsum(gs_hvals)) 


# =============================================================================
# plot results simple pendulum
# x-axis is scaled to time, independent of varying step sizes
# =============================================================================


#%% positions

fig, ax = plt.subplots( layout='constrained')
ax.scatter(gs_x, gs_y, label='x', s=10)  
ax.set_xlabel('x')  
ax.set_ylabel('y')  
ax.set_title("MPRK GGL Simple pendulum positions") 
ax.axis('equal')
# ax.legend(frameon=True)  
plt.show()


#%% velocities

fig, ax = plt.subplots( layout='constrained')
ax.plot(gs_x_ax, gs_u, label='u')  
ax.plot(gs_x_ax, gs_v, label='v')  
ax.set_xlabel('time')  
ax.set_ylabel('velocity')  
ax.set_title("MPRK GGL Simple pendulum velocities") 
ax.legend(frameon=True)  
plt.show()

#%% constraints


fig, ax = plt.subplots( layout='constrained')
ax.plot(gs_x_ax, gs_constr_pos, label='positional constraint')  
ax.plot(gs_x_ax, gs_constr_vel, label='velocity constraint')  
ax.set_xlabel('time')  
ax.set_title("MPRK GGL Simple pendulum constraints") 
ax.legend(frameon=True)  
plt.show()

#%% energies

fig, ax = plt.subplots( layout='constrained')
ax.plot(gs_x_ax, gs_kin, label='kinetic energy')  
ax.plot(gs_x_ax, gs_pot, label='potential energy')
ax.plot(gs_x_ax, gs_tot, label='total energy')  
ax.set_xlabel('time')  
ax.set_ylabel('energy')  
ax.set_title("MPRK GGL Simple pendulum energies")
ax.legend(frameon=True) 
plt.show()



