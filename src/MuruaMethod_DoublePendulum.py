#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:07:15 2025

@author: aron
"""


# =============================================================================
# comments
# =============================================================================

# file is structured as follows: 
    
# imports
# set up coefficients MPRK method
# set up MPRK solver function (optional to add GGL formulation)

# set up double pendulum
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
# double pendulum
# =============================================================================
# =============================================================================
# 
# =============================================================================


#%% set up double pendulum


l_1=1
l_2=1
m_1=3
m_2=1
g=-9.81
k=0  # viscosity term


bs = 2   #number of bodies
ds = 2   #spatial dimension of problem
dim = bs * ds     # dimension of positional variables
d = 2*dim   # dimension of differential variables
c = 2       # dimension (number) of constraints
dt = d + c   # total number of variables 


#different dimensions for GGL formulation
g_c = 2*c
g_dt = d + g_c


#mass matrix
M_1 = m_1 * np.identity(ds)
M_2 = m_2 * np.identity(ds)
M=linalg.block_diag(M_1,M_2)


# define right hand side functions. make sure variables are in the right order
# to fit the framework of E = diag(I_d, M) we need to set the variables as [x1, y1, x2, y2, u1, v1, u2, v2]
#  and the discretized stages accordingly 

#differential functions

# MPRK
def fd(W):
    x_1,y_1,x_2,y_2,u_1,v_1,u_2,v_2,z_1,z_2 = W
    rhsd = np.array([
        u_1,
        v_1,
        u_2,
        v_2,
        z_1*x_1 + (x_1-x_2)*z_2 - k*u_1,
        g*m_1 + y_1*z_1 + (y_1-y_2)*z_2- k*v_1,
        (x_2-x_1)*z_2 - k*u_2,
        g*m_2 + (y_2-y_1)*z_2 - k*v_2,        
        ])
    return rhsd


# GGL
def g_fd(W):
    x_1,y_1,x_2,y_2,u_1,v_1,u_2,v_2,z_1,z_2,µ_1,µ_2 = W
    rhsd = np.array([
        u_1 + µ_1*x_1 + (x_1-x_2)*µ_2,
        v_1 + y_1*µ_1 + (y_1-y_2)*µ_2,
        u_2 + (x_2-x_1)*µ_2,
        v_2 + (y_2-y_1)*µ_2, 
        z_1*x_1 + (x_1-x_2)*z_2 - k*u_1,
        g*m_1 + y_1*z_1 + (y_1-y_2)*z_2 - k*v_1,
        (x_2-x_1)*z_2 - k*u_2,
        g*m_2 + (y_2-y_1)*z_2 - k*v_2,        
        ])
    return rhsd



#constraint functions
#MPRK
def fc(W):
    x_1,y_1,x_2,y_2,u_1,v_1,u_2,v_2 = W     #note that algebraic variables are omitted here
    rhsc= np.array([
       x_1*u_1 + y_1*v_1,
      (x_1-x_2)*u_1 + (y_1-y_2)*v_1 + (x_2-x_1)*u_2 + (y_2-y_1)*v_2
        ])    
    return rhsc


# GGL
def g_fc(W):
    x_1,y_1,x_2,y_2,u_1,v_1,u_2,v_2 = W     #note that algebraic variables are omitted here
    rhsc = np.array([
       x_1*u_1 + y_1*v_1,
       (x_1-x_2)*u_1 + (y_1-y_2)*v_1 + (x_2-x_1)*u_2 + (y_2-y_1)*v_2,
       x_1**2 + y_1**2 -l_1**2,
       (x_1-x_2)**2 + (y_1-y_2)**2 - l_2**2,
        ])    
    return rhsc


#%%
# =============================================================================
# compute and plot results
# =============================================================================
# =============================================================================
# double pendulum
# =============================================================================

#%% MPRK compute results double pendulum

#adat solver parameters in arguments of wrapper_Murua according to documentation
#to change model parameters, go to set up above

#initial values
md_w_st = np.array([l_1,0,l_1+l_2,0,0,0,0,0,0,0], dtype='float64')



start=time.time() #tracks computation time

md_sols, md_PRK_sols, md_Fvals, md_hvals, md_F_invals = wrapper_Murua(md_w_st, 30, 3, 1e-4, maxstepsize=0.01, minstepsize=1e-20, 
                                                           hfactor=5, chunksize=50, tol=1e-8, max_iter=50000, label='MPRK double pendulum')
end=time.time()
print('computation took {}s'.format(round(end-start,2)))   #prints how long the computation took in seconds
t_md_enCon=end-start


#store solutions
md_x1,md_y1,md_x2,md_y2,md_u1,md_v1,md_u2,md_v2,md_z1,md_z2 = np.split(md_sols, 10, 1)


#compute constraints
md_constr_pos_1 = md_x1**2 + md_y1**2 -l_1**2
md_constr_vel_1 = md_x1*md_u1 + md_y1*md_v1

md_constr_pos_2 = (md_x1-md_x2)**2 + (md_y1-md_y2)**2 - l_2**2
md_constr_vel_2 = (md_x1-md_x2)*md_u1 + (md_y1-md_y2)*md_v1 + (md_x2-md_x1)*md_u2 + (md_y2-md_y1)*md_v2


# compute energies
md_kin = 0.5 * m_1 * (md_u1**2 + md_v1**2) + 0.5 * m_2 * (md_u2**2 + md_v2**2)  
md_pot = -m_1*g*(l_1+md_y1) - m_2*g*(md_y2+l_1+l_2)
md_tot = md_kin+md_pot


#creates values for x-axis of plots s.t. variable stepsizes are accounted for (x-axis corresponds to 'real-time')
md_x_ax = np.append(0, np.cumsum(md_hvals)) 




# =============================================================================
# plot results double pendulum 
# =============================================================================

#%% MPRK positions


fig, ax = plt.subplots( layout='constrained')
ax.plot(md_x1, md_y1, label='mass 1')  
ax.plot(md_x2, md_y2, label='mass 2')  
ax.set_xlabel('x')  
ax.set_ylabel('y')  
ax.set_title("MPRK double pendulum positions") 
ax.axis('equal')
ax.legend(frameon=True)  
plt.show()


#%% MPRK velocities

fig, ax = plt.subplots( layout='constrained')
ax.plot(md_x_ax, md_u1, label='u1')  
ax.plot(md_x_ax, md_v1, label='v1')  
ax.plot(md_x_ax, md_u2, label='u2')  
ax.plot(md_x_ax, md_v2, label='v2')  
ax.set_xlabel('time')  
ax.set_ylabel('velocity')  
ax.set_title("MPRK double pendulum velocities") 
ax.legend(frameon=True)  
plt.show()

#%% MPRK constraints

fig, ax = plt.subplots( layout='constrained')
ax.plot(md_x_ax, md_constr_pos_1, label='positional constraint mass 1')  
ax.plot(md_x_ax, md_constr_pos_2, label='positional constraint mass 2')  
ax.plot(md_x_ax, md_constr_vel_1, label='velocity constraint mass 1')  
ax.plot(md_x_ax, md_constr_vel_2, label='velocity constraint mass 2')  
ax.set_xlabel('time')  
ax.set_ylabel('constraint')  
ax.set_title("MPRK double pendulum constraints") 
ax.legend(frameon=True)  
plt.show()

#%% MPRK energies

fig, ax = plt.subplots( layout='constrained')
ax.plot(md_x_ax, md_kin, label='kinetic energy')  
ax.plot(md_x_ax, md_pot, label='potential energy')
ax.plot(md_x_ax, md_tot, label='total energy')  
ax.set_xlabel('time')  
ax.set_ylabel('energy')  
ax.set_title("MPRK double pendulum energies")
ax.legend(frameon=True) 
plt.show()

#%% MPRK GGL compute results double pendulum

#adat solver parameters in arguments of wrapper_Murua according to documentation
#to change model parameters, go to set up above

#initial values
gd_w_st = np.array([l_1,0,l_1+l_2,0,0,0,0,0,0,0,0,0], dtype='float64')

start=time.time() #tracks computation time

gd_sols, gd_PRK_sols, gd_Fvals, gd_hvals, gd_F_invals = wrapper_Murua(gd_w_st, 30, 3, 0.01, maxstepsize=0.01, minstepsize=1e-20, 
                                                           hfactor=5, chunksize=50, tol=1e-8, max_iter=50000, GGL=True,  label='MPRK GGL double pendulum')
end=time.time()
print('computation took {}s'.format(round(end-start,2)))   #prints how long the computation took in seconds
t_gd_enCon=end-start

#store solutions
gd_x1,gd_y1,gd_x2,gd_y2,gd_u1,gd_v1,gd_u2,gd_v2,gd_z1,gd_z2,gd_µ1,gd_µ2  = np.split(gd_sols, 12, 1)


#compute constraints
gd_constr_pos_1 = gd_x1**2 + gd_y1**2 -l_1**2
gd_constr_vel_1 = gd_x1*gd_u1 + gd_y1*gd_v1

gd_constr_pos_2 = (gd_x1-gd_x2)**2 + (gd_y1-gd_y2)**2 - l_2**2
gd_constr_vel_2 = (gd_x1-gd_x2)*gd_u1 + (gd_y1-gd_y2)*gd_v1 + (gd_x2-gd_x1)*gd_u2 + (gd_y2-gd_y1)*gd_v2


# compute energies
gd_kin = 0.5 * m_1 * (gd_u1**2 + gd_v1**2) + 0.5 * m_2 * (gd_u2**2 + gd_v2**2)  
gd_pot = -m_1*g*(l_1+gd_y1) - m_2*g*(gd_y2+l_1+l_2)
gd_tot = gd_kin+gd_pot


#creates values for x-axis of plots s.t. variable stepsizes are accounted for (x-axis corresponds to 'real-time')
gd_x_ax = np.append(0, np.cumsum(gd_hvals)) 




# =============================================================================
# plot results double pendulum 
# =============================================================================

#%% MPRK GGL positions

fig, ax = plt.subplots( layout='constrained')
ax.plot(gd_x1, gd_y1, label='mass 1')  
ax.plot(gd_x2, gd_y2, label='mass 2')  
ax.set_xlabel('x')  
ax.set_ylabel('y')  
ax.set_title("MPRK GGL double pendulum positions") 
ax.axis('equal')
ax.legend(frameon=True)  
plt.show()


#%% MPRK GGL velocities

fig, ax = plt.subplots( layout='constrained')
ax.plot(gd_x_ax, gd_u1, label='u1')  
ax.plot(gd_x_ax, gd_v1, label='v1')  
ax.plot(gd_x_ax, gd_u2, label='u2')  
ax.plot(gd_x_ax, gd_v2, label='v2')  
ax.set_xlabel('time')  
ax.set_ylabel('velocity')  
ax.set_title("MPRK GGL double pendulum velocities") 
ax.legend(frameon=True)  
plt.show()

#%% MPRK GGL constraints

fig, ax = plt.subplots( layout='constrained')
ax.plot(gd_x_ax, gd_constr_pos_1, label='positional constraint mass 1')  
ax.plot(gd_x_ax, gd_constr_pos_2, label='positional constraint mass 2')  
ax.plot(gd_x_ax, gd_constr_vel_1, label='velocity constraint mass 1')  
ax.plot(gd_x_ax, gd_constr_vel_2, label='velocity constraint mass 2')  
ax.set_xlabel('time')  
ax.set_ylabel('constraint')  
ax.set_title("MPRK GGL double pendulum constraints") 
ax.legend(frameon=True)  
plt.show()

#%% MPRK GGL energies

fig, ax = plt.subplots( layout='constrained')
ax.plot(gd_x_ax, gd_kin, label='kinetic energy')  
ax.plot(gd_x_ax, gd_pot, label='potential energy')
ax.plot(gd_x_ax, gd_tot, label='total energy')  
ax.set_xlabel('time')  
ax.set_ylabel('energy')  
ax.set_title("MPRK GGL double pendulum energies")
ax.legend(frameon=True) 
plt.show()

