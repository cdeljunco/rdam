#!/usr/bin/env python

import numpy as np
#for finding peaks
from scipy import signal
#for linear regressions
from scipy.optimize import curve_fit
#for matrix algebra
from scipy.linalg import solve_banded
from scipy import sparse
from scipy.sparse.linalg import spsolve
#for plotting
import matplotlib.pyplot as plt
#system
import os
import multiprocessing as mp

class Simulation:

    '''
    Base class for simulations. 
    '''

    def __init__(self, Da, Omega, stress, init, bc, dt, t_max, t_save_interval, dx, x_max, x_save_interval):
        
        print("The Simulation class doesn't do anything on its own. Use Repeat_Simulation or Unique_Simulation")

        pass

    def save(self):

        np.savez_compressed(self.filename, t = self.t, x = self.x, c = self.sol_c, v = self.sol_v)

    def load(self):
        
        npzfile = np.load(self.filename)
        self.t = npzfile['t']
        self.x = npzfile['x']
        self.sol_c = npzfile['c']
        self.sol_v = npzfile['v'] 

    def delete_file(self):
        
        os.remove(self.filename)
       
        return


class Repeat_Simulation(Simulation):
    
     '''
     Class for running simulations with random initial conditions. In that we want several simulations with the same parameters but different initial conditions, for averaging. This class treats the set of N repeated simulations as one object and saves them to the same file. They are accessed as self.c[0]...self.c[N-1] and self.v[0]...self.c[N-1]. self.t and self.x are 1D arrays that are good for all repeats.
     '''

     def __init__(self, Da, Omega, stress, init, bc, dt, t_max, t_save_interval, dx, x_max, x_save_interval, repeats = 2, t_min_save = 0, check = False, rxn = 'quad'):

        '''
        Search for a saved simulation file with given parameters in the folder ./Data/. If it exists, load it. If it doesn't exist, run the simulation and save it in the folder ./Data/. 
        '''

        self.dx = dx
        self.dt = dt
        self.t_min_save = t_min_save
        self.t_max = t_max
        self.x_max = x_max
        self.t_save_interval = t_save_interval
        self.x_save_interval = x_save_interval

        self.bc = bc
        self.init = init   
        self.stress = stress
        self.rxn = rxn
        self.Da = Da
        self.Omega = Omega

        if self.t_min_save==0 and rxn == 'quad':
            self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-init{:}-bc{:}-dt{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.init, self.bc, self.dt, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval)
        elif self.t_min_save!=0 and rxn == 'quad': 
            self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-init{:}-bc{:}-dt{:.2e}-tmin{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.init, self.bc, self.dt, self.t_min_save, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval) 
        elif rxn != 'quad':
            self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-rxn{:}-init{:}-bc{:}-dt{:.2e}-tmin{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.rxn, self.init, self.bc, self.dt, self.t_min_save, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval) 

        
         #if check = True, check for file and return
        if check:
            if os.path.isfile(self.filename):
                print("file found")
                self.exists = True
            else:
                print("file not found, rerun with check = False (default) to run simulation and save to file.") 
                self.exists = False
            return

        #check whether file with this data already exists and load it in

        if os.path.isfile(self.filename):
            print("file found, loading...")
            self.load()

            #integrate remaining repeats
            if self.get_repeats() < repeats:
                print("integrating...")

                pool = mp.Pool(mp.cpu_count())
                results = pool.map(self.run, range(repeats - self.get_repeats()))
                pool.close()
                
                for i, result in enumerate(results):

                    success = result[0]
    
                    if success:
                        self.sol_c = np.append(self.sol_c, [result[1]], axis = 0)
                        self.sol_v = np.append(self.sol_v, [result[2]], axis = 0)

                    else:
                        print('overflow with these parameters')
                        return()
                   
                self.save() 
            
        #if it doesn't exist, initialize
        else:
            print("file not found, integrating...") 
            pool = mp.Pool(mp.cpu_count())
            results = pool.map(self.run, range(repeats))
            pool.close()

            success = results[0][0]

            if success:
                self.sol_c = [results[0][1]]
                self.sol_v = [results[0][2]]
                self.t = results[0][3]
                self.x = results[0][4]

            else:
                print('overflow with these parameters, deleting file')
                self.delete_file() 
                return()

            #integrate remaining repeats
            for i, result in enumerate(results[1:]):
                success = result[0]
                if success:
                    self.sol_c = np.append(self.sol_c, [result[1]], axis = 0)
                    self.sol_v = np.append(self.sol_v, [result[2]], axis = 0)

                else:
                    print('overflow with these parameters, deleting file')
                    self.delete_file() 
                    return()

            self.save()

     def get_repeats(self):
        '''
        Returns the number of repeat simulations
        '''
        
        return(len(self.sol_c))

     def get_phys_params(self):

        names = ['Omega', 'Da', 'initialization', 'stress', 'boundary conditions', 'reaction term']
        values = [self.Omega, self.Da, self.init, self.stress, self.bc, self.rxn]

        for name, value in zip(names, values):
            if type(value) == int or type(value) == float:
                print(name + ': {:.2e}'.format(value))
            elif type(value) == str: 
                print(name + ': ' + value) 
        return


     def get_sim_params(self):

        names = ['dt', 't_max', 't_save_interval', 't_min_save', 'dx', 'x_max', 'x_save_interval']
        values = [self.dt, self.t_max, self.t_save_interval, self.t_min_save, self.dx, self.x_max, self.x_save_interval]

        for name, value in zip(names, values):
            if type(value) == int or type(value) == float:
                print(name + ': {:.2e}'.format(value))
            elif type(value) == str: 
                print(name + ': ' + value)
   
        return

       
     def run(self, i):
       
        seed =  i * np.random.randint(1, high = 1E6) 
        M = int(self.t_max/self.dt)
        N = int(self.x_max/self.dx)
        success, t, x, new_sol_c, new_sol_v = solve_implicit(self.t_max, self.x_max, M, N, self.t_min_save, self.t_save_interval, self.x_save_interval, self.Da, self.Omega, self.stress, self.init, self.bc, seed = seed, rxn = self.rxn)

        return success, new_sol_c, new_sol_v, t, x 

        #if len(self.sol_v) == 0 and len(self.sol_c) == 0: 
        #    self.sol_c = [new_sol_c]
        #    self.sol_v = [new_sol_v]
            
        #else:
        #    self.sol_c = np.append(self.sol_c, [new_sol_c], axis = 0)
        #    self.sol_v = np.append(self.sol_v, [new_sol_v], axis = 0)

        #return success

     def plot_c(self, t_plot, repeat, ax = None):

        if ax == None:
            fig, ax = plt.subplots()

        if repeat > len(self.sol_c) - 1:
            print('index too high, there are ', len(self.sol_c), 'trajectories for these parameters')
        
        plot_index = np.where(self.t > t_plot)[0][0]
        ax.plot(self.x, self.sol_c[repeat, plot_index])
        ax.set_xlabel('x')
        ax.set_ylabel('c')
        

     def plot_v(self, t_plot, repeat, ax = None):

        if ax == None:
            fig, ax = plt.subplots()
        
        if repeat > len(self.sol_v) - 1:
            print('index too high, there are ', len(self.sol_v), 'trajectories for these parameters')
        
        plot_index = np.where(self.t > t_plot)[0][0]
        ax.plot(self.x, self.sol_v[repeat, plot_index])
        ax.set_xlabel('x')
        ax.set_ylabel('v')
       

 
class Unique_Simulation(Simulation):

     '''
     Class for running a simulation with deterministic initial conditions.    
     '''


     def __init__(self, Da, Omega, stress, init, bc, dt, t_max, t_save_interval, dx, x_max, x_save_interval, slope = None, t_min_save = 0, check = False, rxn = 'quad'):
        
        '''
        Search for a saved simulation file with given parameters in the folder ./Data/. If it exists, load it. If it doesn't exist, run the simulation and save it in the folder ./Data/. 
        '''


        if init == "step" and bc == "pbc":
            print("step initial conditions and pbc are not consistent")
        if init == "tanh" and bc == "pbc":
            print("tanh initial conditions and pbc are not consistent")
        if init == "tanh" and slope == None:
            print("provide a slope for tanh initial conditions")

        self.dx = dx
        self.dt = dt
        self.t_max = t_max
        self.x_max = x_max
        self.t_min_save = t_min_save
        self.t_save_interval = t_save_interval
        self.x_save_interval = x_save_interval

        self.bc = bc
        self.init = init 
        self.slope = slope 
        self.stress = stress
        self.rxn = rxn
        self.Da = Da
        self.Omega = Omega

        if self.t_min_save == 0 and rxn == 'quad':
            if self.init != "tanh":
                self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-init{:}-bc{:}-dt{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.init, self.bc, self.dt, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval) 
            elif self.init == "tanh":
                self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-init{:}{:.2e}-bc{:}-dt{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.init, self.slope, self.bc, self.dt, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval)      
        elif self.t_min_save!=0 and rxn == 'quad': 
            if self.init != "tanh":
                self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-init{:}-bc{:}-dt{:.2e}-tmin{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.init, self.bc, self.dt, self.t_min_save, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval) 
            elif self.init == "tanh":
                self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-init{:}{:.2e}-bc{:}-dt{:.2e}-tmin{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.init, self.slope, self.bc, self.dt, self.t_min_save, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval)    

        elif rxn != 'quad': 
            if self.init == "step":
                self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-rxn{:}-init{:}-bc{:}-dt{:.2e}-tmin{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.rxn, self.init, self.bc, self.dt, self.t_min_save, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval) 
            elif self.init == "tanh":
                self.filename = "Data/da{:.2e}-omega{:.2e}-stress{:}-rxn{:}-init{:}{:.2e}-bc{:}-dt{:.2e}-tmin{:.2e}-tmax{:.2e}-tsaveint{:.2e}-dx{:.2e}-xmax{:.2e}-xsaveint{:.2e}.npz".format(self.Da, self.Omega, self.stress, self.rxn, self.init, self.slope, self.bc, self.dt, self.t_min_save, self.t_max, self.t_save_interval, self.dx, self.x_max, self.x_save_interval) 
   
         #if check = True, check for file and return
        if check:
            if os.path.isfile(self.filename):
                print("file found")
                self.exists = True
            else:
                print("file not found, rerun with check = False (default) to run simulation and save to file.") 
                self.exists = False
            return

        #check whether file with this data already exists and load it in

        if os.path.isfile(self.filename):
            print("file found, loading...")
            self.load()

        #if sim does not exist, run it

        else: 
            print("file not found, integrating...")
            success = self.run()
            if success:
                self.save()
            else:
                print('overflow with these parameters')
                return None

     def get_phys_params(self):

        names = ['Omega', 'Da', 'initialization', 'slope', 'stress', 'boundary conditions', 'reaction term']
        values = [self.Omega, self.Da, self.init, self.slope, self.stress, self.bc, self.rxn]

        for name, value in zip(names, values):
            if type(value) == int or type(value) == float:
                print(name + ': {:.2e}'.format(value))
            elif type(value) == str: 
                print(name + ': ' + value)
   
        return


     def get_sim_params(self):

        names = ['dt', 't_max', 't_save_interval', 't_min_save', 'dx', 'x_max', 'x_save_interval']
        values = [self.dt, self.t_max, self.t_save_interval, self.t_min_save, self.dx, self.x_max, self.x_save_interval]

        for name, value in zip(names, values):
            if type(value) == int or type(value) == float:
                print(name + ': {:.2e}'.format(value))
            elif type(value) == str: 
                print(name + ': ' + value)
   
        return


     def run(self):
        
        M = int(self.t_max/self.dt)
        N = int(self.x_max/self.dx)
        success, self.t, self.x, self.sol_c, self.sol_v = solve_implicit(self.t_max, self.x_max, M, N, self.t_min_save, self.t_save_interval, self.x_save_interval, self.Da, self.Omega, self.stress, self.init, self.bc, slope = self.slope, rxn = self.rxn)

        return success

     def plot_c(self, t_plot, ax = None):

        if ax == None:
            fig, ax = plt.subplots() 
        
        plot_index = np.where(self.t > t_plot)[0][0]
        ax.plot(self.x, self.sol_c[plot_index])
        ax.set_xlabel('x')
        ax.set_ylabel('c')
        

     def plot_v(self, t_plot, ax = None):

        if ax == None:
            fig, ax = plt.subplots()
        
        plot_index = np.where(self.t > t_plot)[0][0]
        ax.plot(self.x, self.sol_v[plot_index])
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        

def lin_fxn(x, m, b):
    return m*x + b          


def solve_implicit(T, L, M, N, t_min_save, t_save_interval, x_save_interval, Da, omega, stress, init, bc, seed = np.random.randint(1), slope = None, rxn = 'quad'):

        
    """
    uses implicit in time scheme to integrate the active RD system. 
    call: solve_implicit(total time, box length, time steps, space steps, diffusion coefficient, reaction rate, gamma/eta, alpha/eta, stress type, boundary conditions)
    """

    #discretization
    dx, dt = L/(N + 1), T/(M + 1)

    if bc == "fixed":

        #set up "dirichlet matrix" for implicit solution for c
        #set constants
        F, b = (1/Da)*dt/(dx*dx), dt
        #set up full matrix for calculation
        #G = np.eye(N) + F*A
        G_banded = [np.concatenate(([0], np.full(N - 1, -F)), axis = None), np.full(N, 1 + 2*F), np.concatenate((np.full(N - 1, -F), [0]), axis=None)]
        
        #set up "dirichlet matrix" for implicit solution for v
        m =  dx * dx
        #H = (2 + m) * np.diag(np.ones(N)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
        H_banded = [np.concatenate(([0], np.full(N - 1, -1)), axis = None), np.full(N, 2 + m), np.concatenate((np.full(N - 1, -1), [0]), axis=None)]
        #set constants
        p, u = dx * omega, dt/dx
        
        c_old = np.zeros(N + 2)
        c_new = np.zeros(N + 2)
        v = np.zeros(N + 2) #already contains boundary conditions v(0) = v(L) = 0
    
        c_return = []
        v_return = []
        
        #set initial and boundary conditions
        c_old, c_new = initialize_c(init, N, seed, slope = slope, dx = dx)

        #this will store the reaction term
        W = np.zeros(N)
        #this will store the gradient term (vc)_x in the concentration equation
        Y = np.zeros(N)
        #this will store the gradient term c_x in the velocity equation
        Z = np.zeros(N)
        temp = np.zeros(N + 2)
        
        for i in range(M + 1):

            try:
           
                if stress == "lin":
                    Z = c_old[1:-1] - c_old[0:-2]
                elif stress == "sat":
                    temp = c_old[:]/(1 + c_old[:])
                    Z = temp[1:-1] - temp[0:-2]
                elif stress == "c2":
                    temp = c_old[:]*c_old[:]
                    Z = temp[1:-1] - temp[0:-2]
     
                #print(len(Z), N, len(v[1:-1]), len(c_old[1:-1]))
    
                #compute v
                v[1:-1] = solve_banded((1, 1), H_banded, p*Z)
             
                #reaction term
                if rxn == 'quad':
                    W = c_old[1:-1]*(1 - c_old[1:-1])
                elif rxn == 'cube':
                    W = c_old[1:-1]*(1 - c_old[1:-1]*c_old[1:-1])
    
                W[0], W[-1] = boundary_conditions(init, F, b, W[0], W[-1])
                
                #gradient term
                vc = v*c_old
                Y = vc[1:-1] - vc[0:-2]
                c_new[1:-1] = solve_banded((1, 1), G_banded, c_old[1:-1] + b*W - u*Y)
        
                if i % t_save_interval == 0 and i*dt > t_min_save:
                    c_return.append([c_new[j] for j in range(0, N + 2, x_save_interval)])
                    v_return.append([v[j] for j in range(0, N + 2, x_save_interval)]) 
     
                c_old = c_new  
    
            except ValueError as err:
                print('overflow at timestep ', i, ' omega = ', omega)
                return 0,0,0,0,0
    
        t_return = np.linspace(t_min_save, T, len(c_return))
        x_return = np.linspace(0, L, len(c_return[-1]))
 

    
    elif bc == "pbc":

        
        #set up "dirichlet matrix" for implicit solution for c
        A = 2 * np.diag(np.ones(N + 2)) - np.diag(np.ones(N + 1), 1) - np.diag(np.ones(N + 1), -1)
        #set corner elements for PBC
        A[0, N + 1] = -1
        A[N + 1, 0] = -1
        #set constants
        F, b = (1/Da)*dt/(dx*dx), dt
        #set up full matrix for calculation
        G = np.eye(N + 2) + F*A
        #convert to sparse for storage and computational efficiency
        sG = sparse.csr_matrix(G)               
        
        #set up "dirichlet matrix" for implicit solution for v
        m =  dx * dx
        H = (2 + m) * np.diag(np.ones(N + 2)) - np.diag(np.ones(N + 1), 1) - np.diag(np.ones(N + 1), -1)
        #set corner elements for PBC
        H[0, N + 1] = -1
        H[N + 1, 0] = -1
        sH = sparse.csr_matrix(H)
        #set constants
        p, u = dx * omega, dt/dx

        c_old = np.zeros(N + 2)
        c_new = np.zeros(N + 2)
        v = np.zeros(N + 2)
    
        c_return = []
        v_return = []
        
        #set initial and boundary conditions
        np.random.seed(seed)
        c_old = 1 + 0.001*(np.random.randn(N + 2))
        
        #this will store the reaction term
        W = np.zeros(N + 2)
        #this will store the gradient term (vc)_x in the concentration equation
        Y = np.zeros(N + 2)
        #this will store the gradient term c_x in the velocity equation
        Z = np.zeros(N + 2)
        temp = np.zeros(N + 2)
        
        for i in range(M + 1):
           
            try:

              if stress == "lin":
                  Z[1:-1] = c_old[1:-1] - c_old[0:-2]
                  Z[0] = c_old[0] - c_old[-1]
                  Z[-1] = c_old[-1] - c_old[-2]

              elif stress == "sat":
                  temp = c_old[:]/(1 + c_old[:])
                  Z[1:-1] = temp[1:-1] - temp[0:-2]
                  Z[0] = temp[0] - temp[-1]
                  Z[-1] = temp[-1] - temp[-2]

              elif stress == "c2":
                  temp = c_old[:]*c_old[:]
                  Z[1:-1] = temp[1:-1] - temp[0:-2]
                  Z[0] = temp[0] - temp[-1]
                  Z[-1] = temp[-1] - temp[-2]
                       #compute v
              v[:] = spsolve(sH, p*Z)         

              #reaction term
              if rxn == 'quad':
                  W = c_old[:]*(1 - c_old[:])
              elif rxn == 'cube':
                  W = c_old[:]*(1 - c_old[:]*c_old[:])
                  
              #gradient term
              vc = v * c_old
              Y[1:-1] = vc[1:-1] - vc[0:-2]
              Y[0] = vc[0] - vc[-1]
              Y[-1] = vc[-1] - vc[-2]
              
              c_new[:] = spsolve(sG, c_old[:] + b*W - u*Y)

              if i % t_save_interval == 0 and i*dt > t_min_save:
                  c_return.append([c_new[j] for j in range(0, N + 2, x_save_interval)])
                  v_return.append([v[j] for j in range(0, N + 2, x_save_interval)]) 
 
              c_old = c_new       
            
            except ValueError as err:
                print('overflow at timestep ', i, ' omega = ', omega)
                return 0,0,0,0,0
    

        t_return = np.linspace(t_min_save, T, len(c_return))
        x_return = np.linspace(0, L, len(c_return[-1]))
    
    return 1, np.array(t_return), np.array(x_return), np.array(c_return), np.array(v_return)

def I(N):
    
    """
    initial conditions; a step function
    """
    x = np.arange(N)
    y = np.zeros(N)
    y += 1 * (x < 5)
    return y  

def my_tanh(N, dx, slope):
    
    """
    initial conditions; a hyperbolic tangent function
    """
    x = np.arange(0, N * dx, dx)

    return -0.5*np.tanh((x - 25)*1./slope) + 0.5

def initialize_c(init, N, seed, slope = None, dx = None):

    c_initial = np.zeros(N + 2)
    c_later = np.zeros(N + 2)
        
    if init == "step":
        c_initial = I(N + 2)
        c_later[0] = 1

    elif init == "tanh":
        c_initial = my_tanh(N + 2, dx, slope)
        plt.plot(np.arange(0, (N+2) * dx, dx), c_initial)
        c_later[0] = 1

    else:
        print("invalid initial conditions")

    #elif init == "sine":

    return c_initial, c_later

def boundary_conditions(init, F, b, wleft, wright):
        
    wlnew = wleft + (F/b)
    if init == "pert":
        wrnew = wright + (F/b)
    elif (init == "step") or (init =="tanh"):
        wrnew = 0

    return wlnew, wrnew

