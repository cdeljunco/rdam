import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit

def plot_snapshots(sim, i, var = 'c', colormap = 'Greys', ax = None):
    
    if var not in ['c', 'v', 'cgrad']:
        print("allowed values of var: 'c' for concentration, 'v' for flow field, 'cgrad' gradient of concentration")
        return

    if ax == None:
        fig, ax = plt.subplots()
    
    x = sim.x
   
    #for sim containing multiple simulations with different random initial conditions
    if hasattr(sim.sol_c[0][0], "__len__"):
        if var in ['c', 'cgrad']:
            y = sim.sol_c[0]
        elif var == 'v':
            y = sim.sol_v[0]
    #for sim containing a single simulation
    else:
        if var in ['c', 'cgrad']:
            y = sim.sol_c
        elif var == 'v':
            y = sim.sol_v

    color = cm.get_cmap(colormap)
    shades = [color(1./x) for x in range(i, 0, -1)]

    #plot snapshots of the simulation with time intervals. 
    #small offset (len(y) - 5) to make sure 5*t_interval is less than len(y)
    t_interval = int((1./i)*(len(y) - 5))
    if var in ['c', 'v']:
        for j in range(i):
            ax.plot(x, y[t_interval*j + 5], label = '{:.0f}'.format(np.ceil(sim.t[t_interval*j + 5])), color = shades[j])
    elif var == 'cgrad':
        for j in range(i):
            temp = [sim.Omega*(y[t_interval*j + 5, k] - y[t_interval*j + 5, k-1])/(x[k] - x[k-1]) for k in range(1, len(x))]
            ax.plot(x[1:], temp, label = '{:.0f}'.format(np.ceil(sim.t[t_interval*j + 5])), color = shades[j], ls = '--')
            
    plt.xlabel("x")
    #make legend
#     lgd = plt.legend(title = 't', bbox_to_anchor=(-0.2, 3), ncol=1)

def get_front_speed(sim, plot = False, ax = None):
    """
    calculates the speed of the wavefront, defined as the point where c = some constant between 0 and 1.
    NB: this function assumes that the speed is constant, it does not check whether
    this is the case! Use plot = True to check graphically.
    """
    tmin = int(len(sim.t) - 0.2*len(sim.t))
    tmax = int(len(sim.t) - 0.1*len(sim.t))


    #define the wavefront as the rightmost location where c = 1/2
    front_loc = [sim.x[np.where(sim.sol_c[i] > 0.5)[0][-1]] for i in range(len(sim.t))]
    if any([fl > (sim.x_max - 10)  for fl in front_loc]):
        t_hit = sim.t[np.where(np.array(front_loc) > (sim.x_max - 10))[0][0]]
        print("warning, front nearing end of box at time {:}".format(t_hit))
        if t_hit < sim.t[tmax]:
            print("valid front speed cannot be calculated")
            return
        else:
            print("valid front speed can still be calculated; check fitted region of curve using plot=True") 
  
    z, pcov = curve_fit(lin_fxn, sim.t[tmin:tmax], front_loc[tmin:tmax])

    if plot:
        if ax == None:
            fig, ax = plt.subplots()
        ax.scatter(sim.t, front_loc)
        ax.plot(sim.t[tmin:tmax], lin_fxn(sim.t[tmin:tmax], *z), c = 'red')
        #ax.set_ylabel("front position")
        #ax.set_xlabel("t")
        #ax.set_title(title)

    return z[0]

def lin_fxn(x, m, b):
    return m*x + b
