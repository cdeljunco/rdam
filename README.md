### Reaction-diffusion active matter (RDAM)



This repository contains the code used for all numerical results in the paper "Front speed and pattern selection of a propagating chemical front in an active fluid" by del Junco, Estevez-Torres, and Maitra. Contents:

- **active_rd_simulation/** contains a simple Python package for simulations and analysis of the system studied in the paper. To install the package, navigate to this folder in a Linux terminal and type `pip install -e .`. The `-e` makes sure any edits are updated in the appropriate place in your Python library.
  - Note: this package writes and saves simulation files to a folder rdam/Data/. You'll need to create that folder or it will be confused.
- **active-rd-analysis.ipynb** is the jupyter notebook where all the simulations and analysis were run (except for the ones using MATLAB...). 
- **get_kmin.m** and **particle.m** are MATLAB scripts that were used to integrate eq. 11 of the paper and find the minimum front speed yielding trajectories with $c \geq 0$.
- All other files in rdam/ and rdam/Plots are produced by the scripts above.

