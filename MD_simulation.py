import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import newaxis
from os.path import dirname, exists
from os import makedirs
from numpy import empty, zeros
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor

"""
=================================================================================================
                     PROGRAM STRUCTURE

==============             ======                =============
| Thermostat | ----------- | MD | -------------- | Potential |
==============             ======                =============
                             |
                          ========
                          | main |
                          ========

- The MD class contains the primary logic of the algorithm
- Thermostat and Potential class are separated from it to allow different
thermostats and potentials can be implemented by inheritance and used flexibly.
can be used
- main() calls MD with the parameters needed for the task parts
=================================================================================================
"""

#################### GLOBAL VARIABLES
epsilon=1
sigma=1
kb=1
number_dimensions=2
mass=1
inf_radius_interaction=2*sigma
#####################


# ================================ Potential-class ================================================

# Virtual class from which concrete potentials can be inherited
# (Here only Lennard-Jones necessary, but so you can quickly implement other potentials).
class Potential:
    def V(r2):
        None
    def F(r):
        None

class PotentialLJ(Potential):
    def V(self,r2):
        # gets squared distance, returns potential
        return 4*epsilon*((sigma**12/r2**6)-(sigma**6/r2**3))

    def F(self,r):
        # gets position returns force
        r2 = np.sum(r**2,axis=len(r.shape)-1)
        factor = (48/r2**6-24/r2**3)
        if np.isscalar(r2): return r/r2* factor 
        else: return r/r2[...,newaxis] * factor[...,newaxis] 

# ------------------------------ End of Potential-class -------------------------------------------

# ================================ Thermostat class ===============================================

# Virtual class from which concrete thermostats can be inherited
class Thermostat:
    def __init__(self,T_target = None):
        self.T_target = T_target 

    def rescale(self,v, T):
        # gets list of 2d Vectors and temperature
        None

# No thermostat
class NoThermostat(Thermostat):
    def rescale(self,v, T):
        # gets list of 2d Vectors and temperature
        # not doing anything on the velocities
        return v

# Isokinetic thermostat for task d)
class IsokinThermostat(Thermostat):
    def rescale(self, v, T):
        # gets list of 2d Vectors and temperature
        # returns new velocities 
        T_measured=(1/(kb*(2*v.shape[0]-2)))*np.sum(v**2)
        v*=(T/T_measured)**(1/2)
        return v
# ------------------------------ End of Thermostat class ------------------------------------------
#

# ================================ Molecular Dynamics class ===============================================
class MD:
    def __init__(self, L, N, T, potential, thermostat, numBins, name, radius_interaction):
        self.radius_interaction=radius_interaction
        self.N = N
        self.L = L
        self.T = T

        self.numBins = numBins
        self.bin_edges = np.linspace(0,self.L/2,self.numBins+1)**2

        self.potential = potential()
        self.thermostat = thermostat()

        self.name=name

        self.t = 0

        self.r_g = 0
       
        n_side = int(np.ceil(np.sqrt(self.N)))
        grid_spacing = self.L / (n_side)
        # savign positions
        self.r = np.array([[i * grid_spacing, j * grid_spacing] for i in range(n_side) for j in range(n_side)])[:self.N]
        
        _ = self._calc_DistanceVectors()
        assert np.all(self.r2[self.r2 < 1e-4] == np.inf), "Particles too close!"

        self.v = np.random.uniform(-1,1,self.r.shape)
        self.v = np.random.uniform(-1,1,self.r.shape)
        self.v -= self._calc_vCOM()
        self.v *= np.sqrt(self.T / ((np.sum(self.v**2)/((2*self.N-2)* kb))))
        
        #print(self.v)

        # to understand how many imaginary particles to consider
        self.numbersubvolumes=np.ceil(self.radius_interaction/self.L).astype(int)

        # to speed up particles observable calculations
        self.indices_ij_upper = [[j for j in range(i+1,self.N-1)] for i in range(self.N)]
        self.indices_ij = np.array([j for i in range(self.N) for j in range(self.N) if i!=j]).reshape(self.N,self.N-1)
        self.indices_i = np.arange(self.N,dtype=int)
        
        #periodic images (if the interaction is major than L/2)
        #self.translations = np.array([[i, j] for i in range(-self.numbersubvolumes, self.numbersubvolumes+1)
        #    for j in range(-self.numbersubvolumes, self.numbersubvolumes+1)],dtype=int).reshape(-1, number_dimensions)

        # translate back into the simulation volume the particles with position outside of it 
        #self.r-=np.floor(self.r[self.indices_i,:]/self.L).astype(int)*self.L
        ## boundary conditions
        self.r = self.r - self.L * np.round(self.r / self.L)

        # initial acceleration
        self._calc_Acc()

        
    # Integration without data acquisition for pure equilibration
    def equilibrate(self, dt, n):
        for _ in range(n):
            self._verlet_step(dt) 
        # no return

    # Integration with data acquisition
    def measure(self, dt, n, radial_distribution_flag):
        self._init_data(n)
        self._update_data(0,radial_distribution_flag)
        
        for i in range(1,n+1):
            #print(i/n*100,"%")
            self._verlet_step(dt)
            self._update_data(i,radial_distribution_flag)
        
        if radial_distribution_flag:
            self.data["g"]*=2/n 
            delta_V=2*np.pi*(self.bin_edges[1:]**2-self.bin_edges[:-1]**2)
            self.data["g"]*=self.L**2/(delta_V*(self.N)**2)

        self._save()

    def _init_data(self, n):
        self.data = {
            "N": self.N,
            "numBins": self.numBins,
            "L": self.L,
            "T_0": self.T,
            "t": empty(n+1 , dtype="float32"),
            "T": empty(n+1 , dtype="float32"),
            "Ekin": empty(n+1 , dtype="float32"),
            "Epot": empty(n+1 , dtype="float32"),
            "vCOMx": empty(n+1 , dtype="float32"),
            "vCOMy": empty(n+1 , dtype="float32"),
            "g": zeros(self.numBins, dtype="float32"),
            "r": empty((n+1, *self.r.shape), dtype="float32")
        }

    def _update_data(self,i,radial_distribution_flag):
        #saves current state in self.data
        self.data["t"][i]=i
        self.data["Ekin"][i]=self._calc_Ekin()
        self.data["T"][i]=self._calc_T(self.data["Ekin"][i])
        self.data["Epot"][i]=self._calc_Epot()
        center_of_mass=self._calc_vCOM()
        self.data["vCOMx"][i]=center_of_mass[0]
        self.data["vCOMy"][i]=center_of_mass[1]
        self.data["r"][i]=np.copy(self.r)
        if radial_distribution_flag:
            #r_g=r_g[np.linalg.norm(r_g)>inf_radius_interaction]
            r2_upper = self.r2[np.triu_indices(self.N, k=1)]
            self.data["g"] += np.histogram(r2_upper, bins=self.bin_edges)[0]

    def _save(self):
        path = dirname(self.name)
        if path != "" and not exists(path):
            makedirs(path)
        np.save(self.name,self.data)

    def _calc_T(self, Ekin=None):
        if Ekin is not None:
            return 2*Ekin/(2*self.N-2) * kb
        else:
        # returns temperature of current state
            return (1/((2*self.N-2)*kb))*np.sum(self.v**2)

    def _calc_Ekin(self):
        # returns kinetic energy
        return (1/2)*np.sum(self.v**2)
       
    def _calc_Epot(self):
        # gets squared distance, returns potential energy
        return np.sum(self.potential.V(self.r2[self.r2<=self.radius_interaction**2]))

    def _calc_vCOM(self):
        # returns center of mass velocity 
        return np.sum(self.v,axis=0)/self.N

    def _calc_DistanceVectors(self):
        # returns the distance vectors between all particles
        # shape(return) = (N, N-1, 2)
        #r=np.array([self.r[i,:]-self.r[self.indices_ij[i,:],:] for i in range(self.N)]).reshape(self.N,self.N-1,number_dimensions)
        r=np.array([self.r[i,:]-self.r[j,:] for i in range(self.N) for j in range(self.N)]).reshape(self.N,self.N,number_dimensions)
        #r = (self.r[:,newaxis]-self.r)[self._lower_triangle_mask]
        self.r2=np.sum(r**2,axis=r.ndim-1)
        np.fill_diagonal(self.r2, np.inf)
        self.r2[self.r2 < inf_radius_interaction] = np.inf
        #np.fill_diagonal(self.r2<inf_radius_interaction,np.inf)
        #print(self.r2)
        return r

    #def _calc_Acc(self):
    #    # calcualtes accelaration and either stores it in self.a or returns it     
    #    # calculating all possible particle distances including lattice translations and considering only the ones that are inside the interaction radius
    #    # calculating the interaction with the imaginary particles (number dimensions is explicitly 2 here, can be easily generalized)
    #    # creating a list of distances_ij considering translations of the unit cell (translation 00 corresponds to interactions between real particles)
    #    distances_ij=self._calc_DistanceVectors()
    #    force=np.zeros((self.N,self.N-1,number_dimensions))
    #    #for i in range(self.translations.shape[0]):
    #    #translated_distances_ij=distances_ij+translation[i,:]
    #    translated_distances_ij=distances_ij-self.L*np.round(distances_ij/self.L)
    #    modulus_translated_distances_ij=np.linalg.norm(translated_distances_ij,axis=2)
    #    condition1=modulus_translated_distances_ij<self.radius_interaction
    #    condition2=modulus_translated_distances_ij>=inf_radius_interaction
    #    upper_mask=np.triu(np.ones_like(modulus_translated_distances_ij,dtype=bool),k=1)&condition1&condition2
    #    force[upper_mask]=self.potential.F(translated_distances_ij[upper_mask])
    #    idx,jdx=np.where(upper_mask)
    #    force[jdx,idx]=-force[idx,jdx]
    #    self.a=(1/mass)*np.sum(force,axis=1)

    def _calc_Acc(self):
        
        force=np.zeros((self.N,self.N,number_dimensions))
        r = self._calc_DistanceVectors()

        ## boundary conditions
        r = r - self.L * np.round(r / self.L)

        upper_mask=np.triu(np.ones_like(self.r2,dtype=bool),k=1)*(self.r2 <= self.radius_interaction**2).astype(bool)
        #mask=condition1&condition2
        force[upper_mask]=self.potential.F(r[upper_mask])
        idx,jdx=np.where(upper_mask)
        force[jdx,idx]=-force[idx,jdx]
        
        self.a = float(1/mass)*np.sum(force,axis=1)
        #r, self.r2 = self._calc_DistanceVectors()
        #F = np.zeros((self.N,self.N-1,2))
        #CutOff = self.r2<self.radius_interaction**2
        #print(self.r2.shape)
        #print(CutOff.shape)
        #F[(self._lower_triangle_mask[0][CutOff],self._lower_triangle_mask[1][CutOff])] = self.potential.F(r[self.r2<self.radius_interaction**2])
        #np.transpose(F,axes=[1,0,2])[(self._transposed_lower_triangle_mask[0][CutOff],self._transposed_lower_triangle_mask[1][CutOff])] = -F[self._lower_triangle_mask][CutOff]
        #self.a = np.sum(F,axis=1)


    def _verlet_step(self, dt):
        #print(self.a,self.v)
        self.v += 0.5*self.a*dt
        # performs a single interaction with the verlet algorithm
        self.r += self.v*dt
        # translate back into the simulation volume the particles with position outside of it 
        #self.r-=np.floor(self.r[self.indices_i,:]/self.L).astype(int)*self.L
        ## boundary conditions
        self.r = self.r - self.L * np.round(self.r / self.L)
        # calculate velocity step
        self._calc_Acc()
        self.v += 0.5*self.a*dt
        self.v=self.thermostat.rescale(self.v,self.T)
    
    ### plotting dynamics
    def plot_r(self, interval=200, trail=60):
        history = np.array(self.data["r"]).reshape(-1, self.N, number_dimensions)
        num_frames = history.shape[0]

        fig, ax = plt.subplots()
        ax.set_xlim(-1 - self.L/2  , self.L/2  + 1)
        ax.set_ylim(-1 -self.L/2 , self.L/2  + 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Create trail lines for each particle
        trails = [ax.plot([], [], 'b-', alpha=0.3)[0] for _ in range(self.N)]
        dots = [ax.plot([], [], 'bo')[0] for _ in range(self.N)]

        def update(frame):
            for i in range(self.N):
                # Current position                y = history[frame, i, 1]
                dots[i].set_data([history[frame, i, 0]], [history[frame, i, 1]])
            return dots 

        ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
        plt.tight_layout()
        plt.show()
# ------------------------------ End of Molecular Dynamics class ------------------------------------------
#

#################### PLOTTING QUANTITIES RELATED TO THREE RANGES OF TEMPERATURES
def plot_3T(radial_distribution_flag,show=False):
    if not exists("Plots"):
       makedirs("Plots")
    md_rT001 = np.load("Data/3_md_rT0.01.npy",allow_pickle=True)[None][0]
    md_rT1 = np.load("Data/3_md_rT1.npy",allow_pickle=True)[None][0]
    md_rT100 = np.load("Data/3_md_rT100.npy",allow_pickle=True)[None][0]
    #md_rT1 = np.load("Data/1_md_rT1.npy",allow_pickle=True)[None][0]
    
    def plot_E(*args):
        for data in args:
            t = data["t"]
            plt.plot(t, data["Ekin"], label="Ekin")
            plt.plot(t, data["Epot"], label="Epot")
            plt.plot(t, data["Ekin"]+data["Epot"], label="Etot")
            plt.legend(loc="best",title="")
            plt.title(f"Energy Distribution, T_0 = {data['T_0']}")
            plt.xlabel("$t$")
            plt.ylabel("$E$")
            plt.tight_layout()
            plt.savefig(f"Plots/E_n{int(np.sqrt(data['N']))}.pdf")
            if show: plt.show()
            else: plt.close()
    
    def plot_T(*args):
        for data in args:
            t = data["t"]
            plt.plot(t, data["T"], label="T")
            plt.legend(loc="best",title="")
            plt.title(f"Temperature Distribution, T_0 = {data['T_0']}")
            plt.xlabel("$t$")
            plt.ylabel("$T$")
            plt.tight_layout()
            plt.savefig(f"Plots/T_n{int(np.sqrt(data['N']))}.pdf")
            if show: plt.show()
            else: plt.close()

    def plot_CM(*args):
        for data in args:
            t = data["t"]
            plt.plot(t, data["vCOMx"], label="vCOMx")
            plt.plot(t, data["vCOMy"], label="vCOMy")
            plt.legend(loc="best",title="")
            plt.title(f"CM velocity distribution, T_0 = {data['T_0']}")
            plt.xlabel("$t$")
            plt.ylabel("$vCOM(xy)$")
            plt.tight_layout()
            plt.savefig(f"Plots/vCOM_n{int(np.sqrt(data['N']))}.pdf")
            if show: plt.show()
            else: plt.close()

    if radial_distribution_flag:
        def plot_g(*args):   
            for data in args:
                plt.plot(np.linspace(0,np.sqrt(data['N']),len(data["g"])),data["g"], label=f"$T_0 = {data['T_0']}$")
            plt.legend(loc="best",title="")
            plt.title(f"Pair Correlation Function (N = {args[0]['N']})")
            plt.xlabel("$r$")
            plt.ylabel("$g(r)$")
            plt.tight_layout()
            plt.savefig(f"Plots/PairCorr_n{int(np.sqrt(args[0]['N']))}.pdf")
            if show: plt.show()
            else: plt.close()

    #plot_E(md_rT1)
    #plot_T(md_rT1)
    #plot_CM(md_rT1)
    #plot_g(md_rT1)
    plot_E(md_rT001,md_rT1,md_rT100)
    plot_CM(md_rT001,md_rT1,md_rT100)
    plot_T(md_rT001,md_rT1,md_rT100)
    plot_g(md_rT001,md_rT1,md_rT100)#


#################### MAIN FUNCTION
def main():
    
    numBins = 500 # Number of bins for pair correlation function

    # b) Equilibration test
    N = 16
    L = 10
    #N = 256
    #L = 32
    name="Data/1_md_rT"
    do_b = False
    radial_distribution_flag=False
    if do_b:
        T = 1
        dt = 0.0001
        steps = 10000
        md = MD(L, N, T, PotentialLJ, NoThermostat, numBins, f"{name}{T}",L/2)
        md.equilibrate(dt, steps)
        md.measure(dt, steps, radial_distribution_flag)
        plot_3T(radial_distribution_flag,True)
            

    # c) Pair correlation function
    N=16
    L=10
    name="Data/2_md_rT"
    do_c = False
    radial_distribution_flag=False
    if do_c:
        for T in [0.01, 1, 100]:
            dt = 0.01
            equiSteps = int(10000)
            steps = int(1000)
            md = MD(L, N, T, PotentialLJ, NoThermostat, numBins, f"{name}{T}",L/2)
            print("equilibration")
            md.equilibrate(dt, equiSteps)
            print("measure")
            md.measure(dt, steps, radial_distribution_flag)
            md.plot_r()
        plot_3T(radial_distribution_flag,True)

    # d) Thermostat
    N=10
    L=10
    name="Data/3_md_rT"
    do_c = True
    radial_distribution_flag=True
    if do_c:
        for T in [0.01, 1, 100]:
            dt = 0.001
            equiSteps = int(10000)
            steps = int(1000)
            md = MD(L, N, T, PotentialLJ, IsokinThermostat, numBins, f"{name}{T}",L/2)
            print("equilibration")
            md.equilibrate(dt, equiSteps)
            print("measure")
            md.measure(dt, steps,radial_distribution_flag)
            #md.plot_r()
        plot_3T(radial_distribution_flag,True)

if __name__ == "__main__":
    main()
