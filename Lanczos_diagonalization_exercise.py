import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec


# Defining the Krylov space from an initial vector
# Taking into account a number of iterations and a threshold 
# In case the Krylov space is not empy, it is enlarged
# saving in the Krylov space the vectors v_i not the q_i
def building_krylov_space(krylov_space,matrix,max_dimension,threshold_krylov,initial_vector):
    identity=np.identity(np.shape(matrix)[0])
    enlarged=False

    if krylov_space is None:
        count_dimension=0
    else:
        count_dimension=np.shape(krylov_space)[0]-1

    while count_dimension<max_dimension-1:
        if krylov_space is None:
            krylov_space=np.zeros((1,len(initial_vector)))
            krylov_space[0,:]=initial_vector/np.sqrt(np.dot(initial_vector,initial_vector))
            prec_krylov_space_vector=np.zeros(len(initial_vector))
        else:
            prec_krylov_space_vector=krylov_space[-2,:]/np.sqrt(np.dot(krylov_space[-2,:],krylov_space[-2,:]))

        gamma=np.sqrt(np.dot(krylov_space[-1,:],krylov_space[-1,:]))
        delta=(krylov_space[-1,:].T)@matrix@krylov_space[-1,:]/gamma**2
        next_element=((matrix-identity*delta)@krylov_space[-1,:]/gamma)-gamma*prec_krylov_space_vector
        next_element_norm=np.sqrt(np.dot(next_element,next_element))

        if(next_element_norm>threshold_krylov):
            krylov_space=np.vstack([krylov_space,next_element])
            enlarged=True
            count_dimension+=1
        else:
            break

    return enlarged,krylov_space

# Representing the matrix in the Krylov space
def building_tridiagonal_matrix(matrix,krylov_space):
    tridiagonal_matrix=np.zeros((np.shape(krylov_space)[0],np.shape(krylov_space)[0]))
    i=0
    
    while i < np.shape(krylov_space)[0]-1:
        gammai=np.sqrt(np.dot(krylov_space[i,:],krylov_space[i,:]))
        tridiagonal_matrix[i,i]=(krylov_space[i,:].T)@matrix@krylov_space[i,:]/gammai**2
        gammaiplus=np.sqrt(np.dot(krylov_space[i+1,:],krylov_space[i+1,:]))
        tridiagonal_matrix[i,i+1]=(krylov_space[i,:].T)@matrix@krylov_space[i+1,:]/(gammai*gammaiplus)
        tridiagonal_matrix[i+1,i]=tridiagonal_matrix[i,i+1]
        i+=1

    tridiagonal_matrix[-1,-1]=(krylov_space[-1,:].T)@matrix@krylov_space[-1,:]/np.dot(krylov_space[-1,:],krylov_space[-1,:])
    return tridiagonal_matrix

# Lanczos algorithm
def Lanczos_method(matrix,max_dimension,threshold_krylov,initial_vector,threshold_lanczos):
    # In order not to be trivial, the krylov base has to have at least two elements (the case one can be considered adding some conditioning)
    if(max_dimension==1):
        max_dimension+=1
    #print("Begin Lanczos procedure...")
    convergence=threshold_lanczos+1
    iteration_lanczos=0
    krylov_space=None
    tridiagonal_matrix=None
    while convergence > threshold_lanczos:

        # Building Krylov space 
        enlarged,krylov_space=building_krylov_space(krylov_space,matrix,max_dimension,threshold_krylov,initial_vector)
        
        if enlarged is False:
            #print("Not enlarged")
            break

        tridiagonal_matrix=building_tridiagonal_matrix(matrix,krylov_space)
    
        # Diagonalizing hamiltonian
        eigenvalues,eigenvectors=np.linalg.eig(tridiagonal_matrix)

        # Considering the ground-state energy
        min_i=np.argmin(eigenvalues)
        #print(eigenvalues)
        new_ground_state_eigv=eigenvalues[min_i]
        new_ground_state_eigf=eigenvectors[:,min_i]
        
        #print("Number of the step:", iteration_lanczos)
        #print("Dimension of the Krylov space:", np.shape(krylov_space)[0])
        #print("Ground-state energy:", new_ground_state_eigv)

        if iteration_lanczos > 0:
        # One additional iteration (krylov space is enlarged) is considered if the difference between the two ground-state energies is lower than the threshold
            convergence=np.abs(new_ground_state_eigv-old_ground_state_eigv)
            #print(old_ground_state_eigv,new_ground_state_eigv)
            #print("Difference in ground-state energy:", convergence)
        
        old_ground_state_eigv=new_ground_state_eigv
        iteration_lanczos+=1
        max_dimension+=1

    # Projecting the ground-state eigenvector into the initial basis set
    final_ground_state_eigv=np.zeros(len(initial_vector))
    for i in range(len(initial_vector)):
        for j in range(np.shape(krylov_space)[0]):
            final_ground_state_eigv[i]+=new_ground_state_eigf[j]*krylov_space[j,i]/np.sqrt(np.dot(krylov_space[j,:],krylov_space[j,:]))
    # Calculating particle density
    particle_density=np.zeros((len(initial_vector)))
    for i in range(len(initial_vector)):
        particle_density[i]=final_ground_state_eigv[i]**2
    # Normalization of particle density
    particle_density=particle_density/np.sqrt(np.dot(particle_density,particle_density))
    
    #print("End Lanczos procedure!")
    return new_ground_state_eigv,particle_density

# Plotting function
def plot_particle_density(number_sites, defect_state_position,ground_state_particle_densities,ground_state_energies,defect_state_energies):

    sites = np.arange(number_sites)
    num_states = np.shape(ground_state_particle_densities)[0]

    color_map = cm.get_cmap('viridis', num_states)

    # Create multiplot with legend as a second plot (without axis)
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, width_ratios=[12, 1.5], wspace=0.01)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    legend_lines = []

    offset = 1.5

    for idx, (density, gs_energy, def_energy) in enumerate(zip(ground_state_particle_densities,ground_state_energies,defect_state_energies)):

        shifted_density = np.array(density) + idx * offset + 0.3
        
        x = np.linspace(sites.min(), sites.max(), 500)
        y = np.interp(x,sites,shifted_density) 

        color = color_map(idx)

        ax.plot(x, y - 0.05, color='gray', lw=2.5, alpha=0.3)
        line, = ax.plot(x, y, color=color,lw=2,label=f"State {idx}: E_gs={gs_energy:.2f}, E_def={def_energy:.2f}")
        legend_lines.append(line)

    # Associating to each site a circle
    for i in sites:
        ax.add_patch(plt.Circle((i, 0), 0.2,color='red' if i == defect_state_position else 'blue',zorder=3))

    # Pointing to the defect with an arrow
    ax.annotate('Defect Site',xy=(defect_state_position, -0.25),xytext=(defect_state_position, -1.7),arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5))

    ax.set_xlim(-1, number_sites)
    ax.set_ylim(-2.0, num_states * offset + 1.5)
    ax.set_xticks(sites)
    ax.set_yticks([])
    ax.set_xlabel("Site Index")
    ax.set_title("Particle Densities")
    ax.grid(True, linestyle='--', alpha=0.3)

    ax_leg.legend(handles=legend_lines,loc='center left',fontsize=9)

    filename="particle_densities.png" 
    plt.savefig(filename, dpi=300)
    
    #plt.show()

def main():

    # Initial parameters
    number_sites=50
    hopping_parameter=1.0
    defect_state_position=int((number_sites-1)/2)

    # Building hamiltonian without defect
    hamiltonian=np.zeros((number_sites,number_sites))
    for i in range(number_sites-1):
        hamiltonian[i,i+1]=-hopping_parameter
        hamiltonian[i+1,i]=-hopping_parameter
    # Periodic boundary conditions
    hamiltonian[0,number_sites-1]=-hopping_parameter
    hamiltonian[number_sites-1,0]=-hopping_parameter

    # Initial dimension of the Krylov basis
    max_dimension=2
    # Initial threshold on the Krylov basis building
    threshold_krylov=1.0e-8
    # Initial vector used for the Krylov basis building (normalized inside the Krylov basis building function)
    initial_vector=np.zeros(number_sites)
    initial_vector[0]=1.0

    # threshold over the Lanczos procedure
    threshold_lanczos=1.0e-8

    # Considering different energies of the defect state
    defect_state_energies=np.arange(0,2,2) 
    # Saving the ground-state energies and electronic densities
    ground_state_energies=np.zeros((len(defect_state_energies)))
    ground_state_particle_densities=np.zeros((len(defect_state_energies),number_sites))

    for i in range(len(defect_state_energies)):
        
        # Adding the defect state
        hamiltonian[defect_state_position,defect_state_position]=defect_state_energies[i]
        
        # Evaluation of ground-state energy and electronic density
        ground_state_energies[i],ground_state_particle_densities[i,:]=Lanczos_method(hamiltonian,max_dimension,threshold_krylov,initial_vector,threshold_lanczos)

        #print("The final ground-state energy is:", ground_state_energies[i])
    
    plot_particle_density(number_sites,defect_state_position,ground_state_particle_densities,ground_state_energies,defect_state_energies)

if __name__ == "__main__":
    main()