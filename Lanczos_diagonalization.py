import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import make_interp_spline
from matplotlib.gridspec import GridSpec

# Function printing a generic matrix
def print_matrix(matrix, fmt="{:6.2f}"):
    for row in matrix:
        print(" ".join(fmt.format(elem) for elem in row))

# Function calculating the distance of a generic (symmetric)-matrix from a diagonal matrix
# addition_i points to the minimum element to consider in the calculation of the offset
def T_dist_from_diag(matrix,addition_i):
    #print("offset",addition_i,matrix)
    offset=0
    for i in range(addition_i,np.shape(matrix)[0]-1):
        for j in range(i+1,np.shape(matrix)[0]):
            offset+=2*(matrix[i,j])**2
    return offset

# Function finding the out-of-diagonal indices of a (symmetric)-matrix with maximum absolute value
# addition_i points to the minimum element to consider in the search
def max_min_indices_symm(matrix,addition_i):
    i_max,j_max,max_value=addition_i,addition_i+1,np.abs(matrix[addition_i,addition_i+1])
    for i in range(addition_i,np.shape(matrix)[0]-1):
        for j in range(i+1,np.shape(matrix)[0]):
            if np.abs(matrix[i,j])>max_value:
                i_max=i
                j_max=j
                max_value=np.abs(matrix[i,j])
    return i_max,j_max

# Defining the Krylov space from an initial vector
# Taking into account a number of iterations and a threshold 
# In case the Krylov space is not empy, it is enlarged
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
# In case the matrix is already partially built the new entries are added
# The position from which the matrix is enlarged is given in return
def building_tridiagonal_matrix(tridiagonal_matrix,matrix,krylov_space):
    #print(np.shape(tridiagonal_matrix),np.shape(krylov_space))
    if tridiagonal_matrix is None:
        addition_i=0
        tridiagonal_matrix=np.zeros((np.shape(krylov_space)[0],np.shape(krylov_space)[0]))
    else:
        addition_i=np.shape(tridiagonal_matrix)[0]-1  
        #print_matrix(tridiagonal_matrix)
        enlarged_tridiagonal_matrix=np.zeros((np.shape(krylov_space)[0],np.shape(krylov_space)[0]))
        enlarged_tridiagonal_matrix[:addition_i+1,:addition_i+1]=tridiagonal_matrix
        tridiagonal_matrix=enlarged_tridiagonal_matrix
        #print_matrix(tridiagonal_matrix)
    
    i=addition_i
    #print(addition_i,i)
    while i < np.shape(krylov_space)[0]-1:
        gammai=np.sqrt(np.dot(krylov_space[i,:],krylov_space[i,:]))
        tridiagonal_matrix[i,i]=(krylov_space[i,:].T)@matrix@krylov_space[i,:]/gammai**2
        gammaiplus=np.sqrt(np.dot(krylov_space[i+1,:],krylov_space[i+1,:]))
        tridiagonal_matrix[i,i+1]=(krylov_space[i,:].T)@matrix@krylov_space[i+1,:]/(gammai*gammaiplus)
        tridiagonal_matrix[i+1,i]=tridiagonal_matrix[i,i+1]
        i+=1
    #print_matrix(tridiagonal_matrix)   
    #print(i,np.shape(tridiagonal_matrix),addition_i)
    gamma=np.sqrt(np.dot(krylov_space[-1,:],krylov_space[-1,:]))
    tridiagonal_matrix[-1,-1]=(krylov_space[-1,:].T)@matrix@krylov_space[-1,:]/gamma**2
    #print_matrix(tridiagonal_matrix)

    return addition_i,tridiagonal_matrix

# If addition_i !=0 only the part of the matrix with i,j > addition_i,addition_i is considered in the Jacobi rotation
# The Jacobi rotation is performed until the offset (with respect to a diagonal matrix) is below the given threshold 
def Jacobi_rotation(matrix,transformation,addition_i,threshold_diag):
    # addition_i is an integer so no problem should result from this condition
    if addition_i == 0:
        transformation=np.identity((np.shape(matrix)[0]))
    else:
        enlarged_transformation=np.identity(np.shape(matrix)[0])
        enlarged_transformation[:addition_i+1,:addition_i+1]=transformation
        transformation=enlarged_transformation

    offset=T_dist_from_diag(matrix,addition_i)
    #print("offset",offset)
    while offset>threshold_diag:
        z=np.identity(np.shape(matrix)[0])
        # looking for largest off-diagonal element (considering symmetric matrix)
        i_max,j_max=max_min_indices_symm(matrix,addition_i)
        # defining z matrix so that a_{i_max,j_max}=0 and a_{j_max,i_max}=0
        if matrix[i_max,j_max]!=0:
            omega=(matrix[j_max,j_max]-matrix[i_max,i_max])/(2*matrix[i_max,j_max])
        else:
            omega=0
        sign = np.sign(omega) if omega != 0 else 1
        theta=np.arctan(sign/(np.abs(omega)+np.sqrt(1+omega**2)))
        z[i_max,i_max]=np.cos(theta)
        z[j_max,j_max]=z[i_max,i_max]
        z[i_max,j_max]=np.sin(theta)
        z[j_max,i_max]=-z[i_max,j_max]
        matrix=z.T@matrix@z
        #print_matrix(z)
        #print_matrix(transformation)
        transformation=transformation@z
        offset=T_dist_from_diag(matrix,addition_i)
    return matrix,transformation

# Lanczos algorithm
def Lanczos_method(matrix,max_dimension,threshold_krylov,initial_vector,threshold_diag,threshold_lanczos):
    print("Begin Lanczos procedure...")
    convergence=threshold_lanczos+1
    iteration_lanczos=0
    krylov_space=None
    tridiagonal_matrix=None
    transformation=None
    while convergence > threshold_lanczos:
        
        # determining/enlarging the Lanczos basis (Krylov space)
        enlarged,krylov_space=building_krylov_space(krylov_space,matrix,max_dimension,threshold_krylov,initial_vector)
        #print(" dimension krylov space: ", np.shape(krylov_space)[0])
        # if the Krylov space can not be enlarged (the threshold on the krylov basis building procedure can be increased) 
        if enlarged is False:
            break
        addition_i,tridiagonal_matrix=building_tridiagonal_matrix(tridiagonal_matrix,matrix,krylov_space)

        eigenvalues,eigenvectors=np.linalg.eig(tridiagonal_matrix)
        tridiagonal_matrix=np.diag(eigenvalues)
        transformation=eigenvectors
        #tridiagonal_matrix,transformation=Jacobi_rotation(tridiagonal_matrix,transformation,addition_i,threshold_diag)

        # considering the ground-state energy
        min_i=np.argmin(np.diag(tridiagonal_matrix))
        new_ground_state_eigv=tridiagonal_matrix[min_i,min_i]
        new_ground_state_eigf=transformation[:,min_i]
        #print("ground state energy: ", new_ground_state_eigv)
        
        print("Number of the step:", iteration_lanczos+1)
        print("Dimension of the Krylov space:", np.shape(krylov_space)[0])
        print("Ground-state energy:", new_ground_state_eigv)

        if iteration_lanczos > 0:
        # one additional iteration (krylov space is enlarged) is considered if the difference between the two ground-state energies is lower than the threshold
            convergence=np.abs(new_ground_state_eigv-old_ground_state_eigv)
            print(old_ground_state_eigv,new_ground_state_eigv)
            print("Difference in ground-state energy:", convergence)
        
        old_ground_state_eigv=new_ground_state_eigv
        iteration_lanczos+=1
        max_dimension+=1
    # the ground-state eigenfunction is in terms of the krylov space; here we express it in terms of the initial space
    original_ground_steate_eigf=np.zeros(len(initial_vector))
    for i in range(len(initial_vector)):
        for j in range(np.shape(krylov_space)[0]):
            original_ground_steate_eigf[i]+=new_ground_state_eigf[j]*krylov_space[j,i]/np.sqrt(np.dot(krylov_space[j,:],krylov_space[j,:]))
    #print("ground state wavefunction: ",new_ground_state_eigf, original_ground_steate_eigf)
    # calculating particle density
    particle_density=np.zeros((len(initial_vector)))
    for i in range(len(initial_vector)):
        particle_density[i]=original_ground_steate_eigf[i]**2
    # normalization of particle density
    particle_density=particle_density/np.sqrt(np.dot(particle_density,particle_density))
    print("End Lanczos procedure!")
    return new_ground_state_eigv,particle_density

def plot_particle_density(number_sites, defect_state_position,ground_state_particle_densities,ground_state_energies,defect_state_energies):

    sites = np.arange(number_sites)
    num_states = len(ground_state_particle_densities)
    offset = 1.5
    color_map = cm.get_cmap('viridis', num_states)

    # Create layout with legend on the right
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, width_ratios=[12, 1.5], wspace=0.01)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    legend_handles = []

    for idx, (density, gs_energy, def_energy) in enumerate(zip(ground_state_particle_densities,
                                                                ground_state_energies,
                                                                defect_state_energies)):
        shift = np.array(density) + idx * offset + 0.3
        x_smooth = np.linspace(sites.min(), sites.max(), 500)
        y_smooth = make_interp_spline(sites, shift, k=3)(x_smooth)
        color = color_map(idx)

        ax.plot(x_smooth, y_smooth - 0.05, color='gray', lw=2.5, alpha=0.3)
        line, = ax.plot(x_smooth, y_smooth, color=color, lw=2,
                        label=f"State {idx}: E_gs={gs_energy:.2f}, E_def={def_energy:.2f}")
        legend_handles.append(line)

    for i in sites:
        ax.add_patch(plt.Circle((i, 0), 0.2,
                                color='red' if i == defect_state_position else 'skyblue',
                                ec='black', zorder=3))

    ax.plot([defect_state_position - 0.3, defect_state_position + 0.3],
            [-0.25, -0.25], color='red', lw=2)

    # Annotation moved lower to prevent overlap
    ax.annotate('Defect Site',
                xy=(defect_state_position, -0.25),
                xytext=(defect_state_position, -1.7),  # Lowered
                arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5),
                ha='center', fontsize=10, color='red')

    # Give more space below the x-axis
    ax.set_xlim(-1, number_sites)
    ax.set_ylim(-2.0, num_states * offset + 1.5)  # Enlarged vertical space
    ax.set_xticks(sites)
    ax.set_yticks([])
    ax.set_xlabel("Site Index")
    ax.set_title("Particle Densities")
    ax.grid(True, linestyle='--', alpha=0.3)

    ax_leg.legend(handles=legend_handles,
                  loc='center left',
                  fontsize=9,
                  frameon=False)

    plt.tight_layout()
    filename="particle_densities.png" 
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
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

    # Initial dimension of the Krylov basis (for the case max_dimensions==1 some additional conditions in the different functions should be added)
    max_dimension=2
    # Initial threshold on the Krylov basis building
    threshold_krylov=1.0e-15
    # Initial vector used for the Krylov basis building (normalized inside the Krylov basis building function)
    initial_vector=np.random.rand(number_sites)
    # Threshold on the distance between the matrix and its diagonal form
    threshold_diag=1.0e-7
    # Threshold over the ground-state energies obtained from two different cycles of the Lanczos procedure
    threshold_lanczos=1.0e-6

    # Considering different energies of the defect state
    defect_state_energies=np.arange(-20,20,2) 
    # Saving the ground-state energies and electronic densities
    ground_state_energies=np.zeros((len(defect_state_energies)))
    ground_state_particle_densities=np.zeros((len(defect_state_energies),number_sites))

    for i in range(len(defect_state_energies)):
        
        # Adding the defect state
        hamiltonian[defect_state_position,defect_state_position]=defect_state_energies[i]
        
        # Evaluation of ground-state energy and electronic density
        ground_state_energies[i],ground_state_particle_densities[i,:]=Lanczos_method(hamiltonian,max_dimension,threshold_krylov,initial_vector,threshold_diag,threshold_lanczos)

        #print("The final ground-state energy is:", ground_state_energies[i])
    plot_particle_density(number_sites,defect_state_position,ground_state_particle_densities,ground_state_energies,defect_state_energies)

if __name__ == "__main__":
    main()