#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include </usr/local/Eigen>
#include <cstdio>
#include <functional>
#include <omp.h>
#include <bits/stdc++.h>

double delta(int sigma1,int sigma2){
    if(sigma1==sigma2)
        return 1.0;
    else   
        return 0.0;
}

class multiple_electrons_basis_set{
    private:
        /// number_electrons = Q
        int number_electrons;
        /// total_spin = S
        int total_spin;
        int number_orbitals;
        std::vector<int> degeneracies;
        int dimension_of_the_basis_state;
        std::vector<std::vector<int>> basis_set;
        int number_of_basis_states;
        /// Given a certain position in the state of the basis_set, give back the corresponding orbital and spin (the positioning of the orbitals and spins is the same in all the states of the basis set)
        Eigen::MatrixXd spin_and_orbital;
    public:
    multiple_electrons_basis_set(int init_number_electrons,int init_total_spin,int init_number_orbitals,std::vector<int> init_degeneracies,int total_number_of_single_particle_states){
        number_electrons=init_number_electrons;
        total_spin=init_number_electrons;
        number_orbitals=init_number_electrons;
        degeneracies=init_degeneracies;
        ///adding spin degeneracy to A1, A2, A3, .... B1, B2, ...
        dimension_of_the_basis_state=total_number_of_single_particle_states*2;
        ///orbitals = A, B... 
        ///degeneracy of A = A1, A2...
        ///spin of A1 = A1_up, A1_down 
        ///A1_up A1_down A2_up A2_down ... B1_up
        int number_of_basis_states=0;
    };
    void filling_basis_set_conserving_Q_and_S(){
        ////possible filling of the orbitals: binomial_coefficient(dimension_of_the_basis_state number_electrons)
        /// in case of crystalline field Oh (t2g,eg), and 5 electrons : 252 states
        std::vector<int> test_element(dimension_of_the_basis_state);
        for(int i=0;i<number_electrons;i++)
            test_element[i]=1;
        for(int i=number_electrons;i<dimension_of_the_basis_state;i++)
            test_element[i]=0;
        // generate all permuatation of the initial array = conserving the total number of electrons
        int n = sizeof(test_element)/sizeof(test_element[0]);
        int spin_of_element;
        do{
            spin_of_element=0;
            for (int i = 0; i < n; i++)
                if(test_element[i]!=0){
                    if(i%2==0)
                        spin_of_element+=1;
                    else
                        spin_of_element-=1;
                }
            // considering only those states conserving the total spin momentum
            if(spin_of_element==total_spin){
                if(number_of_basis_states!=0){
                    ///adding one row
                    basis_set.resize(basis_set.size()+1);
                    basis_set[basis_set.size()-1]=test_element[i];
                }else{
                    basis_set.resize(1);
                    for (auto& row : basis_set){
                        row.resize(dimension_of_the_basis_state);
                    }
                }
                number_of_basis_states+=1;
            }
        }while(std::next_permutation(test_element,test_element+n));
    
        spin_and_orbital.resize(dimension_of_the_basis_state,2);
        int counting=0;
        for(int i=0;i<number_orbitals;i++){
            for(int j=0;j<degeneracies[i]*2;j++){
                spin_and_orbital(counting+j,1)=i;
                (spin_and_orbital(counting+j,0) % 2 == 0) ? 0.0 : 1.0;
            }
            counting+=degeneracies[i]*2;
        }
    };
    std::vector<std::vector<int>> pull_basis_set(){
        return basis_set;
    };
    int pull_dimension_of_the_basis_state(){
        return dimension_of_the_basis_state;
    };
    int pull_number_of_basis_states(){
        return number_of_basis_states;
    };
    Eigen::MatrixXd pull_spin_orbital(){
        return spin_and_orbital;
    };
}

class Hamiltonian{
    private:
        std::vector<std::vector<int>> basis_set;
        std::vector<double> E_0;
        Eigen::MatrixXd U;
        Eigen::MatrixXd spin_orbital;
        int number_of_basis_states;
        int dimension_of_the_basis_state;
        int number_orbitals;
        std::vector<int> degeneracies;
    public:
        Hamiltonian(std::vector<std::vector<int>> init_basis_set,int init_dimension_of_the_basis_state,int init_number_of_basis_states,std::vector<double> init_E_0,Eigen::MatrixXd init_U,int init_number_orbitals,std::vector<int> init_degeneracies,Eigen::MatrixXd init_spin_orbital){
            U=init_U;
            E_0=init_E_0;
            basis_set=init_basis_set;
            dimension_of_the_basis_state=init_dimension_of_the_basis_state;
            number_of_basis_states=init_number_of_basis_states;
            number_orbitals=init_number_orbitals;
            degeneracies=init_degeneracies;
            spin_orbital=init_spin_orbital;
        };
        Eigen::MatrixXd writing_in_basis_set(){
            Eigen::MatrixXd h(number_of_basis_states,number_of_basis_states);
            h.setZero();
    
            for(int i=0;i<number_of_basis_states;i++)
                for(int j=0;j<dimension_of_the_basis_state;j++)
                    h(i,i)+=E_0[j]*basis_set(i,j);

            for(int i=0;i<number_of_basis_states;i++)     
                for(int k=0;k<number_of_basis_states;k++)
                    for(int j=0;j<dimension_of_the_basis_state;j++)
                        for(int l=0;l<dimension_of_the_basis_state;l++)
                            for(int m=0;m<dimension_of_the_basis_state;m++)
                                for(int n=0;n<dimension_of_the_basis_state;n++)
                                    h(i,k)=U[spin_orbital(j,1)*number_orbitals+spin_orbital(l,1)][spin_orbital(m,1)*number_orbitals+spin_orbital(n,1)]*basis_set(k,m)*basis_set(k,n)*(1-basis_set(i,j))*(1-basis_set(i,l))*delta(spin_orbital(j,0),spin_orbital(n,0))**delta(spin_orbital(l,0),spin_orbital(m,0));
        };
}


int main(int argc,char** argv){	

///IMPURITY HAMILTONIAN EXACT DIAGONALIZATION
    int number_electrons=6;
    int total_spin=1;
    ////PARAMETERS BELOW ARE GIVEN TAKING INTO ACCOUNT NUMBER_ORBITALS AND DEGENERACIES...
    int number_orbitals=2;
    std::vector<int> degeneracies(number_orbitals);
    ///t2g and eg
    degeneracies[0]=3; 
    degeneracies[1]=2;
    
    int total_number_of_single_particle_states=0;
    for(int i=0;i<number_orbitals;i++)
        total_number_of_single_particle_states+=degeneracies[i];

    /// spin degree of freedom is considered in the building of the basis
    class multiple_electrons_basis_set(number_electrons,total_spin,number_orbitals,degeneracies,total_number_of_single_particle_states);
    multiple_electrons_basis_set.filling_basis_set_conserving_Q_and_S();
    
    ///writing the Hamiltonian in the basis set
    
    ///single-particle term: spin_channel, total_number_of_single_particle_states
    /// \Sum_{l=total_number_of_single_particle_states}\Sum_{\sigma=spin}\epsilon_{l\sigma}\hat{d}^{\adj}_{l\sigma}\hat{d}_{l\sigma}
    /// due to the degeneracy we expect that N (=degeneracy[x]) \epsilon_{l\sigma} would be the same for those l of total_number_of_single_particle_states being part of orbital x
    std::vector<double> E_0(total_number_of_single_particle_states*2);
    /// t2g anf eg
    /// t2g1_up t2g1_dn t2g2_up t2g2_dn t2g3_up t2g3_dn eg1_up eg1_dn eg2_up eg2_dn
    E_0[0]=1.0;
    E_0[1]=1.0;
    E_0[degeneracies[0]*2]=2.0;
    E_0[degeneracies[0]*2+1]=2.0;
    /// associating energy to the other orbitals taking into account the degeneracy of each orbital
    int counting=0;
    for(int j=0;j<number_orbitals;j++){
        for(int i=2;i<degeneracies[j]*2-2;i=i+2){
            E_0[i+counting]=E_0[counting];
            E_0[i+1+counting]=E_0[counting+1];
        }
        counting+=degeneracies[j]*2;
    }
    /// single-particle term could be treated as the interaction term, using the matrix spin_and_orbital

    ///interaction part
    ///More anisotropic matrix can be built
    ///\Sum_{mnpq=total_number_of_single_particle_states}U_{mn,pq}\hat{d}^{\adj}_{m\sigma}\hat{d}^{\adj}_{n\sigma}\hat{d}_{p\sigma'}\hat{d}_{q\sigma'}
    Eigen::MatrixXd U(number_orbitals*number_orbitals,number_orbitals*number_orbitals);
    U.setZero();
    ///creation(t2g1 t2g1) annihilation(t2g1 t2g1)
    ///interaction inter-t2g manifold (Uaaaa)
    U[0,0]=0.4;
    ///interaction inter-eg manifold (Uaaaa)
    U[number_orbitals,number_orbitals+1]=0.4;
    ///interaction (t2g eg) | (eg t2g) and vice-versa (Uabba)
    U[1,number_orbitals]=0.3;
    U[number_orbitals,1]=0.3;
    ///interaction (t2g eg) | (t2g eg) and vice-versa (Uabab)
    U[1,1]=0.3;
    U[number_orbitals,number_orbitals]=0.3;
    ///interaction (t2g t2g) | (eg eg) and vice-versa (Uaabb)
    U[1,number_orbitals+1]=-0.1;
    U[number_orbitals+1,1]=-0.1;

    ///Building the Hamiltonian in the defined basis set
    Hamiltonian H(multiple_electrons_basis_set,E_0,U,....);

    return 1;
};
