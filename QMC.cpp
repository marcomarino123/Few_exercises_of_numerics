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

/// Defining Complex Matrices
using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

/// Heaviside step function: Θ(x)
double modified_Theta(int x,int s1,int s2,int m1,int m2){
    if(s1!=s2)
        return (x >= 0) ? 1.0 : 0.0;
    else
        return (x+m2-m1 >= 0) ? 1.0 : 0.0;
}
/// The initial local Green function is given in real-frequency space
/// Fourier-transformed in terms of Totter-discretisation
/// Sigma (self-energy) has the same structure
class Local_green{
    private:
        int time_steps;
        int num_orb;
        double dtime;
        std::vector<MatrixXcd> green;
    public:
        Local_green(int init_time_steps,int init_num_orb,double init_dtime,std::vector<std::vector<MatrixXcd>> init_green,int omega_steps,double domega){
            time_steps=init_time_steps;
            num_orb=init_num_orb;
            dtime=init_dtime;
            green.resize(2,MatrixXcd(time_steps*num_orb,time_steps*num_orb));
            for(int s=0;s<2;s++)
                green[s].setZero();
            
            for(int s=0;s<2;s++)
                for(int i=0;i<time_steps;i++)
                    for(int j=0;j<time_steps;j++)
                        for(int m1=0;m1<num_orb;m1++)
                            for(int m2=0;m2<num_orb;m2++)
                                for(int o=0;o<omega_steps;o++)    
                                    green[s](i*num_orb+m1,j*num_orb+m2)+=init_green[s][o](m1,m2)*std::complex<double>(std::cos(domega*o*(i-j)*dtime),std::sin(domega*o*(i-j)*dtime));
        };
        std::vector<MatrixXcd> pull(){
            return green;
        };
};
class J_coupling{
    private:
        int num_orb;
        double dtime;
        int time_steps;
        std::vector<Eigen::MatrixXd> U;
    public:
        J_coupling(int init_num_orb,double init_dtime,int init_time_steps,std::vector<Eigen::MatrixXd> init_U){
            num_orb=init_num_orb;
            dtime=init_dtime;
            time_steps=init_time_steps;
            U=init_U;
        };
        std::vector<std::vector<std::vector<double>>> pull_J(std::vector<std::vector<Eigen::MatrixXd>> field){
            std::vector<std::vector<std::vector<double>>> J(2,std::vector<std::vector<double>>(time_steps,std::vector<double>(num_orb)));
            std::vector<double> zeros(num_orb);
            for(int m1=0;m1<num_orb;m1++)
                zeros[m1]=0.0;
            for(int s1=0;s1<2;s1++){
                for(int i=0;i<time_steps;i++){   
                    J[s1][i]=zeros;
                    for(int m1=0;m1<num_orb;m1++){
                        for(int s2=0;s2<2;s2++)
                            for(int m2=0;m2<num_orb;m2++)
                                J[s1][i][m1]+=(std::log(std::exp(dtime*U[s1*2+s2](m1,m2)/2)-std::sqrt(std::exp(dtime*U[s1*2+s2](m1,m2))-1)))*field[s1*2+s2][i](m1,m2)*2*modified_Theta(s2-s1-1,s1,s2,m2,m1);
                        J[s1][i][m1]=std::exp(J[s1][i][m1]);
                    }
                }
            }
            return J;
        };
};
class Aux_Field{
    private:
        int time_steps;
        int num_orb;
        double dtime;
        std::vector<std::vector<Eigen::MatrixXd>> field;
        /// Spin configurations of field
        /// (0 0) (0 1) (1 0) (1 1) 
        std::vector<MatrixXcd> green;
        std::vector<Eigen::MatrixXd> U;
        std::random_device seed;
    public:
        Aux_Field(int init_time_steps,int init_num_orb,double init_dtime,Local_green init_green,std::vector<Eigen::MatrixXd> init_U){
            time_steps=init_time_steps;
            num_orb=init_num_orb;
            dtime=init_dtime;
            U=init_U;
            green=init_green.pull();
            field.resize(4,std::vector<Eigen::MatrixXd>(time_steps,Eigen::MatrixXd(num_orb,num_orb)));
            std::mt19937 gen(seed());
            std::uniform_real_distribution<> dis(-1,1);
            for(int s1=0;s1<4;s1++)
                for(int i=0;i<time_steps;i++)
                    for(int m1=0;m1<num_orb;m1++)
                        for(int m2=0;m2<num_orb;m2++){
                            // Convert values to -1 or 1 based on a condition
                            field[s1][i](m1,m2) = (dis(gen) > 0.0) ? 1.0 : -1.0;
                        }
        };
        void print(){
            for(int s=0;s<4;s++)
                for(int i=0;i<time_steps;i++){
                    std::cout<<"Spin: "<<s<<" Time: "<<i<<std::endl;
                    for(int m1=0;m1<num_orb;m1++){
                        for(int m2=0;m2<num_orb;m2++)
                            std::cout<<field[s][i](m1,m2)<<"   ";
                        std::cout<<std::endl;
                        }
                    std::cout<<std::endl;
                }
        };
        std::vector<MatrixXcd> pull_M_matrix(){
            std::vector<MatrixXcd> M_matrix(2, MatrixXcd(time_steps*num_orb,time_steps*num_orb));
            
            J_coupling j(num_orb,dtime,time_steps,U);
            std::vector<std::vector<std::vector<double>>> expj=j.pull_J(field);

            Eigen::MatrixXd Itime = Eigen::MatrixXd::Identity(time_steps,time_steps);
            Eigen::MatrixXd Iorb = Eigen::MatrixXd::Identity(num_orb,num_orb);
            
            for(int s1=0;s1<2;s1++){
                for(int i=0;i<time_steps;i++)
                    for(int j=0;j<time_steps;j++)
                        for(int m1=0;m1<num_orb;m1++)
                            for(int m2=0;m2<num_orb;m2++)
                                M_matrix[s1](i*num_orb+m1,j*num_orb+m2)=(std::pow(dtime,2)*(green[s1](i*num_orb+m1,j*num_orb+m2))*expj[s1][j][m2]-Itime(i,j)*Iorb(m1,m2)*expj[s1][j][m1]+Itime(i,j));
            }
            return M_matrix;
        };
        void all_spins_flipped_Metropolis(int number_iterations){
            std::vector<MatrixXcd> new_matrix=pull_M_matrix();

            double old_probability=(std::real(new_matrix[0].determinant()))*(std::real(new_matrix[1].determinant()))-(std::imag(new_matrix[0].determinant()))*(std::imag(new_matrix[1].determinant()));
            double new_probability;
            double flip_probability;
            
            bool new_configuration;

            std::mt19937 gen(seed());
            std::uniform_real_distribution<> dis1(-1,1);
            std::uniform_real_distribution<> dis2(0,1);

            for(int i=0;i<number_iterations;i++){
                new_configuration=false;
                while(new_configuration==false){
                    #pragma omp parallel
                    for(int s1=0;s1<4;s1++)
                        for(int i=0;i<time_steps;i++)
                            for(int m1=0;m1<num_orb;m1++)
                                for(int m2=0;m2<num_orb;m2++){
                                    // Convert values to -1 or 1 based on a condition
                                    field[s1][i](m1,m2) = (dis1(gen) > 0.0) ? 1.0 : -1.0;
                                }
                    new_matrix=pull_M_matrix();
                    new_probability=(std::real(new_matrix[0].determinant()))*(std::real(new_matrix[1].determinant()))-(std::imag(new_matrix[0].determinant()))*(std::imag(new_matrix[1].determinant()));
                    if(new_probability/old_probability<1)
                        flip_probability=new_probability/old_probability;
                    else
                        flip_probability=1;
                    if(dis2(gen)<flip_probability)
                        new_configuration=true;
                }
                old_probability=new_probability;
            }
        };
        std::vector<MatrixXcd> pull_G_matrix(){
            std::vector<MatrixXcd> G=pull_M_matrix();
            for(int s=0;s<2;s++)
                G[s]=G[s].inverse();
            return G;
        };
};



int main(int argc,char** argv){	
// Anderson Impurity Model

    // Inverse temperature
	double beta=1.00;
    // Time steps
    int time_steps=100;
    // dt
    double dtime=beta/time_steps;
    // Number orbitals
    int num_orb=2;
    // U matrix
    double U_value=4.00;
    double V_value=2.00;
    double J_value=1.00;
    std::vector<Eigen::MatrixXd> U(4,Eigen::MatrixXd(num_orb,num_orb));
    for(int s1=0;s1<2;s1++)
        for(int s2=0;s2<2;s2++){
            if(s1==s1){
                for(int m1=0;m1<num_orb;m1++)
                    for(int m2=0;m2<num_orb;m2++){
                        if(m1==m2)
                            U[s1*2+s2](m1,m2)=0;
                        else   
                            U[s1*2+s2](m1,m2)=V_value-J_value;
                    }
            }else{
                for(int m1=0;m1<num_orb;m1++)
                    for(int m2=0;m2<num_orb;m2++){
                        if(m1==m2)
                            U[s1*2+s2](m1,m2)=U_value;
                        else   
                            U[s1*2+s2](m1,m2)=V_value;
                    }
            }
        }    
    /// Weiss Field Green-function
    int omega_steps=100;
    double domega=0.1;
    std::random_device seed;
	std::mt19937 gen(seed());
	std::uniform_real_distribution<> disr(-1,1);
    std::uniform_real_distribution<> disi(-1,1);
			
    std::vector<std::vector<MatrixXcd>> init_green(2,std::vector<MatrixXcd>(omega_steps,MatrixXcd(num_orb,num_orb)));
    for(int s=0;s<2;s++)
        for(int o=0;o<omega_steps;o++)
            for(int m1=0;m1<num_orb;m1++)
                for(int m2=0;m2<num_orb;m2++)
                    init_green[s][o](m1,m2)=std::complex<double>(disr(gen),disi(gen)); 
    Local_green local_g(time_steps,num_orb,dtime,init_green,omega_steps,domega);
   
    /// Generating the auxiliary field lattice and initialize it randomly
    Aux_Field s_field(time_steps,num_orb,dtime,local_g,U);
    ///s_field.print();

    /// Applying Monte-Carlo (warm-up+extraction)
    int number_iterations=10;
    s_field.all_spins_flipped_Metropolis(number_iterations); 
    ---> qua dovrebbe essere fatta una media sulle diverse estrazioni
    s_field.all_spins_flipped_Metropolis(number_iterations);
    s_field.pull_G_matrix();

	return 1;
};
