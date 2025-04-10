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
};
double delta(int i,int j){
    if(i==j)
        return 1.0;
    else   
        return 0.0;
}
/// spin_pair
std::tuple<int,int> spin_pair(int spin){
    int spin1;
    int spin2;
    if(spin==0){
        spin1=0;
        spin2=0;
    }else if(spin==1){
        spin1=0;
        spin2=1;
    }else if(spin==2){
        spin1=1;
        spin2=0;
    }else{
        spin1=1;
        spin2=1;
    }
    return {spin1,spin2};
};
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
        std::vector<MatrixXcd> invgreen;
        std::vector<Eigen::MatrixXd> U;
        std::random_device seed;
        std::vector<MatrixXcd> current_invM_matrix;
        double R1;
        double R2;
    public:
        Aux_Field(int init_time_steps,int init_num_orb,double init_dtime,Local_green init_green,std::vector<Eigen::MatrixXd> init_U){
            time_steps=init_time_steps;
            num_orb=init_num_orb;
            dtime=init_dtime;
            U=init_U;
            invgreen=init_green.pull();
            for(int s=0;s<2;s++)
                invgreen[s]=invgreen[s].inverse();
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
            current_invM_matrix=pull_invM_matrix();

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
        std::vector<MatrixXcd> pull_invM_matrix(){
            std::vector<MatrixXcd> invM_matrix(2, MatrixXcd(time_steps*num_orb,time_steps*num_orb));
            
            J_coupling j(num_orb,dtime,time_steps,U);
            std::vector<std::vector<std::vector<double>>> expj=j.pull_J(field);

            Eigen::MatrixXd Itime = Eigen::MatrixXd::Identity(time_steps,time_steps);
            Eigen::MatrixXd Iorb = Eigen::MatrixXd::Identity(num_orb,num_orb);
            
            for(int s1=0;s1<2;s1++){
                #pragma omp parallel for collapse(4) private(s1)
                for(int i=0;i<time_steps;i++)
                    for(int j=0;j<time_steps;j++)
                        for(int m1=0;m1<num_orb;m1++)
                            for(int m2=0;m2<num_orb;m2++)
                                invM_matrix[s1](i*num_orb+m1,j*num_orb+m2)=(std::pow(dtime,2)*(invgreen[s1](i*num_orb+m1,j*num_orb+m2))*expj[s1][j][m2]-Itime(i,j)*Iorb(m1,m2)*expj[s1][j][m1]+Itime(i,j));
            }
            return invM_matrix;
        };
        std::vector<MatrixXcd> pull_G_matrix(){
            std::vector<MatrixXcd> G=current_invM_matrix;
            for(int s=0;s<2;s++)
                G[s]=G[s].inverse();
            return G;
        };
        std::vector<MatrixXcd> single_spin_flipped_Metropolis_recalculating_allG(bool warmup,int number_iterations){
            std::vector<MatrixXcd> new_invmatrix=current_invM_matrix;
            std::vector<MatrixXcd> final_invmatrix=new_invmatrix;

            // Use LU decomposition for determinant (log det is more stable)
            double logdet_real = 0.0;
            for (int s = 0; s < 2; s++){
                Eigen::FullPivLU<MatrixXcd> lu(new_invmatrix[s]);
                logdet_real += lu.matrixLU().diagonal().array().log().real().sum();
            }
            double new_logdet_real;
            
            bool new_configuration;

            std::mt19937 gen(seed());
            std::uniform_int_distribution<> rand_s1(0, 3);
            std::uniform_int_distribution<> rand_i(0, time_steps - 1);
            std::uniform_int_distribution<> rand_m(0, num_orb - 1);
            std::uniform_real_distribution<> dis(0.0, 1.0);

            for(int i=0;i<number_iterations;i++){
                new_configuration=false;
                while(new_configuration==false){
                    int s1 = rand_s1(gen);
                    int i = rand_i(gen);
                    int m1 = rand_m(gen);
                    int m2 = rand_m(gen);

                    // Flip a single spin
                    field[s1][i](m1, m2) *= -1;
                   
                    new_invmatrix = pull_invM_matrix();

                    double new_logdet_real = 0.0;
                    for (int s = 0; s < 2; s++) {
                        Eigen::FullPivLU<MatrixXcd> lu(new_invmatrix[s]);
                        new_logdet_real += lu.matrixLU().diagonal().array().log().real().sum();
                    }
            
                    double log_ratio = - new_logdet_real + logdet_real;

                    if(log_ratio >= 0 || std::log(dis(gen)) < log_ratio) {
                        new_configuration=true;
                        logdet_real = new_logdet_real;
                        if(warmup==false){
                            for(int s=0;s<2;s++)
                                final_invmatrix[s]+=new_invmatrix[s]/number_iterations;
                        }
                    }else{
                        // Reject: flip back
                        field[s1][i](m1, m2) *= -1;
                    }
                }
            }
            return final_invmatrix;
        };
        double single_spin_flip_probability_ratio(int s,int i,int m1,int m2){
            double probability_ratio;
            int s1=std::get<0>(spin_pair(s));
            int s2=std::get<1>(spin_pair(s));
            double J=2*std::log(std::exp(dtime*U[s1*2+s2](m1,m2)/2)-std::sqrt(std::exp(dtime*U[s1*2+s2](m1,m2))-1))*modified_Theta(s2-s1-1,s1,s2,m2,m1)*field[s1][i](m1, m2);
            R1=1.0+(1.0-current_invM_matrix[s1](i*num_orb+m1,i*num_orb+m1).real())*(std::exp(J)-1.0);
            R2=1.0+(1.0-current_invM_matrix[s2](i*num_orb+m2,i*num_orb+m2).real())*(std::exp(-J)-1.0);
            probability_ratio=R1*R2;
            if(s1==s2)
                probability_ratio-=current_invM_matrix[s1](i*num_orb+m1,i*num_orb+m2).real()*(std::exp(-J)-1.0)*current_invM_matrix[s1](i*num_orb+m2,i*num_orb+m1).real()*(std::exp(J)-1.0);
            return probability_ratio;
        };
        std::vector<MatrixXcd> single_spin_flip_new_invmatrix(int s,int i,int m1,int m2){
            std::vector<MatrixXcd> new_invM_matrix=current_invM_matrix;
            std::vector<MatrixXcd> intermediate_matrix=current_invM_matrix;
            int s1=std::get<0>(spin_pair(s));
            int s2=std::get<1>(spin_pair(s));
            double J=2*std::log(std::exp(dtime*U[s1*2+s2](m1,m2)/2)-std::sqrt(std::exp(dtime*U[s1*2+s2](m1,m2))-1.0))*modified_Theta(s2-s1-1,s1,s2,m2,m1)*field[s1][i](m1, m2);
            #pragma omp parallel for collapse(4) 
            for(int l=0;l<time_steps;l++)
                for(int k=0;k<time_steps;k++)
                    for(int m3=0;m3<num_orb;m3++)
                        for(int m4=0;m4<num_orb;m4++){
                            intermediate_matrix[s1](l*num_orb+m3,k*num_orb+m4)=current_invM_matrix[s1](l*num_orb+m3,k*num_orb+m4)+((std::exp(J)-1.0)/R1)*(current_invM_matrix[s1](l*num_orb+m3,i*num_orb+m1)-delta(l,i)*delta(m3,m1))*current_invM_matrix[s1](i*num_orb+m1,k*num_orb+m4);
                        }
            double R3=1.0+(1.0-intermediate_matrix[s2](i*num_orb+m2,i*num_orb+m2).real())*(std::exp(-J)-1);
            #pragma omp parallel for collapse(4) 
            for(int l=0;l<time_steps;l++)
                for(int k=0;k<time_steps;k++)
                    for(int m3=0;m3<num_orb;m3++)
                        for(int m4=0;m4<num_orb;m4++){
                            new_invM_matrix[s2](l*num_orb+m3,k*num_orb+m4)=intermediate_matrix[s2](l*num_orb+m3,k*num_orb+m4)+((std::exp(J)-1.0)/R3)*(intermediate_matrix[s2](l*num_orb+m3,i*num_orb+m2)-delta(l,i)*delta(m3,m2))*intermediate_matrix[s2](i*num_orb+m2,k*num_orb+m4);
                    }
            return new_invM_matrix;
        };
        std::vector<MatrixXcd> single_spin_flipped_Metropolis(bool warmup,int number_iterations){
            std::vector<MatrixXcd> final_invmatrix=current_invM_matrix;

            double probability_ratio;
            
            bool new_configuration;

            std::mt19937 gen(seed());
            std::uniform_int_distribution<> rand_s(0, 3);
            std::uniform_int_distribution<> rand_i(0, time_steps - 1);
            std::uniform_int_distribution<> rand_m(0, num_orb - 1);
            std::uniform_real_distribution<> dis(0.0, 1.0);

            for(int i=0;i<number_iterations;i++){
                new_configuration=false;
                while(new_configuration==false){
                    int s = rand_s(gen);
                    int i = rand_i(gen);
                    int m1 = rand_m(gen);
                    int m2 = rand_m(gen);

                    // Flip a single spin
                    field[s][i](m1, m2) *= -1;
                   
                    probability_ratio=single_spin_flip_probability_ratio(s,i,m1,m2);

                    if(probability_ratio >= 0 || dis(gen) < probability_ratio) {
                        new_configuration=true;
                        if(warmup==false){
                            current_invM_matrix=single_spin_flip_new_invmatrix(s,i,m1,m2);
                            for(int s=0;s<2;s++)
                                final_invmatrix[s]+=current_invM_matrix[s]/number_iterations;
                        }
                    }else{
                        // Reject: flip back
                        field[s][i](m1, m2) *= -1;
                    }
                }
            }
            return final_invmatrix;
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
    int omega_steps=1000;
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

    ///ONE SPIN FLIP RECALCULATING ALL G FUNCTION
    ///int number_iterations_warmup=10;
    ///bool warmup=true;
    ///s_field.single_spin_flipped_Metropolis_recalculating_allG(warmup,number_iterations_warmup);
    ///int number_measuraments=100;
    ///warmup=false;
    ///std::vector<MatrixXcd> G=s_field.single_spin_flipped_Metropolis_recalculating_allG(warmup,number_measuraments);
    ///std::cout<<G[0].inverse()<<std::endl;

    ///ONE SPIN FLIP NOT-RECALCULATING ALL G FUNCTION
    int number_iterations_warmup=10;
    bool warmup=true;
    s_field.single_spin_flipped_Metropolis(warmup,number_iterations_warmup);
    int number_measuraments=100;
    warmup=false;
    std::vector<MatrixXcd> G=s_field.single_spin_flipped_Metropolis(warmup,number_measuraments);
    std::cout<<G[0].inverse()<<std::endl;;
    
	return 1;
};