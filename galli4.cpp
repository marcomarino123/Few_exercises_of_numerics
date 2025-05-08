#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <numbers>
#include <random>
#include <algorithm>
#include <chrono>
#include </usr/local/Eigen>

class Spin_Lattice{
    private:
        int number_sites;
        int rows;
        int columns;
        Eigen::MatrixXd spins;
        Eigen::MatrixXd periodicity;
        double coupling;
        double beta;
        std::random_device seed;
    public:
        explicit Spin_Lattice(int init_rows,int init_columns,double init_coupling,double init_beta,bool ordered)
        : rows(init_rows),columns(init_columns),number_sites(rows*columns),spins(Eigen::MatrixXd::Ones(rows,columns)),
          periodicity(Eigen::MatrixXd::Zero((rows+2)*(columns+2),2)),coupling(init_coupling),beta(init_beta){
            number_sites=rows*columns;
            if(ordered==false){
                ///Seed the random number generator
                std::mt19937 gen(seed());
                std::uniform_real_distribution<> dis(0,1); 
                ///Fill matrix with +1 or -1 randomly
                for(int i=0;i<rows;++i)
                    for(int j=0;j<columns;++j)
                        spins(i,j)=(dis(gen) > 0.5) ? 1 : -1;
            }
            ///Matrix of indices allowing periodicity
            for(int i=0;i<rows;++i)
                for(int j=0;j<columns;++j){
                    periodicity((columns+2)*(i+1)+j+1,0)=i;
                    periodicity((columns+2)*(i+1)+j+1,1)=j;
                }
            for(int j=0;j<columns;++j){
                periodicity(j+1,0)=rows-1;
                periodicity(j+1,1)=j;
            }
            for(int j=0;j<columns;++j){
                periodicity((columns+2)*(rows+1)+j+1,0)=0;
                periodicity((columns+2)*(rows+1)+j+1,1)=j;
            }
            for(int i=0;i<rows;++i){
                periodicity((columns+2)*(i+1),0)=i;
                periodicity((columns+2)*(i+1),1)=columns-1;
            }
            for(int i=0;i<rows;++i){
                periodicity((columns+2)*(i+1)+columns+1,0)=i;
                periodicity((columns+2)*(i+1)+columns+1,1)=0;
            }
            ///for(int i=-1;i<rows+1;++i){
            //    for(int j=-1;j<columns+1;++j)
            //        std::cout<<"("<<periodicity((columns+2)*(i+1)+j+1,0)<<","<< periodicity((columns+2)*(i+1)+j+1,1)<<") ";
            //    std::cout<<std::endl;
            //}
        };
        void printing(){
            std::cout<<"Spin Lattice"<<std::endl;
            for(int i=0;i<rows;++i){
                for(int j=0;j<columns;++j){
                    if(spins(i,j)>=0.0)
                        std::cout<<"*"<<" ";
                    else  
                        std::cout<<"o"<<" ";
                }
                ///std::cout<<((spins(i,j)==1) ? 0 : 1)<<" ";
                std::cout<<std::endl;
            }
        };
        void spin_flipping(int i,int j){
            spins(i,j)=-spins(i,j);
        };
        double difference_of_energy(int i,int j){
            double difference=0;
            for(int l:{-1,1})
                for(int k:{-1,1}){
                    difference+=spins(static_cast<int>(periodicity((columns+2)*(i+1+k)+j+l+1,0)),static_cast<int>(periodicity((columns+2)*(i+1+k)+j+l+1,1)));
                }
            difference*=2*coupling*spins(i,j);
            return difference;
        };
        void metropolis(int number_iterations){
            std::mt19937 gen(seed());
            std::uniform_int_distribution<> disr(0,rows-1);
            std::uniform_int_distribution<> disc(0,columns-1);
            std::mt19937 gen_mov(seed());
            std::uniform_real_distribution<> dis(0,1);
            double de=0.0;
            double a;
            int i,j;
            for(int k=0;k<number_iterations;k++){
                i=disc(gen);
                j=disr(gen);
                de=difference_of_energy(i,j);
                a=std::min(1.0,std::exp(-beta*de));
                //if(de<=0){
                //    spin_flipping(i,j);
                //}else{
                //    if(dis(gen_mov)<std::exp(-beta*de))
                //        spin_flipping(i,j);
                //}
                if(dis(gen_mov)<=a)
                    spin_flipping(i,j);
            }
        };
        void gibbs(int number_iterations){
            std::mt19937 gen(seed());
            std::uniform_int_distribution<> disr(0,rows-1);
            std::uniform_int_distribution<> disc(0,columns-1);
            std::mt19937 gen_mov(seed());
            std::uniform_real_distribution<> dis(0,1);
            double de=0.0;
            double a;
            int i,j;
            for(int k=0;k<number_iterations;k++){
                i=disc(gen);
                j=disr(gen);
                de=difference_of_energy(i,j);
                a=1.0/(1.0+std::exp(beta*de));
                //if(de<=0){
                //    spin_flipping(i,j);
                //}else{
                //    if(dis(gen_mov)<std::exp(-beta*de))
                //        spin_flipping(i,j);
                //}
                if(dis(gen_mov)<=a)
                    spin_flipping(i,j);
            }
        };
};

int main(int argc,char** argv){
    int rows=20;
    int columns=20;
    double beta=1.0e+5;
    double coupling=-100;
    bool ordered=false;
    Spin_Lattice S(rows,columns,coupling,beta,ordered);
    S.printing();
    int number_iterations=static_cast<int>(1.0e+7);
    // timing
    auto start1 = std::chrono::high_resolution_clock::now();
    S.metropolis(number_iterations);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    std::cout << "Metropolis took " << duration1.count() << " microseconds.\n";
    S.printing();
    auto start2 = std::chrono::high_resolution_clock::now();
    S.gibbs(number_iterations);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    std::cout << "Gibbs took " << duration2.count() << " microseconds.\n";
    S.printing();
    return 1;
}