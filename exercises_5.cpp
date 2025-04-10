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
#include <SFML/Graphics.hpp>
#include </home/marco-marino/Desktop/TA_Computational_Physics/exercises_3.cpp>

class Poisson_1d{
	private:
		/// charge density
		Eigen::VectorXd rho;
		/// boundary conditions
		double phi_1;
		double phi_2;
		double x1;
		double x2;
		/// number of spatial points 
		int space_points;
	public:
		Poisson_1d(Eigen::VectorXd init_rho, int init_space_points, double init_phi_1, double init_phi_2, double init_x1, double init_x2){
			rho=init_rho;
			phi_1=init_phi_1;
			phi_2=init_phi_2;
			x1=init_x1;
			x2=init_x2;
			space_points=init_space_points;
		};
		Eigen::VectorXd evolution(){
			///y'(x)=-\rho(x)
			///\phi'(x)=y(x)
			///y(x1)=\alpha
			///\phi(x1)=\bar{\phi_1}
			Eigen::MatrixXd y_phi(2,space_points);
			double dx=std::abs(x2-x1)/space_points;
			bool alpha_found=false;
			double lowestDouble = std::numeric_limits<double>::lowest();
			double maxDouble = std::numeric_limits<double>::max();
			Minimization_1d minimization_1d;
			int iter_1d;
			double acc=0.0001;
			std::function<double(double)> y_phi_alpha = [&](double alpha){
					y_phi(0,0)=alpha;
					y_phi(1,0)=phi_1;
					for(int i=1;i<space_points;i++){
						y_phi(0,i)=y_phi(0,i-1)-dx*rho(i-1);
						y_phi(1,i)=y_phi(1,i-1)+dx*y_phi(0,i-1);
					}
					return std::abs(y_phi(1,space_points-1)-alpha);
			};
			minimization_1d.chg_func(y_phi_alpha);
			iter_1d=0;
			///indicate a smartest intervall
			double min=y_phi_alpha(minimization_1d.interval_bisection(-100,0,100,acc,&iter_1d));
			return y_phi.row(1);
		};	
};



int main(int argc,char** argv){	

	double x1=0.0;
	double x2=10.0;
	double phi1=0.0;
	double phi2=1.0;
	int space_points=1000;
	Eigen::VectorXd rho(space_points);
	rho.setRandom();
	Poisson_1d poisson_1d(rho,space_points,phi1,phi2,x1,x2);
	std::cout<<poisson_1d.evolution()<<std::endl;


	return 1;
};