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
#include </home/marco-marino/Desktop/TA_Computational_Physics/exercises_1.cpp>
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
		Eigen::MatrixXd discretization_in_space(){
			///reducing the partial differential equation to a matricial problem Ax=b
			///due to the discretization of the differential operator
			Eigen::MatrixXd A(space_points-2,space_points-2);
			Eigen::MatrixXd b(space_points-2,1);
			double dx=std::abs(x2-x1)/space_points;
			///b.col(0)=-rho.block(1,0,space_points-2,1).transpose();
			b(0,0)=-phi_1/pow(dx,2);
			b(space_points-3,0)=-phi_2/pow(dx,2);
			for(int i=0;i<space_points-3;i++){
				A(i,i)=-2.0/pow(dx,2);
				A(i,i+1)=1.0/pow(dx,2);
				A(i+1,i)=1.0/pow(dx,2);
			}
			LGS lgs(A,b);
			return lgs.LU_decomposition_Eigen_routine(space_points-2,space_points-2,A,b);
		};
		Eigen::VectorXd iteration_procedure(double threshold){
			Eigen::VectorXd before_phi(space_points);
			Eigen::VectorXd next_phi(space_points);
			double dx=std::abs(x2-x1)/space_points;
			next_phi.setZero();
			next_phi(0)=phi_1;next_phi(space_points-1)=phi_2;
			do{
				before_phi=next_phi;
				for(int i=1;i<space_points-1;i++)
					next_phi(i)=(before_phi(i-1)+before_phi(i+1))/2.0+std::pow(dx,2)*rho(i)/2;
			}while((before_phi-next_phi).norm()>threshold);
			return next_phi;
		};
};

class Poisson_2d{
	private:
		/// charge density
		Eigen::VectorXd rho;
		/// boundary conditions
		Eigen::VectorXd phi_boundary;
		/// border points - square grid
		double x1; double y1;
		double x2; double y2;
		/// number of spatial points 
		int space_points_x;
		int space_points_y;
	public:
		Poisson_2d(Eigen::VectorXd init_rho,Eigen::VectorXd init_phi_boundary,std::vector<double> xy,int init_space_points_x,int init_space_points_y){
			rho=init_rho;
			phi_boundary=init_phi_boundary;
			x1=xy[0];y1=xy[2];
			x2=xy[1];y2=xy[3];
			space_points_x=init_space_points_x;
			space_points_y=init_space_points_y;
		};
		Eigen::MatrixXd discretization(){
			Eigen::MatrixXd A((space_points_x-2)*(space_points_y-2),(space_points_x-2)*(space_points_y-2));
			Eigen::VectorXd phi((space_points_x-2)*(space_points_y-2));
			double dx=std::abs(x2-x1)/space_points_x;
			double dy=std::abs(y2-y1)/space_points_y;
			/// x0y0,x0y1,x0y2,x0y3....x1y0,x1y1,x1y2....
			/// left space points, right, top and bottom
			Eigen::VectorXd left(space_points_y);
			Eigen::VectorXd right(space_points_y);
			Eigen::VectorXd top(space_points_x);
			Eigen::VectorXd bottom(space_points_x);
			///positioning of borders in the vector
			//for(int i=0;i<space_points_y;i++){
			//	left(i)=i;
			//	right(i)=(space_points_x-1)*(space_points_y-2)+i;
			//}
			//for(int i=0;i<space_points_x;i++){
			//	bottom(i)=i*space_points_y;
			//	top(i)=i*space_points_y+(space_points_y-1);
			//}
			/////adding to the source the borders
			for(int i=0;i<space_points_y-2;i++){
				///left and right
				rho(i)+=phi_boundary(2*space_points_x+i)/std::pow(dx,2);
				rho((space_points_x-2)*(space_points_y-3)+i)+=phi_boundary(2*space_points_x+space_points_y-2+i)/std::pow(dx,2);
			}
			for(int i=0;i<space_points_x-2;i++){
				///top and bottom
				rho(i*(space_points_y-3)+space_points_y-2)+=phi_boundary(i)/std::pow(dy,2);
				rho(i*(space_points_y-2))+=phi_boundary(space_points_x+i)/std::pow(dy,2);
			}			
			for(int i=0;i<(space_points_x-2)*(space_points_y-2);i++)
				rho(i)=-rho(i);

			///building matrix A, discretizing harmonic operator in 2 dimensions
			for(int i=0;i<space_points_x-2;i++)
				for(int j=0;j<space_points_y-2;j++)
					A(i*(space_points_y-2)+j,i*(space_points_y-2)+j)=-2/std::pow(dx,2)-2/std::pow(dy,2);
			for(int i=1;i<space_points_x-2;i++)
				for(int j=0;j<space_points_y-2;j++)
					A(i*(space_points_y-2)+j,(i-1)*(space_points_y-2)+j)=1/std::pow(dx,2);
			for(int i=0;i<space_points_x-3;i++)
				for(int j=0;j<space_points_y-2;j++)
						A(i*(space_points_y-2)+j,(i+1)*(space_points_y-2)+j)=1/std::pow(dx,2);
			
			for(int i=0;i<space_points_x-2;i++)
				for(int j=0;j<space_points_y-3;j++)	
					A(i*(space_points_y-2)+j,i*(space_points_y-2)+j+1)=1/std::pow(dy,2);
			for(int i=0;i<space_points_x-2;i++)
				for(int j=1;j<space_points_y-2;j++)			
					A(i*(space_points_y-2)+j,i*(space_points_y-2)+j-1)=1/std::pow(dy,2);

			///std::cout<<A<<std::endl;
			////A_{ij}\phi_{j}=-rho_{i}
			LGS lgs(A,rho);
			return lgs.LU_decomposition_Eigen_routine((space_points_x-2)*(space_points_y-2),(space_points_x-2)*(space_points_y-2),A,rho);
		};
};

int main(int argc,char** argv){
	///double x1=0.0;
	///double x2=10.0;
	///double phi1=0.0;
	///double phi2=1.0;
	///int space_points=1000;
	///Eigen::VectorXd rho(space_points);
	///rho.setRandom();
	///Poisson_1d poisson_1d(rho,space_points,phi1,phi2,x1,x2);
	//////std::cout<<poisson_1d.evolution()<<std::endl;
	//////std::cout<<poisson_1d.discretization_in_space()<<std::endl;
	///double threshold=0.001;
	///std::cout<<poisson_1d.iteration_procedure(threshold)<<std::endl;
	
	std::vector<double> xy(4);
	xy[0]=-1; xy[1]=1;
	xy[2]=-1; xy[3]=1;
	int space_points_x=100;
	int space_points_y=100;
	Eigen::VectorXd rho((space_points_x-2)*(space_points_y-2));
	/// METHODS OF WRITING THE MATRICES AS VECTORS
	/// 1) harder method:
	/// +1y,-y till y0 per yn>y2 junping the first y,+1x,-2x till x0 per xn>x2 jumping the first x,+1x+1y....
	/// x0y0,x0y1,x1y0,x1y1,x1y2,x2y1,x2y2,x2y3,x2y0,x3y2,x0y2,x3y3
	/// x3y4,x3y1,x3y0
	/// 2) heasier method:
	/// -y till y0, -x till x0, +1x+1y....
	///  x3y3,x3y2,x3y1,x3y0,x2y3,x1y3,x0y3,x4y4,x4y3,x4y2....
	///  X0Y0,x1y1,x1y0,x0y1,x2y2,x2y1,x2y0,....
	/// 3) used here
	//// x0y0,x0y1.....
	rho.setRandom();
	Eigen::VectorXd phi_boundary(2*space_points_x+2*(space_points_y-2)); ///avoiding double counting of corners
	/// boundaries: bottom and top of the square
	for(int i=0;i<space_points_x;i++){
		phi_boundary(i)=1.0;
		phi_boundary(space_points_x+i)=1.0;
	}
	/// boundaries: left and right of the square 
	for(int i=0;i<space_points_y-2;i++){
		phi_boundary(2*space_points_x+i)=0.0;
		phi_boundary(2*space_points_x+space_points_y-2+i)=0.0;
	}
	Poisson_2d poisson_2d(rho,phi_boundary,xy,space_points_x,space_points_y);
	std::cout<<poisson_2d.discretization()<<std::endl;



	
	return 1;
};