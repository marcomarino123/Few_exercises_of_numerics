
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
#include <thread> // for sleep
#include <chrono> // for time

// Function to clear console (works on most terminals)
void clearScreen() {
#ifdef _WIN32
    system("cls"); // Windows
#else
    system("clear"); // Unix/Linux/Mac
#endif
}


class Advection_1d{
/// \partial a/\partial t = -c\partial a/\partial x
	private:
		Eigen::VectorXd a_0;
		int space_points;
		int time_points;
		double dx;
		double dt;
		Eigen::VectorXd t_01{(2)};
		Eigen::VectorXd x_01{(2)};
		double diff_constant;
	public:
		Advection_1d(Eigen::VectorXd init_a_0,int init_space_points,int init_time_points,Eigen::VectorXd init_t_01,Eigen::VectorXd init_x_01,double init_diff_constant){
			a_0=init_a_0;
			space_points=init_space_points;
			time_points=init_time_points;
			t_01=init_t_01;
			x_01=init_x_01;
			dt=std::abs(t_01(1)-t_01(0))/time_points;
			dx=std::abs(x_01(1)-x_01(0))/space_points;
			diff_constant=init_diff_constant;
			/// Checking Courant-Friedrichs-Lewy criterium
			double ratio=dx/dt;
			if(ratio<diff_constant){
				std::cerr<<"Error: Courant-Friedrichs-Lewy criterium not satisfied"<<std::endl;
				std::cout<<"dx: "<<dx<<" and dt: "<<dt<<" and ratio: "<<ratio<<" which should be larger than the diffusion constant: "<<diff_constant<<std::endl;
			}
		};
		Eigen::MatrixXd lax_method(){
			Eigen::MatrixXd a(time_points,space_points);
			a.row(0)=a_0;
			///std::cout<<a.row(0)<<std::endl;
			for(int t=1;t<time_points;t++){
				for(int x=1;x<space_points-1;x++)
					a(t,x)=(0.5)*(a(t-1,x+1)+a(t-1,x-1))-diff_constant*0.5*(dt/dx)*(a(t-1,x+1)-a(t-1,x-1));
					a(t,0)=a(t-1,0)-diff_constant*(dt/dx)*(a(t-1,1)-a(t-1,0));
					a(t,space_points-1)=a(t-1,space_points-1)+diff_constant*(dt/dx)*(a(t-1,space_points-2)-a(t-1,space_points-1));
				///std::cout<<a.row(t)<<std::endl;
			}
			return a;
		};
		void draw_histogram(Eigen::MatrixXd a, double step_histogram){
    		int number_bins = static_cast<int>(std::ceil(std::abs(x_01(0) - x_01(1)) / step_histogram));
			Eigen::MatrixXd frequencies(time_points,number_bins);
			int bin;
			Eigen::VectorXd total(time_points);

			for(int t=0;t<time_points;t++){	
				frequencies.row(t).setZero();
				total(t)=0;
    			// Fill frequency bins
    			for(int x=0;x<space_points;x++){
    			    bin=static_cast<int>(std::abs(x_01(0)+x*dx)/step_histogram);
					////std::cout<<a(t,x)<<std::endl;
    			    frequencies(t,bin)+=a(t,x);
					total(t)+=a(t,x);
    			}
			}
			
			for (int t = 0; t < time_points; t++) {
    			clearScreen(); // <<--- clear before drawing new frame

    			// Draw histogram
    			for (int j = 0; j < number_bins; j++) {
    			    double bin_start = x_01(0) + j * step_histogram;
    			    std::cout << bin_start << " | ";
    			    for (int i = 0; i < (static_cast<double>(frequencies(t,j)) / total(t)) * 100; ++i) {
    			        std::cout << '.';
    			    }
    			    std::cout << " (" << int((static_cast<double>(frequencies(t,j)) / total(t)) * 100) << ")\n";
    			}
			
    			// Wait a bit so you can see changes over time
    			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		};
};



int main(int argc,char** argv){
	
	int space_points=100;
	int time_points=10000;
	Eigen::VectorXd a_0(space_points);
	///a_0.setRandom();
	///a_0.normalize();
	a_0.setZero();
	a_0(0)=1.0;
	a_0(space_points-1)=1.0;

	Eigen::VectorXd t_01(2);
	t_01(0)=0.0; t_01(1)=1.0;
	Eigen::VectorXd x_01(2);
	x_01(0)=0.0; x_01(1)=1.0;
	
	double diff_constant=5.0;
	Advection_1d adv1d(a_0,space_points,time_points,t_01,x_01,diff_constant);
	Eigen::MatrixXd a=adv1d.lax_method();
	adv1d.draw_histogram(a,0.1);

	return 1;
};