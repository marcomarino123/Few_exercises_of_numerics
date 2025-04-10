#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include </usr/local/Eigen>
///#include </home/marco-marino/Desktop/TA_Computational_Physics/exercises_1.cpp>
///#include </home/marco-marino/delaunator-cpp/include/delaunator.hpp>
#include <cstdio>
#include <functional>

double test_function_1(double x){
	return std::pow(x,3.0);
};

double test_function_n(Eigen::VectorXd x){
	double sum=0;
	for(int i=0;i<x.size();i++)
		sum+=x(i)*x(i);
	return std::cos(sum);
};

class Minimization_1d{
private:
	///double (*func)(double);
	std::function<double(double)> func;
public:
	///Minimization_1d(double (*init_func)(double)){
	///	func=init_func;
	///};
	Minimization_1d(std::function<double(double)> init_func){
		func=init_func;
	};
	Minimization_1d(){
	};
	void chg_func(std::function<double(double)> new_func){
		func=new_func;
	};
	double interval_bisection(double a,double b,double c,const double acc,int *iter){
		std::cout<<a<<" "<<b<<" "<<c<<" "<<acc<<" "<<*iter<<std::endl;
		if(std::abs(c-a)<acc){
			return a;
		}
		if(*iter==0){
			do{
			std::random_device seed;
			std::mt19937 gen(seed());
			std::uniform_real_distribution<> dis(a,c);
			b=dis(gen);
			if(func(b)<func(a)||func(b)<func(c))
				continue;
			}while(func(b)>func(a)&&func(b)>func(c));
		}
		*iter=*iter+1;
		if(std::abs(b-a)<std::abs(c-b)){
			if(func((c+b)/2.0)<func(b)){
				return interval_bisection(b,(c+b)/2.0,c,acc,iter);
			}else{
				return interval_bisection(a,b,(c+b)/2.0,acc,iter);
			}
		}else{
			if(func((b+a)/2.0)<func(b)){
				return interval_bisection(a,(b+a)/2.0,b,acc,iter);
			}else{
				return interval_bisection((b+a)/2.0,b,c,acc,iter);
			}
		}
	};
	double interval_bisection_wolfe_condition(double a,double b,double c,const double acc,int *iter,double dm0,double dml,int numb_tries){
		std::cout<<a<<" "<<b<<" "<<c<<" "<<acc<<" "<<*iter<<std::endl;
		///if(std::abs(c-a)<acc){
		///	return a;
		///}else{
		///	if(*iter!=0){
		///		if(func(b))
		///		IMPLEMENTARE WOLFE CONDITIONS
///
		///	}
		///}
		if(*iter==0){
			do{
			std::random_device seed;
			std::mt19937 gen(seed());
			std::uniform_real_distribution<> dis(a,c);
			b=dis(gen);
			if(func(b)<func(a)||func(b)<func(c))
				continue;
			}while(func(b)>func(a)&&func(b)>func(c));
		}
		*iter=*iter+1;
		if(std::abs(b-a)<std::abs(c-b)){
			if(func((c+b)/2.0)<func(b)){
				return interval_bisection(b,(c+b)/2.0,c,acc,iter);
			}else{
				return interval_bisection(a,b,(c+b)/2.0,acc,iter);
			}
		}else{
			if(func((b+a)/2.0)<func(b)){
				return interval_bisection(a,(b+a)/2.0,b,acc,iter);
			}else{
				return interval_bisection((b+a)/2.0,b,c,acc,iter);
			}
		}
	};
	double parabolic(double a,double b,double c,const double acc,int *iter){
		double d;
		if(std::abs(c-a)<acc){
			return a;
		}
		if(*iter==0){
			do{
			std::random_device seed;
			std::mt19937 gen(seed());
			std::uniform_real_distribution<> dis(a,c);
			b=dis(gen);
			if(func(b)<func(a)||func(b)<func(c))
				continue;
			}while(func(b)>func(a)&&func(b)>func(c));
		}
		std::cout<<a<<" "<<b<<" "<<c<<" "<<acc<<" "<<*iter<<std::endl;
		*iter=*iter+1;
		/// in order to avoid the instability of the division by the difference between a and b
		if(std::abs((b-a)*(func(b)-func(c))-(b-c)*(func(b)-func(a)))<acc){
			if(std::abs(b-a)<std::abs(c-b))
				d=(c+b)/2.0;
			else	
				d=(b+a)/2.0;
		}else{
			std::cout<<"parabolic approximation"<<std::endl;
			d=b-((b-a)*(func(b)-func(c))-(b-c)*(b-c)*(func(b)-func(a)))/(2.0*((b-a)*(func(b)-func(c))-(b-c)*(func(b)-func(a))));
		}
		if(std::abs(b-a)<std::abs(c-b)){
			if(func(d)<func(b)){
				return parabolic(b,d,c,acc,iter);
			}else{
				return parabolic(a,b,d,acc,iter);
			}
		}else{
			if(func(d)<func(b)){
				return parabolic(a,d,b,acc,iter);
			}else{
				return parabolic(d,b,c,acc,iter);
			}
		}
	};
	double Newton(double a,double b,double c,const double acc,int *iter,double step_diff){
		double d;
		if(c-a<acc){
			return a;
		}
		if(*iter==0){
			do{
			std::random_device seed;
			std::mt19937 gen(seed());
			std::uniform_real_distribution<> dis(a,c);
			b=dis(gen);
			if(func(b)<func(a)||func(b)<func(c))
				continue;
			}while(func(b)>func(a)&&func(b)>func(c));
		}
		//std::cout<<a<<" "<<b<<" "<<c<<" "<<acc<<" "<<*iter<<std::endl;
		*iter=*iter+1;
		/// Three-point central difference formula for first derivative
		if((func(b)+(func(b+step_diff)-func(b-step_diff))/2.0+(func(b+step_diff)+func(b-step_diff)-2*func(b))/2.0<func(b))&&(std::abs(func(b+step_diff)+func(b-step_diff)-2*func(b))>acc)){
			d=b-(func(b+step_diff)-func(b-step_diff))/(2.0*(func(b+step_diff)+func(b-step_diff)-2*func(b)))*step_diff;
			return Newton(a,d,c,acc,iter,step_diff);
		}else{
			if(b-a<c-b){
				d=(c+b)/2.0;
				if(func(d)<func(b)){
					return Newton(b,d,c,acc,iter,step_diff);
				}else{
					return Newton(a,b,d,acc,iter,step_diff);
				}
			}else{
				d=(b+a)/2.0;
				if(func(d)<func(b)){
					return Newton(a,d,b,acc,iter,step_diff);
				}else{
					return Newton(d,b,c,acc,iter,step_diff);
				}
			}
		}
	};
};


class Minimization_nd{
	private:
		int dim;
		/// n-dimensional function
		std::function<double(Eigen::VectorXd)> func;
		/// 1-dimensional function (for line-search)
		std::function<double(double)> red_func;
		/// object for 1-dimensional minimization
		Minimization_1d minimization_1d;
		/// Defining n directions
		Eigen::MatrixXd dirs;
		Eigen::VectorXd grad;
		Eigen::VectorXd gradnext;
		int iter_1d;
	public:
		Minimization_nd(std::function<double(Eigen::VectorXd)> init_func,int init_dim){
			func=init_func;
			dim=init_dim;
			dirs=Eigen::MatrixXd::Identity(dim,dim);
			grad.resize(dim);
			gradnext.resize(dim);
		};
		Eigen::VectorXd powell_process(Eigen::VectorXd a,Eigen::VectorXd b,Eigen::VectorXd c,Eigen::VectorXd previousb,double acc,int* iter,double step_diff){
			///a(i) and c(i) gives you the domain of the i-th coordinate
			if(*iter==0){
				std::random_device seed;
				std::mt19937 gen(seed());
				for(int i=0;i<dim;i++){
					std::uniform_real_distribution<> dis(a(i),c(i));
					b(i)=dis(gen);
				}
			}
			Eigen::VectorXd x0=b;
			//std::cout<<"1 "<<a.transpose()<<" "<<b.transpose()<<" "<<c.transpose()<<" "<<x0.transpose()<<" "<<acc<<" "<<*iter<<std::endl;
			/// Minimizing in each direction at a time through the alghoritms above for 1-dimensional functions
			for(int i=0;i<dim-1;i++){
				/// Defining the one-dimensional function assciated with the n-dimensional function for each direction
				red_func = [&](double lambda) {
					return func(b + lambda * dirs.row(i).transpose());
				};
				iter_1d=0;
				minimization_1d.chg_func(red_func);
				//b=b+(minimization.Newton(0,0,(b-c).norm(),acc,iter_1d,step_diff))*dirs.row(i).transpose();
				b=b+(minimization_1d.interval_bisection(-b(i)+a(i),0,c(i)-b(i),acc,&iter_1d))*dirs.row(i).transpose();
			}
			//std::cout<<"2 "<<a.transpose()<<" "<<b.transpose()<<" "<<c.transpose()<<" "<<x0.transpose()<<" "<<acc<<" "<<*iter<<std::endl;
			/// Last direction pn=b-x0
			Eigen::VectorXd pn(dim);
			if((b-x0).norm()>acc)
				pn=(b-x0)/((b-x0).norm());
			else
				pn=dirs.row(dim-1).transpose();
			iter_1d=0;
			std::function<double(double)> red_func = [&](double lambda) {
				return func(b + lambda * pn);
			};
			minimization_1d.chg_func(red_func);
			////b=b+(minimization.Newton(0,0,(b-c).norm(),acc,iter_1d,step_diff))*(b-x0);
			b=b+(minimization_1d.interval_bisection((a-b).transpose()*pn,0,(c-b).transpose()*pn,acc,&iter_1d))*pn;
			///std::cout<<"3 "<<a.transpose()<<" "<<b.transpose()<<" "<<c.transpose()<<" "<<x0.transpose()<<" "<<acc<<" "<<*iter<<std::endl;
			/// check with respect to previous b
			if(*iter!=0){
				//std::cout<<"comparison "<<b.transpose()<<" "<<previousb.transpose()<<std::endl;
				if((b-previousb).norm()<acc){
					return b;
				}
			}
			/// Recall the function
			*iter=*iter+1;
			previousb=b;
			return powell_process(a,b,c,previousb,acc,iter,step_diff);
		};
		Eigen::VectorXd gradient_method(Eigen::VectorXd a,Eigen::VectorXd b,Eigen::VectorXd c,double acc,int* iter,double step_diff){
			///a(i) and c(i) gives you the domain of the i-th coordinate
			if(*iter==0){
				std::random_device seed;
				std::mt19937 gen(seed());
				for(int i=0;i<dim;i++){
					std::uniform_real_distribution<> dis(a(i),c(i));
					b(i)=dis(gen);
				}
			}	
			for(int i=0;i<dim;i++){
				grad(i)=-(func(b+step_diff*dirs.row(i).transpose())-func(b-step_diff*dirs.row(i).transpose()))/(2*step_diff);
			}
			if(grad.norm()>acc){
				/// Defining the one-dimensional function assciated with the n-dimensional function for each direction
				red_func = [&](double lambda) {
					return func(b + lambda * (grad/grad.norm()));
				};
				minimization_1d.chg_func(red_func);
				iter_1d=0;
				b=b+(minimization_1d.interval_bisection(((a-b).transpose()*(grad/grad.norm())),0,((c-b).transpose()*(grad/grad.norm())),acc,&iter_1d))*(grad/grad.norm());
				std::cout<<" "<<a.transpose()<<" "<<b.transpose()<<" "<<c.transpose()<<" "<<" "<<grad.transpose()<<" "<<acc<<" "<<*iter<<std::endl;
				*iter++;
				return gradient_method(a,b,c,acc,iter,step_diff);
			}else
				return b;

		};
		Eigen::VectorXd conjugate_gradient_method(Eigen::VectorXd a,Eigen::VectorXd b,Eigen::VectorXd c,double acc,int* iter,double step_diff){
			///a(i) and c(i) gives you the domain of the i-th coordinate
			if(*iter==0){
				std::random_device seed;
				std::mt19937 gen(seed());
				for(int i=0;i<dim;i++){
					std::uniform_real_distribution<> dis(a(i),c(i));
					b(i)=dis(gen);
				}
				for(int i=0;i<dim;i++){
					grad(i)=-(func(b+step_diff*dirs.row(i).transpose())-func(b-step_diff*dirs.row(i).transpose()))/(2*step_diff);
				}
			}
			/// HERE THE CODE CAN BE MADE MORE READABLE
			if(grad.norm()>acc){
				/// Defining the one-dimensional function assciated with the n-dimensional function for each direction
				red_func = [&](double lambda) {
					return func(b + lambda * (grad/grad.norm()));
				};
				minimization_1d.chg_func(red_func);
				iter_1d=0;
				b=b+(minimization_1d.interval_bisection(((a-b).transpose()*(grad/grad.norm())),0,((c-b).transpose()*(grad/grad.norm())),acc,&iter_1d))*(grad/grad.norm());
				for(int i=0;i<dim;i++){
					gradnext(i)=-(func(b+step_diff*dirs.row(i).transpose())-func(b-step_diff*dirs.row(i).transpose()))/(2*step_diff);
				}
				if(gradnext.norm()>acc){
					red_func = [&](double lambda) {
						return func(b + lambda * ((gradnext/gradnext.norm())+std::pow((gradnext.norm()/grad.norm()),2.0)*(grad/grad.norm())));
					};
					minimization_1d.chg_func(red_func);
					iter_1d=0;
					b=b+(minimization_1d.interval_bisection_wolfe_condition(((a-b).transpose()*((gradnext/gradnext.norm())+std::pow((gradnext.norm()/grad.norm()),2.0)*(grad/grad.norm()))),0,((c-b).transpose()*((gradnext/gradnext.norm())+std::pow((gradnext.norm()/grad.norm()),2.0)*(grad/grad.norm()))),acc,&iter_1d,-std::pow(grad.norm(),2.0),-grad.transpose()*gradnext,100))*((gradnext/gradnext.norm())+std::pow((gradnext.norm()/grad.norm()),2.0)*(grad/grad.norm()));
					*iter++;
					return gradient_method(a,b,c,acc,iter,step_diff);
				}else
					return b;
			}else
				return b;
		};	
	};

///int main(int argc,char** argv){	
///	///Minimization_1d minimization(test_function_1);
///	///double acc=0.001;
///	///int *iter; iter=new int; *iter=0;
///	///std::cout<<minimization.interval_bisection(-1.0,0.0,1.0,acc,iter)<<std::endl;
///	///////std::cout<<minimization.parabolic(0.0,0.0,1.0,acc,iter)<<std::endl;
///	/////*iter=0;
///	/////double step_diff=0.0001;
///	//////std::cout<<minimization.Newton(0.0,0.0,1.0,acc,iter,step_diff)<<std::endl;
///	
///	
///	
///	int dim=2;
///	///it is better to express the n-dimensional functions explicitly in order to avoid error messages
///	Minimization_nd minimization(test_function_n,dim);
///	Eigen::VectorXd a(dim);
///	Eigen::VectorXd b(dim);
///	Eigen::VectorXd c(dim);
///	for(int i=0;i<dim;i++)
///		c(i)=1.0;
///	a.setZero();
///	Eigen::VectorXd previousb(dim);
///	double step_diff=1.0e-6;
///	double acc=0.001;
///	int iter=0;
///	///First Routine
///	///std::cout<<minimization.powell_process(a,b,c,previousb,acc,&iter,dim,step_diff)<<std::endl;
///	
///	////CHECK IF IN THE GRADIENT METHOD, THE VERSOR HAS TO BE CONSIDERED...
///
///	///Second Routine
///	///in dim=3 this routine can have memory problems....
///	///std::cout<<minimization.gradient_method(a,b,c,acc,&iter,step_diff)<<std::endl;
///
///	///Third Routine
///	std::cout<<minimization.conjugate_gradient_method(a,b,c,acc,&iter,step_diff)<<std::endl;
///	return 1;
///};
///
///
///


