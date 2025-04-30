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
#include </usr/local/Eigen>
#include <omp.h>

const double pi = 3.14159265358979323846;

double wiener_probability(double x,double t){
    return (1.0/std::sqrt(2*pi*t))*std::exp(-std::pow(x,2)/(2*t));
};

double deterministic_function(double t){
    return 1.0;
};

std::vector<std::vector<double>> Metropolis(const int different_metropolis,const double min,const double max,double radius_T,int number_equilibration,int number_measuraments,double initial_time,double dt,double (*wiener_probability)(double,double),int sparse){
    // not considering intial time
    number_equilibration=number_equilibration+1;
    number_measuraments=number_measuraments+1;

    std::vector<std::vector<double>> w(different_metropolis,std::vector<double>(number_measuraments));

    #pragma omp parallel for shared(w)
    for (int m = 0; m < different_metropolis; m++){
        // Local variables
        double x_prec;
        double x_next;
        double a;
        double r;
        int j;
        // Local random generators
        std::random_device rd;
        std::mt19937 gen1(rd());
        std::mt19937 gen2(rd());
        std::uniform_real_distribution<> dis1(min, max);
        std::uniform_real_distribution<> dis2(-1.0, 1.0); 
        x_prec = dis1(gen1);
        for(int k=1;k < number_equilibration;k++){
            x_next = x_prec + dis2(gen2) * radius_T;
            a = std::min(1.0, (*wiener_probability)(x_next, k * dt - initial_time) / (*wiener_probability)(x_prec, (k - 1) * dt - initial_time));
            if (a < dis1(gen1))
                x_next = x_prec;
            x_prec = x_next;
        }
        for(int k=1;k < number_equilibration;k++){
            j=0;
            while(j<sparse){
                x_next = x_prec + dis2(gen2) * radius_T;
                a = std::min(1.0, (*wiener_probability)(x_next, k * dt - initial_time) / (*wiener_probability)(x_prec, (k - 1) * dt - initial_time));
                if (a < dis1(gen1))
                    x_next = x_prec;
                x_prec = x_next;
                j++;
            }
            w[m][k] = x_prec;
        }
    }
    return w;
};

double stochastic_integration_deterministic_function(int different_metropolis,double (*deterministic_function)(double),std::vector<std::vector<double>> w,double dt,double initial_time,int number_measuraments,double alpha){
    std::vector<double> integral(different_metropolis);
    double averaging=0;
    int m=0;
    int i;
    while(m<different_metropolis){
        i=1;
        integral[m]=0;
        while(i<number_measuraments){
            integral[m]+=deterministic_function(initial_time+alpha*i*dt+(1-alpha)*(i-1)*dt)*(w[m][i]-w[m][i-1]);
            i++;
        }
        averaging+=integral[m];
        m++;
    }
    averaging=averaging/different_metropolis;
    return averaging;
};

/// Ito process
/// dx(t)=a[x(t),t]dt+b[x(t),t]dW(t)
/// x(t)=x(t_0)+\int_{t_0}^t(a[x(t'),t'])dt'+\int_{t_0}^t(b[x(t'),t'])dW(t')
/// defining here a[x(t),t] and b[x(t),t]
const double k=0;
double a(double x,double t){
    return -k*std::pow(x,2);
};
const double D=0.01;
double b(double x,double t){
    return std::sqrt(2*D);
};
std::vector<std::vector<double>>  Euler_Maruyama_sde_ito(double initial_condition,int different_metropolis,double (*a)(double,double),double (*b)(double,double),std::vector<std::vector<double>> w,double dt,double initial_time,int number_measuraments,double alpha){
    /// Be careful Metropolis distribution shoud have variance dt (so initial_time=0.0)
    std::vector<std::vector<double>> x_t(different_metropolis,std::vector<double>(number_measuraments));
    for(int m=0;m<different_metropolis;m++){
        x_t[m][0] = initial_condition;
        for(int i=1;i<number_measuraments;i++)
            x_t[m][i]+=x_t[m][i-1]+(*a)(x_t[m][i-1],(i-1)*dt+initial_time)*dt+(*b)(x_t[m][i-1],(i-1)*dt+initial_time)*w[m][i-1];
    }
    return x_t;
};

/// MILSTEIN METHOD ...

const double epsilon=0.0001; 
/// sthocastic function
double f(double x,double t){
    return std::log(x+epsilon);
};
std::vector<std::vector<double>> Euler_Maruyama_stochastic_integration_stocastic_function(double initial_condition,int different_metropolis,double (*a)(double,double),double (*b)(double,double),std::vector<std::vector<double>> w,double dt,double initial_time,int number_measuraments,double alpha){
    /// Be careful Metropolis distribution shoud have variance dt (so initial_time=0.0)
    std::vector<std::vector<double>> x_t=Euler_Maruyama_sde_ito(initial_condition,different_metropolis,a,b,w,dt,initial_time,number_measuraments,alpha);
    std::vector<std::vector<double>> f_xt(different_metropolis,std::vector<double>(number_measuraments));
    #pragma omp parallel for shared(w)
    for(int m=0;m<different_metropolis;m++){
        f_xt[m][0] = f(initial_condition,initial_time);
        for(int i=1;i<number_measuraments;i++)
            f_xt[m][i] = (*a)(x_t[m][i-1],(i-1)*dt+initial_time)*(1/(x_t[m][i-1]+epsilon)) - 0.5*std::pow((*b)(x_t[m][i-1],(i-1)*dt+initial_time),2)*(1/std::pow(x_t[m][i-1]+epsilon,2))*dt +
                + (*b)(x_t[m][i-1],(i-1)*dt+initial_time)*(1/(x_t[m][i-1]+epsilon))*w[m][i-1];
    }
    return f_xt;
};
/// here I use directly the ito formula of the integral, and not an euler approach
std::vector<double> ito_stochastic_integration_stocastic_function(double initial_condition,int different_metropolis,double (*a)(double,double),double (*b)(double,double),std::vector<std::vector<double>> w,double dt,double initial_time,int number_measuraments,double alpha){
    /// Be careful Metropolis distribution shoud have variance dt (so initial_time=0.0)
    std::vector<std::vector<double>> x_t=Euler_Maruyama_sde_ito(initial_condition,different_metropolis,a,b,w,dt,initial_time,number_measuraments,alpha);
    std::vector<double> int_f_xt(different_metropolis);
    for(int m=0;m<different_metropolis;m++){
        int_f_xt[m] = f(initial_condition,initial_time);
        /// to the deterministic integral Newton-Cotes method has been applied
        for(int i=1;i<number_measuraments;i++)
            int_f_xt[m] += 0.5*((*a)(x_t[m][i-1],(i-1)*dt+initial_time)*(1/(x_t[m][i-1]+epsilon)) - 0.5*std::pow((*b)(x_t[m][i-1],(i-1)*dt+initial_time),2)*(1/std::pow(x_t[m][i-1]+epsilon,2))+
                (*a)(x_t[m][i],i*dt+initial_time)*(1/(x_t[m][i]+epsilon)) - 0.5*std::pow((*b)(x_t[m][i],i*dt+initial_time),2)*(1/std::pow(x_t[m][i]+epsilon,2))) +
                + (*b)(x_t[m][i-1],(i-1)*dt+initial_time)*(1/(x_t[m][i-1]+epsilon))*(w[m][i]-w[m][i-1]);
    }
    return int_f_xt;
};

int main(int argc,char** argv){	

    ///extracting a Wiener process from initial time t_0 to time t
    ///where time t = number_measuraments*dt+t_0
    ///each element correspond to a w(t)
    double radius_T=0.01;
    int number_equilibration=10000;
    int number_measuraments=10000;
    double initial_time=0;
    double dt=0.0001;
    double min=-1.0;
    double max=1.0;
    int sparse=10;
    int different_metropolis=1000;
    std::vector<std::vector<double>> w=Metropolis(different_metropolis,min,max,radius_T,number_equilibration,number_measuraments,initial_time,dt,wiener_probability,sparse);

    //considering a stocastic integral of a deterministic function of time
    double alpha=0.50;
    //std::cout<<stocastic_integration_deterministic_function(different_metropolis,deterministic_function,w,dt,initial_time,number_measuraments,alpha)<<std::endl;

    //considering ito sde
    double initial_condition=1.0;
    std::vector<std::vector<double>> x_t=Euler_Maruyama_sde_ito(initial_condition,different_metropolis,a,b,w,dt,initial_time,number_measuraments,alpha);
    std::vector<double> average_x_t(number_measuraments);
    for(int i=0;i<number_measuraments;i++){
        average_x_t[i]=0;    
        for(int m=0;m<different_metropolis;m++)
            average_x_t[i]+=x_t[m][i];
        average_x_t[i]=average_x_t[i]/different_metropolis;
        std::cout<<average_x_t[i]<<std::endl;
    }
    
    return 1;
};