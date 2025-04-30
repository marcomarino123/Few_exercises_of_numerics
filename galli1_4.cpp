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

const double pi = 3.14159265358979323846;
const double mu=0.0;
const double sigma=0.5;
const double Ddt=0.001;


void lag_fib(std::vector<int>* sequence,int dimension,int seed,const int r,const int m,const int s){
    int max=std::max(r,s);
    int min=std::min(r,s);
    (*sequence)[0]=seed;
    for(int i=1;i<min;i++)
        (*sequence)[i]=(*sequence)[i-1]%m;
    for(int i=min;i<min+max;i++)
        (*sequence)[i]=(*sequence)[i-min]%m;
    for(int i=min+max;i<dimension;i++)
        (*sequence)[i]=((*sequence)[i-min]+(*sequence)[i-max])%m;
};

std::vector<std::vector<double>> sphere_3d(double rad_max,int seed_theta,int seed_phi,int seed_rho,int dimension,const int r,const int m,const int s){
    std::vector<int> seq_theta(dimension);
    std::vector<int> seq_phi(dimension);
    std::vector<int> seq_rho(dimension);
    std::vector<std::vector<double>> xyz(3,std::vector<double>(dimension));
    lag_fib(&seq_theta,dimension,seed_theta,r,m,s);
    lag_fib(&seq_phi,dimension,seed_phi,r,m,s);
    lag_fib(&seq_rho,dimension,seed_rho,r,m,s);
    double theta; double phi; double rho;
    /// producing uniform distribution inside a sphere of radius rad_max
    /// p(\theta)d\theta=(1/2)sin(\theta)d\theta
    /// p(\phi)d\phi=(1/(2\pigreco))d\phi
    /// p(r)dr=(3r^2/r_max^3)dr
    /// inverting cumulative
    /// F(\theta)=(1/2)(1-cos\theta) 
    /// F(\phi)=\phi/(2\pigreco)
    /// F(r)=(r/r_max)^3
    for(int i=0;i<dimension;i++){
        theta=acos(1-2*(static_cast<double>(seq_theta[i])/m));
        phi=2*pi*static_cast<double>(seq_phi[i])/m;
        rho=rad_max*std::pow((static_cast<double>(seq_rho[i])/m),1.0/3.0);
        xyz[0][i]=rho*sin(theta)*cos(phi);
        xyz[1][i]=rho*sin(theta)*sin(phi);
        xyz[2][i]=rho*cos(theta);
    }
    return xyz;
};

std::vector<std::vector<double>> random_walk(std::vector<double> initial_xyz,double step_dr,int seed_theta,int seed_phi,int dimension,const int r,const int m,const int s){
    ///random walk in 3D
    std::vector<int> seq_theta(dimension);
    std::vector<int> seq_phi(dimension);
    lag_fib(&seq_theta,dimension,seed_theta,r,m,s);
    lag_fib(&seq_phi,dimension,seed_phi,r,m,s);
    double theta; double phi;
    /// p(\theta)d\theta=(1/2)sin(\theta)d\theta
    /// p(\phi)d\phi=(1/(2\pigreco))d\phi
    /// inverting cumulative
    /// F(\theta)=(1/2)(1-cos\theta) 
    /// F(\phi)=\phi/(2\pigreco)
    std::vector<std::vector<double>> xyz(dimension+1,std::vector<double>(3));
    for(int j=0;j<3;j++)
        xyz[0][j]=initial_xyz[j];

    for(int i=0;i<dimension;i++){
        theta=acos(1-2*(static_cast<double>(seq_theta[i])/m));
        phi=2*pi*static_cast<double>(seq_phi[i])/m;
        xyz[i+1][0]=xyz[i][0]+step_dr*sin(theta)*cos(phi);
        xyz[i+1][1]=xyz[i][1]+step_dr*sin(theta)*sin(phi);
        xyz[i+1][2]=xyz[i][2]+step_dr*cos(theta);
    }
    return xyz;
};

double new_probability(double x){
    return (std::exp(-std::pow(x-mu,2)/(2*std::pow(sigma,2))))/(std::sqrt(2*pi*std::pow(sigma,2)));
};

std::vector<double> Metropolis(const double min,const double max,double radius_T,int number_equilibration,int number_measuraments,double (*new_probability)(double),int sparse,bool smart){
    std::function<double(double,double)> ratio;
    if(smart==false){
        ratio = [&](double x,double y){
		    return 1;
		};
    }else{
        ratio = [&](double x,double y){
            return std::exp(-0.5*((-x-y-2*mu)/std::pow(sigma,2))*(x-y-Ddt*((x-y)/std::pow(sigma,2))));
		};  
    }
    std::random_device seed1;
    std::mt19937 gen1(seed1());
    std::uniform_real_distribution<> dis1(min, max);

    std::random_device seed2;
    std::mt19937 gen2(seed2());
    std::uniform_real_distribution<> dis2(-1.0, 1.0);

    std::vector<double> walk(number_measuraments);

    double x_prec;
    double x_next;
    double a;
    double r;
    
    x_prec = dis1(gen1);
    for(int i=0;i<number_equilibration;i++){    
        x_next = x_prec+dis2(gen2)*radius_T;
        a=std::min(1.0,ratio(x_prec,x_next)*(*new_probability)(x_next)/(*new_probability)(x_prec));
        r = dis1(gen1);
        if(a<r)
            x_next = x_prec;
        x_prec = x_next;
    }
    int i=0;
    int j=0;
    while(i<number_measuraments){
        x_next = x_prec+dis2(gen2)*radius_T;
        a=std::min(1.0,ratio(x_prec,x_next)*(*new_probability)(x_next)/(*new_probability)(x_prec));
        r = dis1(gen1);
        if(a<r)
            x_next = x_prec;
        x_prec = x_next;
        if(j==sparse){
            walk[i] = x_prec;
            i++;
            j=0;
        }else
            j++;
    }
    return walk;
};

void studying_equilibration_correlation(std::vector<double> walk,int number_measuraments){
    /// Equilibration
    std::vector<double> average(number_measuraments);
    for(int i=0;i<number_measuraments;i++)
        average[i]=0.0;
    average[0]=walk[0];
    for(int i=1;i<number_measuraments;i++)
        for(int j=0;j<=i;j++)
            average[i]+=walk[j]/(i+1);
    std::cout<<"measuring equilibration"<<std::endl;
    for(int i=0;i<number_measuraments;i++)
        std::cout<<i<<" "<<average[i]<<std::endl;
    /// Correlations
    std::vector<std::vector<double>> correlation(number_measuraments,std::vector<double>(number_measuraments));
    double average_i=0.0;
    double average_i_k=0.0;
    for(int i=0;i<number_measuraments;i++)
        for(int l=0;l<number_measuraments;l++)
            correlation[i][l]=0.0;  
    for(int l=0;l<number_measuraments;l++)
        for(int i=0;i<number_measuraments;i++){
            average_i=0.0;
            average_i_k=0.0;
            for(int m=0;m<i;m++){
                average_i+=walk[m]/(i+1);
                for(int k=l;k<i;k++){
                    average_i_k+=walk[k]/(i+1-l);
                    correlation[i][l]+=walk[i]*walk[k]/(i+1-l);
                }
                correlation[i][l]=correlation[i][l]/(i+1);
            }
            correlation[i][l]=correlation[i][l]-average_i*average_i_k;
        }     
    ///the first index is measuring how many measuraments are considered in the average
    ///the second index is measuring how far are the two measurements with respect to which the correlation is measured      
    std::cout<<"measuring correlations"<<std::endl;
    for(int i=0;i<number_measuraments;i++)
        std::cout<<i<<" "<<correlation[number_measuraments-1][i]<<std::endl;
    
};

void draw_histogram(std::vector<double> walk, const double min, const double max, int number_measuraments, double step_histogram) {
    int number_bins = static_cast<int>(std::ceil((max - min) / step_histogram));
    std::vector<int> frequencies(number_bins, 0);
    double total=0.0;

    // Fill frequency bins
    for (int i = 0; i < number_measuraments; i++) {
        double value = walk[i];
        if (value < min || value >= max) continue; // Skip values outside the desired range
        int bin = static_cast<int>((value - min) / step_histogram);
        frequencies[bin]++;
        total++;
    }
    // Draw histogram
    for (int j = 0; j < number_bins; j++) {
        double bin_start = min + j * step_histogram;
        std::cout << bin_start << " | ";
        for (int i = 0; i < (static_cast<double>(frequencies[j])/total)*100; ++i) {
            std::cout << '.';
        }
        std::cout << " (" <<  int((static_cast<double>(frequencies[j])/total)*100) << ")\n";
    }
};

double radius_probability(std::vector<double> xyz){
    double r=std::sqrt(std::pow(xyz[0],2)+std::pow(xyz[1],2)+std::pow(xyz[2],2));
    return r*std::sqrt(std::pow(xyz[0],2)+std::pow(xyz[1],2))*std::exp(-2*r)/pi;
};


std::vector<std::vector<double>> sampling_particular_probability(double radius_max,double radius_T,int number_equilibration,int number_measuraments,double (*radius_probability)(std::vector<double>)){
/// sampling to the probability p(r,\theta,\phi)=r^2 sin\theta e^{-2r}/pigreco
/// with Metropolis algorithm in cartesian coordiantes [x,y,z] 
    std::random_device seedx;
    std::random_device seedy;
    std::random_device seedz;
    std::mt19937 genx(seedx());
    std::mt19937 geny(seedy());
    std::mt19937 genz(seedz());
    std::uniform_real_distribution<> dis(-1,1);
    std::random_device Tseedx;
    std::random_device Tseedy;
    std::random_device Tseedz;
    std::mt19937 Tgenx(Tseedx());
    std::mt19937 Tgeny(Tseedy());
    std::mt19937 Tgenz(Tseedz());
    std::uniform_real_distribution<> disT(-1,1);
    std::random_device seedr;
    std::mt19937 genr(seedr());
    std::uniform_real_distribution<> disr(0,1);

    std::vector<double> xyz_prec(3);
    std::vector<double> xyz_next(3);

    double a;
    double r;
    
    xyz_prec[0] = dis(genx)*radius_max;
    xyz_prec[1] = dis(geny)*radius_max;
    xyz_prec[2] = dis(genz)*radius_max;

    for(int i=0;i<number_equilibration;i++){   
        xyz_next[0] = xyz_prec[0]+disT(Tgenx)*radius_T;
        xyz_next[1] = xyz_prec[1]+disT(Tgeny)*radius_T;
        xyz_next[2] = xyz_prec[2]+disT(Tgenz)*radius_T;
        std::cout<<xyz_next[0]<<" "<<xyz_next[1]<<" "<<xyz_next[2]<<std::endl;
        a=std::min(1.0,(*radius_probability)(xyz_next)/(*radius_probability)(xyz_prec));
        r = disr(genr);
        if(a<r)
            for(int k=0;k<3;k++)
                xyz_next[k] = xyz_prec[k];
        for(int k=0;k<3;k++)
            xyz_prec[k] = xyz_next[k];
    }
    
    std::vector<std::vector<double>> walk(3,std::vector<double>(number_measuraments));
    for(int i=0;i<number_measuraments;i++){   
        xyz_next[0] = xyz_prec[0]+disT(Tgenx);
        xyz_next[1] = xyz_prec[1]+disT(Tgeny);
        xyz_next[2] = xyz_prec[2]+disT(Tgenz);
        a=std::min(1.0,(*radius_probability)(xyz_next)/(*radius_probability)(xyz_prec));
        r = disr(genr);
        if(a<r)
            for(int k=0;k<3;k++)
                xyz_next[k] = xyz_prec[k];
        for(int k=0;k<3;k++)
            xyz_prec[k] = xyz_next[k];
        for(int k=0;k<3;k++)
            walk[k][i]=xyz_prec[k];
    }
    return walk;
};

///int main(int argc,char** argv){	
///    ///int seed=10;
///    ///int r=24;
///    ///int s=55;
///    ///int m=1771899;
///    ///int dimension=10000;
///    ///double number;
///    ///double average=0.0;
///    ///double variance=0.0;
///    ///double intervals=1000;
///    ///std::vector<int> interval_occupation(intervals);
///    ///for(int i=0;i<intervals;i++)
///    ///    interval_occupation[i]=0;
///    ///double uniformity=0.0;
///    ///std::vector<int> seq(dimension);
///    ///lag_fib(&seq,dimension,seed,r,m,s);
///    ///for(int i=0;i<dimension;i++){
///    ///    number=static_cast<double>(seq[i])/m;
///    ///    average=average+number;
///    ///    variance=variance+std::pow((number-0.5),2);
///    ///    for(int j=1;j<intervals+1;j++)
///    ///        if(number<static_cast<double>(j)/intervals){
///    ///            interval_occupation[j-1]+=1;
///    ///            break;
///    ///        }
///    ///}
///    ///for(int j=0;j<intervals;j++){
///    ///    //std::cout<<interval_occupation[j]<<" "<<static_cast<double>(dimension)/intervals<<std::endl;
///    ///    uniformity+=pow(interval_occupation[j]-static_cast<double>(dimension)/intervals,2)/static_cast<double>(dimension)/intervals;
///    ///}
///    ///variance=variance/dimension;
///    ///average=average/dimension;
///    ///std::cout<<" uniformity "<<uniformity<<std::endl;
///    ///std::cout<<" average "<<average<<std::endl;
///    ///std::cout<<" variance "<<variance<<std::endl;
///    ///double rad_max=1240;
///    ///int seed_theta=10;
///    ///int seed_phi=24;
///    ///int seed_rho=67;
///    //std::vector<std::vector<double>> xyz=sphere_3d(rad_max,seed_theta,seed_phi,seed_rho,dimension,r,m,s);
///    ///for(int i=0;i<dimension;i++)
///    ///    std::cout<<xyz[0][i]<<std::endl;
///
///    ////METROPOLIS
///    double radius=0.01;
///    int number_equilibration=1000000;
///    int number_measuraments=1000000;
///    double min=-5.0;
///    double max=5.0;
///    int sparse=10;
///    bool smart=false;
///    std::vector<double> walk=Metropolis(min,max,radius,number_equilibration,number_measuraments,new_probability,sparse,smart);
///    ///studying_equilibration_correlation(walk,number_measuraments);
///    //double step_histogram=0.5;
///    //draw_histogram(walk,min,max,number_measuraments,step_histogram);
///    //for(int i=0;i<number_iterations;i++)
///    //    std::cout<<walk[i]<<std::endl;
///
///    double radius_T=10;
///    double radius_max=100;
///    std::vector<std::vector<double>> walkr =  sampling_particular_probability(radius_max,radius_T,number_equilibration,number_measuraments,radius_probability);
///    std::vector<double> radiusr(3);
///    for(int k=0;k<3;k++){
///        radiusr[k]=0.0;
///        for(int i=0;i<number_measuraments;i++)
///            radiusr[k]+=walkr[k][i]/number_measuraments;
///    }
///    std::cout<<radiusr[0]<<" "<<radiusr[1]<<" "<<radiusr[2]<<std::endl;
///
///    return 1;
///};