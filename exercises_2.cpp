#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include </usr/local/Eigen>
#include </home/marco-marino/Desktop/TA_Computational_Physics/exercises_1.cpp>
#include </home/marco-marino/delaunator-cpp/include/delaunator.hpp>
#include <cstdio>

double integrand(double x){
	return std::cos(x);
};

double integrand(double x,double y){
	return std::cos(x)*cos(y);
};

class Integral1D
{
private:
	double a,b;
	double (*func)(double);
public:
	Integral1D(double init_a,double init_b,double (*init_func)(double)){
		a=init_a;
		b=init_b;
		func=init_func;
	};
	double Newton_Cotes(int number_intervals){
		Eigen::VectorXd Ii(number_intervals+1);
		double h=(b-a)/number_intervals;
		for(int i=0;i<number_intervals+1;i++)
			Ii(i)=a+h*i;
		double result=0;
		///std::cout<<"Rectangular"<<std::endl;
		///for(int i=0;i<number_intervals;i++)
		///	result+=(*func)((Ii(i)+Ii(i+1))/2);
		///result=result*h;
		///std::cout<<result<<std::endl;
		///result=0.0;
		///std::cout<<"Trapezzoidal"<<std::endl;
		for(int i=0;i<number_intervals;i++)
			result+=((*func)(Ii(i))+(*func)(Ii(i+1)))/2;
		result+=((*func)(Ii(0))+(*func)(Ii(number_intervals+1)))/2;
		result=result*h;
		//std::cout<<result<<std::endl;
		return result;
	};
	double Newton_Cotes_h(double h){
		int number_intervals=(b-a)/h;
		Eigen::VectorXd Ii(number_intervals+1);
		for(int i=0;i<number_intervals+1;i++)
			Ii(i)=a+h*i;
		double result=0;
		for(int i=0;i<number_intervals;i++)
			result+=((*func)(Ii(i))+(*func)(Ii(i+1)))/2;
		result+=((*func)(Ii(0))+(*func)(Ii(number_intervals+1)))/2;
		result=result*h;
		return result;
	};
	void Romberg(double first_h,double epsilon_h,int number_h){
		Eigen::VectorXd vector_h(number_h+1);
		Eigen::VectorXd b(number_h+1);
		Eigen::MatrixXd mat(number_h+1,number_h+1);
		///Defining different h
		for(int i=0;i<number_h+1;i++)
			vector_h(i)=first_h+epsilon_h*i;
		///Defining the matrix \tau_i h^{2i}
		for(int i=0;i<number_h+1;i++)
			for(int j=0;j<number_h+1;j++)
				mat(i,j)=std::pow(vector_h(i),2.0*j);
		///Defining the b of the 
		for(int i=0;i<number_h+1;i++)
			b(i)=Newton_Cotes_h(vector_h(i));
		LGS lgs(mat,b,number_h+1,number_h+1);
		lgs.LU_decomposition_Eigen_routine(number_h+1,number_h+1,mat,b);
	};
};
// Define a struct for points
struct point{
    double x,y;
};
// Function to check if a point is on the boundary of a segment
bool onSegment(point p, point q, point r) {
    return (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
            q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y));
}
// Function to find the orientation of ordered triplet (p, q, r)
// Returns:
// 0 -> p, q, r are collinear
// 1 -> Clockwise
// 2 -> Counterclockwise
int orientation(point p, point q, point r) {
    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;   // Collinear
    return (val > 0) ? 1 : 2; // Clockwise or Counterclockwise
}
// Function to check if two segments (p1,q1) and (p2,q2) intersect
bool doIntersect(point p1, point q1, point p2, point q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4) return true;

    // Special Cases
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;

    return false; // No intersection
}
// Function to check if a point is inside a polygon using Ray-Casting method
bool check_inout(const std::vector<point>& polygon, point p) {
    int n = polygon.size();
    if (n < 3) return true; // A polygon must have at least 3 sides

    // Create a point outside the polygon for the ray test
    point extreme = {1e9, p.y};

    int count = 0, i = 0;
    do {
        int next = (i + 1) % n;

        // Check if the line segment intersects with the ray
        if (doIntersect(polygon[i], polygon[next], p, extreme)) {
            // If the point is on the edge of the polygon, return false (inside)
            if (orientation(polygon[i], p, polygon[next]) == 0)
                return !onSegment(polygon[i], p, polygon[next]);
            count++;
        }
        i = next;
    } while (i != 0);

    // If count is even, point is outside
    return (count % 2 == 0);
}

class Integral2D
{
private:
	/// Pair of points which define the contour of the integration Area
	std::vector<point> contour;
	/// Integrand
	double (*func)(double,double);
public:
	Integral2D(std::vector<point> init_contour,double (*init_func)(double,double)){
		contour=init_contour;
		func=init_func;
	};
	double random_points(int number_of_points){
		std::vector<point> all_points(number_of_points+contour.size());
		std::random_device rd; std::mt19937 gen(rd());
   		/// Finding max x, and max y and min x and min y
		std::vector<double> min_max_x(2);
		std::vector<double> min_max_y(2);
		min_max_x[0]=0; min_max_x[1]=1e+9;
		min_max_y[0]=0; min_max_y[1]=1e+9;
		for(int i=0;i<contour.size();i++){
			if(contour[i].x<min_max_x[0])
				min_max_x[0]=contour[i].x;
			if(contour[i].x>min_max_x[1])
				min_max_x[1]=contour[i].x;
			if(contour[i].y<min_max_y[0])
				min_max_y[0]=contour[i].y;
			if(contour[i].y>min_max_y[1])
				min_max_y[1]=contour[i].y;
		}
		std::uniform_real_distribution<> disx(min_max_x[0],min_max_x[1]); /// distribution in range [-1,1]
		std::uniform_real_distribution<> disy(min_max_y[0],min_max_y[1]); /// distribution in range [-1,1]
		/// Covering the area with points
		int covering_flag=0; int i=0;
		while(covering_flag==0){
			all_points[i].x=disx(gen);
			all_points[i].y=disy(gen); 
			if(check_inout(contour,all_points[i])==true)
				i++;
			if(i==number_of_points)
				covering_flag=1;
		}
		for(int j=0;j<contour.size();j++){
			all_points[j+number_of_points].x=contour[j].x;
			all_points[j+number_of_points].y=contour[j].y;
		}
		/// Summing the function on the different points
		double sum=0.0;
		for(int j=0;j<contour.size()+number_of_points;j++)
			sum+=(*func)(all_points[j].x,all_points[j].y);
		return sum;
	};
};

int main(int argc,char** argv){	
	return 1;
};





