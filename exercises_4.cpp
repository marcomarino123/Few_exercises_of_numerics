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

const double damp_ratio=0.1;
const double omega_0_2=3;

Eigen::VectorXd trans_op(Eigen::VectorXd x){
	Eigen::MatrixXd mat(x.size(),x.size());
	mat.setZero();
	mat.block(0,0,x.size()/2,x.size()/2)=-2*damp_ratio*omega_0_2*(Eigen::MatrixXd::Identity(x.size()/2,x.size()/2));
	mat.block(x.size()/2,0,x.size()/2,x.size()/2)=Eigen::MatrixXd::Identity(x.size()/2,x.size()/2);
	mat.block(0,x.size()/2,x.size()/2,x.size()/2)=-omega_0_2*(Eigen::MatrixXd::Identity(x.size()/2,x.size()/2));
	return mat*x;
};

const double a_freq_0=3.0;
const double b_freq_0=0.0;
const double freq_0=0.0;

Eigen::VectorXd td_trans_op(double t,Eigen::VectorXd x){
	Eigen::MatrixXd mat(x.size(),x.size());
	mat.setZero();
	mat.block(0,0,x.size()/2,x.size()/2)=-2*damp_ratio*(a_freq_0*std::cos(freq_0*t)+b_freq_0*std::sin(freq_0*t))*(Eigen::MatrixXd::Identity(x.size()/2,x.size()/2));
	mat.block(x.size()/2,0,x.size()/2,x.size()/2)=Eigen::MatrixXd::Identity(x.size()/2,x.size()/2);
	mat.block(0,x.size()/2,x.size()/2,x.size()/2)=-(a_freq_0*std::cos(freq_0*t)+b_freq_0*std::sin(freq_0*t))*(Eigen::MatrixXd::Identity(x.size()/2,x.size()/2));
	return mat*x;
};

/// Single Harmonic Oscillator n-dimensional
class DGLS{
private:
	Eigen::VectorXd x0;
	int dim;
	double dt;
	int num_steps;
	std::function<Eigen::VectorXd(Eigen::VectorXd)> trans_op;
	std::function<Eigen::VectorXd(double,Eigen::VectorXd)> td_trans_op;
public:
	DGLS(Eigen::VectorXd init_x0,std::function<Eigen::VectorXd(Eigen::VectorXd)> init_trans_op){
		x0=init_x0;
		dim=x0.size();
		trans_op=init_trans_op;
	};
	DGLS(Eigen::VectorXd init_x0,std::function<Eigen::VectorXd(double,Eigen::VectorXd)> td_init_trans_op){
		x0=init_x0;
		dim=x0.size();
		td_trans_op=td_init_trans_op;
	};
	Eigen::VectorXd euler_method(double init_dt,int init_num_steps,bool graphical_view){
		dt=init_dt;
		num_steps=init_num_steps;
		Eigen::VectorXd x_prev=x0;
		Eigen::VectorXd x_next(dim);
		if(graphical_view==false){
			for(int i=0;i<num_steps;i++){
				x_next=x_prev+dt*trans_op(x_prev);
				x_prev=x_next;
			}
		}else{
			sf::RenderWindow window(sf::VideoMode(800, 600)," ");
			window.setFramerateLimit(60);
			std::vector<sf::Vertex> points;	
			
			while(window.isOpen()){
				window.clear(sf::Color::Black);
				points.clear();
				for(int i=0;i<num_steps;i++){
					x_next=x_prev+dt*trans_op(x_prev);
					// Map the result (1d) to screen coordinates
					float screenX = (float(i)/float(num_steps)) * window.getSize().x;
					float screenY = window.getSize().y / 2 - x_prev(1) * 100; // Adjust this scaling if necessary
					points.emplace_back(sf::Vector2f(screenX,screenY), sf::Color::White);
					x_prev=x_next;
					window.draw(&points[0], points.size(), sf::LineStrip);
					window.display();
				}
				window.close();		
			}
		}
		return x_next;
	};
	Eigen::VectorXd rungekutta_method(double init_dt,int init_num_steps,bool graphical_view){
		dt=init_dt;
		num_steps=init_num_steps;
		Eigen::VectorXd x_prev=x0;
		Eigen::VectorXd x_next(dim);
		Eigen::VectorXd x_inter(dim);
		if(graphical_view==false){
			for(int i=0;i<num_steps;i++){
				x_inter=x_prev+(dt/2.0)*trans_op(x_prev);
				x_next=x_prev+dt*trans_op(x_inter);
				x_prev=x_next;
			}
		}else{
			sf::RenderWindow window(sf::VideoMode(800, 600)," ");
			window.setFramerateLimit(60);
			std::vector<sf::VertexArray> points(dim, sf::VertexArray(sf::LineStrip));
			std::vector<sf::Color> colors = {sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow, sf::Color::Magenta};
			while(window.isOpen()){
				sf::Event event;
        		while (window.pollEvent(event)){
        		    if (event.type == sf::Event::Closed)
        		        window.close();
        		}
				window.clear(sf::Color::Black);
				for(int i=0;i<num_steps;i++){
					x_inter=x_prev+(dt/2.0)*trans_op(x_prev);
					x_next=x_prev+dt*trans_op(x_inter);
					for(int j=dim/2;j<dim;j++){
						float screenX = (float(i)/float(num_steps)) * window.getSize().x;
						float screenY = window.getSize().y / 2 - x_prev(j) * 100 + (j-dim/2)*100; // Adjust this scaling if necessary
						points[j].append(sf::Vertex(sf::Vector2f(screenX, screenY),colors[j % colors.size()]));
						if (!points.empty()) 
                			window.draw(points[j]);
					} 
					window.display();
           			sf::sleep(sf::milliseconds(20)); 
					x_prev=x_next;
				}
				window.close();
			}
		}
		return x_next;
	};
	Eigen::VectorXd td_rungekutta_method(double init_dt,int init_num_steps,bool graphical_view,bool step_size_adj,double acc){
		dt=init_dt;
		num_steps=init_num_steps;
		Eigen::VectorXd x_prev=x0;
		Eigen::VectorXd x_next(dim);
		Eigen::VectorXd x_inter(dim);
		
		Eigen::VectorXd x_prev1=x0;
		Eigen::VectorXd x_next1(dim);
		Eigen::VectorXd x_inter1(dim);
		Eigen::VectorXd x_inter20(dim);
		int count;


		if(graphical_view==false){
			if(step_size_adj==true){
				for(int i=0;i<num_steps;i++){
					if(i<num_steps-2){
						count=0;
						while(count<2){
							x_inter=x_prev+dt*td_trans_op(i*dt,x_prev);
							x_next=x_prev+dt*td_trans_op(i*dt+(dt/2.0),x_prev+(x_inter/2.0));
							if(count==0)
								x_inter20=x_next;
							x_prev=x_next;
							count++;
						}
						x_inter1=x_prev1+2*dt*td_trans_op(i*2*dt,x_prev1);
						x_next1=x_prev1+2*dt*td_trans_op(i*2*dt+dt,x_prev1+(x_inter1/2.0));
						if((x_prev-x_next1).norm()>acc){
							dt=dt*std::pow((x_inter20.norm()/(x_prev-x_next1).norm()),0.5);
						}
						x_prev1=x_inter20;
						x_prev=x_inter20;
					}else{
						x_inter=x_prev+dt*td_trans_op(i*dt,x_prev);
						x_next=x_prev+dt*td_trans_op(i*dt+(dt/2.0),x_prev+(x_inter/2.0));
						x_prev=x_next;
					}
				}	
			}else{
				for(int i=0;i<num_steps;i++){
					x_inter=x_prev+dt*td_trans_op(i*dt,x_prev);
					x_next=x_prev+dt*td_trans_op(i*dt+(dt/2.0),x_prev+(x_inter/2.0));
					x_prev=x_next;

				}

			}
		}else{
			sf::RenderWindow window(sf::VideoMode(800, 600)," ");
			window.setFramerateLimit(60);
			std::vector<sf::VertexArray> points(dim, sf::VertexArray(sf::LineStrip));
			std::vector<sf::Color> colors = {sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow, sf::Color::Magenta};
			while(window.isOpen()){
				window.clear(sf::Color::Black);
				for(int i=0;i<num_steps;i++){
					x_inter=x_prev+dt*td_trans_op(i*dt,x_prev);
					x_next=x_prev+dt*td_trans_op(i*dt+(dt/2.0),x_prev+(x_inter/2.0));
					for(int j=dim/2;j<dim;j++){
						float screenX = (float(i)/float(num_steps)) * window.getSize().x;
						float screenY = window.getSize().y / 2 - x_prev(j) * 100 + (j-dim/2)*100; // Adjust this scaling if necessary
						points[j].append(sf::Vertex(sf::Vector2f(screenX, screenY),colors[j % colors.size()]));
						if (!points.empty()) 
                			window.draw(points[j]);
					}
					window.display();
           			sf::sleep(sf::milliseconds(20));
					sf::Event event;
					while (window.pollEvent(event)){
						if (event.type == sf::Event::Closed)
							window.close();
					}
					x_prev=x_next;	
				}
			}
		}
		return x_next;
	};


};


int main(int argc,char** argv){	
	int dim=3;
	Eigen::VectorXd x0(dim*2);
	for(int i=0;i<dim;i++){
		///velocity
		x0(i)=0;
		///position
		x0(i+dim)=1;
	}
	///BE CAREFULL THE DIM SAVED IN DGLS IS THE ONE OF THE VECTORS, I.E. DIM*2

	DGLS harmonic_oscillator(x0,trans_op);
	double dt=0.1;
	int num_steps=200;
	bool graphical_view=false;
	///std::cout<<harmonic_oscillator.euler_method(dt,num_steps,graphical_view)<<std::endl;
	///std::cout<<harmonic_oscillator.rungekutta_method(dt,num_steps,graphical_view)<<std::endl;

	DGLS td_harmonic_oscillator(x0,td_trans_op);
	//double dt=0.1;
	//int num_steps=300;
	//bool graphical_view=true;
	bool step_size_adj=true;
	double acc=1.0e-4;
	std::cout<<td_harmonic_oscillator.td_rungekutta_method(dt,num_steps,graphical_view,step_size_adj,acc)<<std::endl;

	

	return 1;
};





