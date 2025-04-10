#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include </usr/local/Eigen>

int sign(double x) {
    return (x == 0) ? 0 : static_cast<int>(std::copysign(1, x));
}

class LGS
{
private:
	Eigen::MatrixXd mat;
	Eigen::MatrixXd b;
	int m; int n;	
public:
	/// Constructor 1: complex LGS
	LGS(Eigen::MatrixXd r_init_mat,Eigen::MatrixXd i_init_mat,Eigen::MatrixXd r_init_b,Eigen::MatrixXd i_init_b){
		m=r_init_mat.rows();
		n=r_init_mat.cols();
		mat.resize(m*2,n*2);
		b.resize(m*2,1);
		
		mat.block(0,0,m,n)=r_init_mat;
		mat.block(m,n,m,n)=r_init_mat;
		mat.block(0,n,m,n)=-i_init_mat;
		mat.block(m,0,m,n)=i_init_mat;

		b.block(0,0,m,1)=r_init_b;
		b.block(m,0,m,1)=i_init_b;
		m=m*2;
		n=n*2;
	};
	/// Constructor 2: real LGS
	LGS(Eigen::MatrixXd init_mat,Eigen::MatrixXd init_b){
		m=init_mat.rows();
		n=init_mat.cols();
		mat.resize(m,n);
		b.resize(m,1);
	};
	Eigen::MatrixXd pull_mat(){
		return mat;
	};
	Eigen::MatrixXd pull_b(){
		return b;
	};
	int pull_rows(){
		return m;
	};
	int pull_cols(){
		return n;
	};
	/// LU decomposition
	void LU_decomposition(int init_m,int init_n,Eigen::MatrixXd init_mat,Eigen::MatrixXd init_b){
		if(init_m==init_n){

			
		}


	};
	void LU_decomposition_Eigen_routine(int init_m,int init_n,Eigen::MatrixXd init_mat,Eigen::MatrixXd init_b){
		if(init_m==init_n){
			Eigen::FullPivLU<Eigen::MatrixXd> lu(init_mat);
			Eigen::MatrixXd l(init_m,init_n); Eigen::MatrixXd u(init_m,init_n);
			l.setZero(); u.setZero();
			for(int j=0;j<init_n;j++){
				l(j,j)=1.0;
				for(int i=0;i<init_m-j;i++)
					l(i,j)=lu.matrixLU()(i,j);
			}
			for(int i=0;i<init_m;i++){
				for(int j=0;j<init_n;j++)
					u(i,j)=lu.matrixLU()(i,j);
			}
			Eigen::MatrixXd x=u.inverse()*l.inverse()*lu.permutationP().inverse()*init_b;
			std::cout<<x<<std::endl;
		}else{
			Eigen::MatrixXd fin_mat=init_mat.transpose()*init_mat;
			Eigen::MatrixXd fin_b=init_mat.transpose()*init_b;
			LU_decomposition_Eigen_routine(init_n,init_n,fin_mat,fin_b);
			std::cout<<"Solution found:Least-Squat-Minimum"<<std::endl;
			}
	};
	void SVD_decomposition_Eigen_routine(int init_m,int init_n,Eigen::MatrixXd init_mat,Eigen::MatrixXd init_b){
		/// A = U*W*V^T
		Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::ComputeFullU|Eigen::ComputeFullV> svd(init_mat);
		Eigen::MatrixXd u = svd.matrixU();
		Eigen::MatrixXd v = svd.matrixV();
		Eigen::VectorXd w_diag = svd.singularValues();
		Eigen::MatrixXd inv_w(init_n,init_m);
		/// Flag to distinguish the solutions
		int solution_found=0;
		inv_w.setZero();
		/// Checking if there are zero values
		int count=0;
		if(init_n<init_m){
			for(int i=0;i<init_n;i++){
				if(w_diag[i]!=0)
					inv_w(i,i)=1/w_diag(i);
				else{
					inv_w(i,i)=0;
					count+=1;
				}
			}
		}else{
			for(int i=0;i<init_m;i++){
				if(w_diag[i]!=0)
					inv_w(i,i)=1/w_diag(i);
				else{
					inv_w(i,i)=0;
					count+=1;
				}
			}
		}
		Eigen::VectorXd vec(init_m);
		for(int i=0;i<init_n;i++){
			vec.setZero();
			for(int j=0;j<init_m;j++)
				for(int k=0;k<init_m;k++)
					vec(j)+=init_b(j)-init_b(k)*init_mat(k,i)*init_mat(j,i);
			if(vec.isZero()){
				if(count!=0)
					solution_found=1;	
				else
				solution_found=2;
			}
		}
		if(solution_found<2){
			/// Solutions of the system are given by x = V*(W^{-1}*(U^T*b))
			Eigen::VectorXd x = v*inv_w*u.transpose()*init_b; 
			if(solution_found==1)
				std::cout<<"Solution found:Least-Squat-Minimum"<<std::endl;
			else
				std::cout<<"Solution found"<<std::endl;
			std::cout<<" "<<x<<std::endl;
		}else
			std::cout<<"No solution found"<<std::endl;
	};	

};

class Eigval_Eigvec
{
private:
	Eigen::MatrixXd mat;
	int m;
public:
	/// Constructor 1: Hermitian matrix
	Eigval_Eigvec(Eigen::MatrixXd r_init_mat,Eigen::MatrixXd i_init_mat){
		m=r_init_mat.rows();
		mat.resize(m*2,m*2);

		mat.block(0,0,m,m)=r_init_mat;
		mat.block(m,m,m,m)=r_init_mat;
		mat.block(0,m,m,m)=-i_init_mat;
		mat.block(m,0,m,m)=i_init_mat;

		m=m*2;
	};
	/// Constructor 2: Symmetric matrix 
	Eigval_Eigvec(Eigen::MatrixXd init_mat){
		m=init_mat.rows();
		mat.resize(m,m);
		mat.block(0,0,m,m)=init_mat;
	};
	void stadard(Eigen::MatrixXd init_mat){
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(init_mat);
		mat.setZero();
		mat.diagonal()=solver.eigenvalues();
	};
	double calc_offset(Eigen::MatrixXd mat_tmp,int n_tmp){
		double offset_tmp=0;
		for(int i=0;i<n_tmp;i++)
			for(int j=i+1;j<n_tmp-1;j++)
				offset_tmp+=std::pow(mat_tmp(i,j),2);
		return offset_tmp;
	};
	void Jacobi_Rotation(Eigen::MatrixXd init_mat,int init_n){
		Eigen::MatrixXd z(init_n,init_n);
		Eigen::MatrixXd fin_mat=init_mat;
		Eigen::MatrixXd intermediate_mat(init_n,init_n);
		double omega;
		double theta;
		double offset;
		double maximum_coeff;
		int i,j;
		double threshold=1e-10;

		// In the case of a tridiagonal matrix this routine can be speed up
		offset=calc_offset(fin_mat,init_n);

		///std::cout<<"Start of the cycle "<<offset<<" / "<<threshold<<std::endl;
		///std::cout<<std::endl;
		while(offset>threshold){
				z.setIdentity();
				intermediate_mat=fin_mat;
				intermediate_mat.diagonal().setZero();
				maximum_coeff=(intermediate_mat.cwiseAbs()).maxCoeff(&i,&j);
				omega=(fin_mat(j,j)-fin_mat(i,i))/(2*fin_mat(i,j));
				theta=std::atan(sign(omega)/(std::abs(omega)+std::sqrt(1+std::pow(omega,2))));
				z(i,i)=std::cos(theta); z(j,j)=z(i,i); 
				z(i,j)=std::sin(theta); z(j,i)=-z(i,j);
				fin_mat=z.transpose()*fin_mat*z;
				offset=calc_offset(fin_mat,init_n);
				//std::cout<<"intermediate step "<< offset<<" "<<threshold<<std::endl;
		}
		///std::cout<<std::endl;
		///std::cout<<"End of the cycle "<<offset<<" / "<<threshold<<std::endl;
		mat=fin_mat;
	};
	Eigen::MatrixXd pull_mat(){
		return mat;
	};
	int pull_dim(){
		return m;
	};
	void Housholder(Eigen::MatrixXd init_mat,int init_n){
		Eigen::MatrixXd fin_mat=init_mat;
		Eigen::VectorXd v(init_n);
		Eigen::VectorXd u(init_n);
		
		//std::cout<<"Start diagonalization procedure"<<std::endl;
		//std::cout<<fin_mat<<std::endl;
		for(int i=0;i<init_n-1;i++){
			v.block(0,0,i+1,1).setZero();
			v.block(i+1,0,init_n-i-1,1)=fin_mat.block(i+1,i,init_n-i-1,1);
			if(v.norm()==0)
				continue;
			fin_mat(i,i+1)=v.norm();
			fin_mat(i+1,i)=fin_mat(i,i+1);
			fin_mat.block(i,i+2,1,init_n-i-2).setZero();
			fin_mat.block(i+2,i,init_n-i-2,1).setZero();
			/// Applying S matrix to the rest of the initial matrix
			u=v; u(i+1)+=(v(i+1)>=0? v.norm():-v.norm());
			u=u/u.norm();
			fin_mat.block(i+1,i+1,init_n-i-1,init_n-i-1)=(4*u*(u.transpose()*(fin_mat*u))*u.transpose()-2*u*(u.transpose()*fin_mat)-2*(fin_mat*u)*u.transpose()+fin_mat).block(i+1,i+1,init_n-i-1,init_n-i-1);	
		}
		mat=fin_mat;
		///std::cout<<"End diagonalization procedure"<<std::endl;
	};

	void Potenzmethod(Eigen::MatrixXd init_mat,int init_n,int number_iter){
		Eigen::MatrixXd fin_mat=init_mat;
		Eigen::MatrixXd eigvec(init_n,init_n);
		Eigen::VectorXd eigval(init_n);
		Eigen::VectorXd init_v(init_n);
		Eigen::VectorXd fin_v(init_n);

		eigvec.setZero();
		eigval.setZero();
	
		init_v.setRandom();
		init_v=init_v/init_v.norm();

		for(int j=0;j<init_n;j++){
			/// Finding a vector orthogonal to the already determined eigenvectors
			if(j!=0){
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigvec.transpose(), Eigen::ComputeFullV);
				init_v = svd.matrixV().row(j);  // Take the next available orthogonal direction
			}
			init_v=init_v/init_v.norm();
			//std::cout<<"Initial vector: "<<init_v.transpose()<<std::endl;
			fin_v=fin_mat*init_v;
			for(int i=0;i<number_iter-1;i++)
				fin_v=fin_mat*fin_v;
			//std::cout<<"Final vector: "<<fin_v.transpose()<<std::endl;
			eigvec.row(j)=fin_v/fin_v.norm();
			eigval(j)=(fin_v*eigvec.row(j)).norm()/(init_v*eigvec.row(j)).norm();
			//std::cout<<"Eigenvector: "<<eigvec.row(j)<<std::endl;
			//std::cout<<"Eigenvalue before sqrt: "<<eigval(j)<<std::endl;
			eigval(j)=std::pow(eigval(j),1.0/number_iter);
			//std::cout<<"Eigenvalue after sqrt: "<<eigval(j)<<std::endl;
			fin_mat=fin_mat-eigval(j)*(eigvec.row(j).transpose()*eigvec.row(j));
		}
		mat.setZero();
		mat.diagonal()=eigval;
	};

	void Krylov_Arnoldi(Eigen::MatrixXd init_mat,int init_n,int number_iter,double threshold){
		Eigen::MatrixXd h(init_n,init_n);
		Eigen::MatrixXd v(number_iter,init_n);
		Eigen::VectorXd w(init_n);

		h.setZero();
		v.row(0).setRandom();
		v.row(0)=v.row(0)/v.row(0).norm();

		for(int j=0;j<number_iter;j++){
			w=init_mat*v.row(j);
			for(int i=0;i<=j;i++){
				h(i,j)=v.row(i)*w;
				w=w-h(i,j)*v.row(i);
			}
			if(w.norm()<threshold)
				break;
			else
				v.row(j+1)=w/w.norm();
		}
	};
	void Krylov_Lanczos(Eigen::MatrixXd init_mat,int init_n,int number_iter,double threshold){
		Eigen::MatrixXd T(number_iter,number_iter);
		Eigen::MatrixXd v(number_iter,init_n);
		Eigen::VectorXd w(init_n);

		T.setZero();
		T(0,1)=0;
		T(1,0)=0;
		T(0,0)=0;

		v.row(0).setZero();
		v.row(1).setRandom();
		v.row(1)=v.row(1)/v.row(1).norm();

		for(int j=1;j<number_iter;j++){
			w=init_mat*v.row(j).transpose();
			T(j,j)=v.row(j)*w;
			w=w-T(j,j)*v.row(j).transpose()-T(j-1,j)*v.row(j-1).transpose();
			T(j,j+1)=w.norm();
			T(j+1,j)=T(j,j+1);
			if(T(j+1,j)<threshold)
				break;
			v.row(j+1).transpose()=w/T(j,j+1);
		}
		mat=T;
		m=number_iter;

	};
};

///int main(int argc,char** argv){
///	
///	//int rows=2;
///	//int columns=3;
///	//
///	//Eigen::MatrixXd mat(rows,columns);
///	//Eigen::MatrixXd b(rows,1);
///	//
///	//mat.setRandom();
///	//b.setRandom();
///	//
///	//LGS lgs(mat,b);
///	//lgs.LU_decomposition_Eigen_routine(rows,columns,mat,b);
///	//lgs.SVD_decomposition_Eigen_routine(rows,columns,mat,b);
///
///	/// Considering a Symmetric matrix
///	int n=4;
///	Eigen::MatrixXd mat(n,n);
///	Eigen::MatrixXd mat_t(n,n);
///	mat.setRandom();
///	std::cout<<mat<<std::endl;
///	mat_t=mat.transpose();
///	mat=(mat+mat_t)/2;
///
///	std::cout<<"Initial Matrix"<<std::endl;
///	std::cout<<mat<<std::endl;
///
///	Eigval_Eigvec eigval_eigvec(mat);
///	/// Zero method (Working)
///	eigval_eigvec.stadard(mat);
///	std::cout<<"Final Matrix (OK)"<<std::endl;
///	std::cout<<eigval_eigvec.pull_mat().diagonal()<<std::endl;
///
///	/// First method (Working)
///	eigval_eigvec.Jacobi_Rotation(mat,n);
///	std::cout<<"Final Matrix (OK)"<<std::endl;
///	std::cout<<eigval_eigvec.pull_mat().diagonal()<<std::endl;
///
///	/// Second method (Not Working)
///	eigval_eigvec.Housholder(mat,n);
///	eigval_eigvec.Jacobi_Rotation(eigval_eigvec.pull_mat(),n);
///	std::cout<<"Final Matrix (OK)"<<std::endl;
///	std::cout<<eigval_eigvec.pull_mat().diagonal()<<std::endl;
///
///	/// Third method (Not Working)
///	int number_iter=100;
///	eigval_eigvec.Potenzmethod(mat,n,number_iter);
///	std::cout<<"Final Matrix"<<std::endl;
///	std::cout<<eigval_eigvec.pull_mat().diagonal()<<std::endl;
///
///	/// Fourth method (Working)
///	double threshold=1e-10;
///	eigval_eigvec.Krylov_Lanczos(mat,n,number_iter,threshold);
///	eigval_eigvec.Jacobi_Rotation(eigval_eigvec.pull_mat(),eigval_eigvec.pull_dim());
///	std::cout<<"Final Matrix (OK)"<<std::endl;
///	std::cout<<eigval_eigvec.pull_mat().block(1,1,n,n).diagonal()<<std::endl;
///
///
///
///	return 1;
///};

