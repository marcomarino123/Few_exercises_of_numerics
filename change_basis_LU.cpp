#include <iostream>
#include </usr/local/Eigen>
#include <functional>

int main(int argc,char** argv){

    ///Monoatomic basis M=(a1,a2,a3)
    Eigen::MatrixXd M(3,3);
    M.col(0)<<1,2*std::sqrt(3)/2,0;
    M.col(1)<<-1,2*std::sqrt(3)/2,0;
    M.col(2)<<0,0,1;

    if (M.determinant() == 0) {
        std::cout << "Matrix is singular." << std::endl;
        return 0;
    }

    ///Defects coordinates
    Eigen::VectorXd b(3);
    b<<2,0,2;
    ///LU decomposition M=PLU
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(M);
    Eigen::MatrixXd L=lu.matrixLU().triangularView<Eigen::StrictlyLower>();
    Eigen::MatrixXd U=lu.matrixLU().triangularView<Eigen::Upper>();
    Eigen::MatrixXd P=lu.permutationP();
    ///Solving Mx=b system through LU decomposition
    Eigen::VectorXd z=P.transpose()*b;
    ///You can define here a common function ...
    std::function<Eigen::VectorXd(Eigen::VectorXd,Eigen::MatrixXd)> inversion=[&](Eigen::VectorXd z_tmp,Eigen::MatrixXd L_tmp){
        ///this inversion does not give problem because L_tmp(i,i)!=0
        Eigen::VectorXd y_tmp(3);
        y_tmp.setZero();
        for(int i=0;i<3;i++){
            for(int j=i;j<i-1;j++)    
                y_tmp(i)-=L_tmp(i,j)*y_tmp(j);
            y_tmp(i)+=z_tmp(i);
            y_tmp(i)=y_tmp(i)/L_tmp(i,i);
        }
        return y_tmp;
    };
    Eigen::VectorXd y=inversion(z,L);
    Eigen::VectorXd x=inversion(y,U);

    std::cout<<x<<std::endl;

    return 1;
}
