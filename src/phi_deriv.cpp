// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(FLasher)]]
#include <FLasher.h>
#include <cppad/cppad.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::export]]
List phi_deriv(std::vector<double> phi0, NumericMatrix z0, NumericVector alphaz0) {
  using CppAD::AD;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  typedef Matrix< AD<double> , Dynamic, Dynamic > a_matrix;
  typedef Matrix< AD<double> , Dynamic , 1>       a_vector;
  
  // Parameter Initialization
  size_t i, j, j_time = alphaz0.size(), phi_len = phi0.size();
  a_vector alphaz(j_time),  likelihood(1), phi(phi_len);
  a_matrix sigma_mat(j_time, j_time), z(j_time, phi_len - 1);
  for(i = 0; i < j_time; i++)
    alphaz[i] = alphaz0[i];
  for(i = 0; i < (phi_len - 1); i++) {
    for (j = 0; j < j_time; j++)
      z(j, i) = z0(j, i);
  }
  Map<a_matrix> alpha(phi.block(0, 0, phi_len - 1, 1).data(), phi_len - 1, 1);
  CppAD::Independent(phi);
  sigma_mat = a_matrix::Zero(j_time , j_time);
  for(i = 0; i < j_time; i++) {
    sigma_mat.diagonal( i).array() = pow(phi[phi_len - 1], i);
    sigma_mat.diagonal(-i).array() = pow(phi[phi_len - 1], i);
  }
  alphaz -= z*alpha;
  
  // compute gradient/hessian of log likelihood
  likelihood[0] =-0.5*log(sigma_mat.determinant())+(-0.5*alphaz.transpose()*sigma_mat.inverse()*alphaz)(0);
  CppAD::ADFun<double> f(phi, likelihood);
  std::vector<double> jac = f.Jacobian(phi0);
  std::vector<double> hes = f.Hessian(phi0, 0);
  List L = List::create(jac, hes);
  return L;
}