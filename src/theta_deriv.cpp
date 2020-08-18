// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(FLasher)]]
#include <FLasher.h>
#include <cppad/cppad.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::export]]
List theta_deriv(std::vector<double> theta0, NumericVector x0, NumericVector alphaz0, NumericVector y0, std::vector<int> complete_pos) {
  using CppAD::AD;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  typedef Matrix< AD<double> , Dynamic, Dynamic > a_matrix;
  typedef Matrix< AD<double> , Dynamic , 1>       a_vector;
  
  // Parameter Initialization
  size_t i, j, k = y0.size(), p = x0.size(), theta_len = k*p + 3*k + k + k*(k-1)/2, count = 0;
  a_vector x(p), y(k),  likelihood(1), theta(theta_len), alphaz(1);
  a_matrix sigma_mat(k, k);
  alphaz[0] = alphaz0[0];
  for(i = 0; i < p; i++)
    x[i] = x0[i];
  for(i = 0; i < k; i++)
    y[i] = y0[i];
  for(i = 0; i < theta_len; i++)
    theta[i] = theta0[i];
  
  Map<a_matrix> beta(theta.block(0, 0, k * p, 1).data(), k, p);
  Map<a_matrix> gamma(theta.block(k * p, 0, 3 * k, 1).data(), k, 3);
  
  CppAD::Independent(theta);
  sigma_mat = a_matrix::Ones(k , k);
  for(i = 0; i < k; i++) {
    sigma_mat(i, i) = pow(theta[k*p + 3*k + i], 2);
  }
  for(i = 0; i < k; i++) {
    for (j = i + 1; j < k; j++) {
      sigma_mat(i, j) = theta[k*p + 3*k + k + count];
      sigma_mat(j, i) = theta[k*p + 3*k + k + count];
      count +=1;
    }
  }
  alphaz[0] = exp(alphaz[0])/(exp(alphaz[0])+1);
  
  for(i = 0; i < k; i++)
    y[i] = y[i] - gamma(i, 0)/(1 + exp(-gamma(i, 1)*(alphaz[0] - gamma(i, 2))));
  y -= beta * x;
  
  // remove NA rows in y and get completed data (y_c and sigma_c)
  a_vector y_c(complete_pos.size());
  a_matrix sigma_c(complete_pos.size(), complete_pos.size());
  for(i = 0; i < complete_pos.size(); i++)
    y_c[i] = y[complete_pos[i]-1];
  
  for(i = 0; i < complete_pos.size(); i++) {
    for (j = 0; j < complete_pos.size(); j++) {
      sigma_c(i, j) = sigma_mat(complete_pos[i]-1, complete_pos[j]-1);
    }
  }
  
  // compute gradient/hessian of log likelihood
  likelihood[0] =-0.5*log(sigma_c.determinant())+(-0.5*y_c.transpose()*sigma_c.inverse()*y_c)(0);
  CppAD::ADFun<double> f(theta, likelihood);
  std::vector<double> jac = f.Jacobian(theta0);
  std::vector<double> hes = f.Hessian(theta0, 0);
  List L = List::create(jac, hes);
  return L;
}
