// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(FLasher)]]
#include <FLasher.h>
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;
using namespace std;
double inv_logit(double x) 
{
  return 1/(1+std::exp(-x));
}

int sumOfAP(int n) 
{ 
  int sum = 0; 
  for (int i=0; i < n; i++) 
  { 
    sum = sum + i; 
  } 
  return sum; 
} 

// [[Rcpp::export]]
Eigen::MatrixXd MH_cal(NumericVector theta0, NumericVector x0, NumericMatrix y0, NumericMatrix z0, NumericVector alpha0, NumericMatrix err0, NumericMatrix ind) {
  size_t k = y0.ncol(), p = x0.size(), nr = err0.nrow(), j_time = err0.ncol(), i, j, count = 0;
  Eigen::MatrixXd alphaz, betax, sigma_mat, mu, output, sigma_m, mu_m;
  
  // Parameter Initialization
  Map<Eigen::VectorXd> theta(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(theta0));
  Map<Eigen::MatrixXd> beta(theta.block(0, 0, k * p, 1).data(), k, p);
  Map<Eigen::MatrixXd> gamma(theta.block(k*p, 0, 3 * k, 1).data(), k, 3);
  Map<Eigen::VectorXd> alpha(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(alpha0));
  Map<Eigen::MatrixXd> err(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(err0));
  Map<Eigen::VectorXd> x1(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(x0));
  Map<Eigen::MatrixXd> z1(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(z0));
  Map<Eigen::MatrixXd> y1(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(y0));
  mu = Eigen::MatrixXd::Zero(nr, k);
  output = Eigen::MatrixXd::Zero(nr, 1);

  sigma_mat = Eigen::MatrixXd::Ones(k, k);
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
  
  betax = (beta*x1).transpose().replicate(nr, 1);
  alphaz = (z1*alpha).transpose().replicate(nr,1) + err;
  alphaz = alphaz.unaryExpr(&inv_logit);
  
  // calcuate f_{y|d} for each d (generated error terms)
  for (j = 0; j < j_time; j++){
    // get completed covariance matrix(Sigma_eplison)
    sigma_m = Eigen::MatrixXd::Zero(ind(j, 0), ind(j, 0));
    for(i = 0; i < ind(j, 0); i++) {
      for (size_t tm = 0; tm < ind(j, 0); tm++) {
        sigma_m(i, tm) = sigma_mat(ind(j, i+1), ind(j, tm+1));
      }
    }
    
    // get completed (y-mu)'
    for (size_t kk = 0; kk < k; kk++){
      for (size_t n = 0; n < nr; n++){
        mu(n, kk)=gamma(kk, 0)/(1 + std::exp(-gamma(kk, 1)*(alphaz(n, j) - gamma(kk, 2))));
      }
    }
    mu = y1.row(j).replicate(nr, 1) - betax - mu;
    mu_m = Eigen::MatrixXd::Zero(nr, ind(j, 0));
    for (i = 0; i < ind(j, 0); i++) 
      mu_m.col(i) = mu .col(ind(j, i+1));
    
    output = output + (mu_m * sigma_m.inverse()).cwiseProduct(mu_m).rowwise().sum() ;
  }
  
  return -0.5*output;
}
