// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(FLasher)]]
#include <FLasher.h>
#include <Rcpp.h>
#include <random>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
Eigen::MatrixXd MH_update(NumericVector x0, int n) {
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0, 1);
  Eigen::MatrixXd d_updated(50000, 2);
  NumericVector x;
  d_updated.Zero(50000,2);
  // the 1st column of "d_updated" : the index of accpeted d
  // the 2nd column of "d_updated" : the frequency of accpeted d
  d_updated.row(0) << 1, 1;
  double baseline, acceptance_ratio;
  int nr=0;
  x= exp(x0);
  baseline = x[0];
  for(int i = 1; i < n; ++i) {
    acceptance_ratio = x[i]/baseline;
    if (dis(gen) > std::min(acceptance_ratio, 1.0)){  // accept d
      d_updated(nr, 1) = d_updated(nr, 1) + 1;
    } else {                                          // reject d
      nr = nr + 1;
      d_updated.row(nr)<< i+1, 1;
      baseline = x[i];
    }
  }
  return d_updated.block(0, 0, nr + 1, 2);
}
