MH_algorithm <- function(x, y, z, theta0, phi0, delta.star){
  theta.hess <- theta.grad <- phi.hess <- phi.grad <- 0
  ## used in louis formula
  louis.aj1 <- 0

  ## find the number/index of non-NA elements in each row
  dm <- dim(y)
  na.mat <- matrix(rep(0:(dm[2] - 1), dm[1]), nrow = dm[1], ncol = dm[2], byrow = T)
  na.mat[which(is.na(y))] <- NA
  ind <- cbind(apply(!is.na(y), 1, sum), t(apply(na.mat, 1, function(x) sort(x, na.last=T))))

  ## MH algorithm : calculate f_{y|d}
  f_yd <- MH_cal(theta0 = theta0, x0 = x, y0 = y,z0 = z, alpha0 = phi0[-length(phi0)], err0 = delta.star, ind = ind)

  ## MH algorithm : update error terms by f_{y|d}
  d_updated <- MH_update(f_yd, n = MC.size)

  ## Monte Carlo approximation for conditional expectation of hessian/gradient of theta/phi
  err.unique <- as.matrix(delta.star[d_updated[,1], ])
  err.freq <-  as.vector(d_updated[,2])
  for (v in 1:length(err.freq)) {
    theta.grad.aj <- 0
    d <- (z %*% phi0[-length(phi0)]) + err.unique[v, ]
    output2 <- phi_deriv(phi0 = phi0, z = z, alphaz = d)
    phi.grad <- phi.grad + err.freq[v] * output2[[1]]
    phi.hess <- phi.hess + err.freq[v] * output2[[2]]
    for (jj in 1:nrow(y)) {
      ## find the index of non-NA elements in y
      complete.pos <- which(!is.na(y[jj, ]))
      complete.len <- length(complete.pos)
      if (complete.len == 0)
        next
      output1 <- theta_deriv(theta0 = theta0, x0 = x, alphaz0 = d[jj], y0 = y[jj, ], complete_pos = complete.pos)
      theta.grad <- theta.grad + err.freq[v] * output1[[1]]
      theta.hess <- theta.hess + err.freq[v] * output1[[2]]
      theta.grad.aj <- theta.grad.aj + output1[[1]]
    }
    louis.aj1 <- louis.aj1 + (err.freq[v]/MC.size) * (c(theta.grad.aj, output2[[1]]) %*% t(c(theta.grad.aj, output2[[1]])))
  }

  output <- list()
  output[["theta.hess"]] <- theta.hess
  output[["theta.grad"]] <- theta.grad
  output[["phi.hess"]] <- phi.hess
  output[["phi.grad"]] <- phi.grad
  output[["louis.aj1"]] <- louis.aj1
  return(output)
}
