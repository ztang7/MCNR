MCNR <- function(x, y, z, MC.size, theta0, phi0, fixed) {
  theta.hess <- theta.grad <- phi.hess <- phi.grad <- 0
  louis.aj1 <- louis.aj2 <- 0
  ## calculate Monte Carlo conditional expectacion of hessian/gradient for each subject
  for (i in 1:N) {
    ## generate sample delta
    cov.delta <- AR1(j[i], rho = phi0[length(phi0)])
    delta.star <- mvrnorm(n = MC.size, mu = rep(0, j[i]), cov.delta)
    ## implement MH algorithm
    output <- MH_algorithm(x = x[[i]], y = y[[i]], z = z[[i]], theta0, phi0, delta.star)
    theta.hess <- theta.hess + output$theta.hess
    phi.hess <- phi.hess + output$phi.hess
    theta.grad <- theta.grad + output$theta.grad
    phi.grad <- phi.grad + output$phi.grad
    
    ## used in louis formula
    para <- c(output$theta.grad, output$phi.grad)/MC.size
    louis.aj2 <- louis.aj2 + (para  %*% t(para))
    louis.aj1 <- louis.aj1 + output$louis.aj1
  }
  theta.hess <- matrix(theta.hess, nrow = length(theta0), ncol = length(theta0))
  phi.hess <- matrix(phi.hess, nrow = length(phi0), ncol = length(phi0))
  
  ## Newton-Raphson method
  theta0[-fixed] <- theta0[-fixed] - solve(theta.hess[-fixed, -fixed]) %*% theta.grad[-fixed]
  phi0 <- phi0 - as.vector(solve(phi.hess) %*% phi.grad)
  
  ## save results
  results <- list()
  results[["theta0"]] <- theta0
  results[["phi0"]] <- phi0
  results[["theta.hess"]] <- theta.hess
  results[["theta.grad"]] <- theta.grad
  results[["phi.hess"]] <- phi.hess
  results[["phi.grad"]] <- phi.grad
  results[["louis.aj2"]] <- louis.aj2
  results[["louis.aj1"]] <- louis.aj1
  return(results)
}




