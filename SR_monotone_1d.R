suppressPackageStartupMessages(library(bsamGP))
suppressPackageStartupMessages(library(viridis))

bsar_monotone <- function(x, y, xt, y_true,
                          nbasis = 25,
                          shape  = "Increasing",
                          alpha  = 0.05,
                          seed   = 1) {
  set.seed(seed)
  t0  <- proc.time()[3]
  
  fit <- bsar(y ~ fs(x), nbasis = nbasis, shape = shape, spm.adequacy = TRUE)
  
  pred <- predict(fit, newnp = data.frame(x = xt), alpha = alpha, HPD = TRUE, type = "mean")
  
  mu <- as.numeric(pred$yhat$mean)
  lo <- as.numeric(pred$yhat$lower)
  hi <- as.numeric(pred$yhat$upper)
  
  mse      <- mean((mu - y_true)^2)
  coverage <- mean(y_true >= lo & y_true <= hi)
  width    <- mean(hi - lo)
  
  z <- qnorm(1 - alpha/2)
  sd_est <- pmax((hi - lo) / (2*z), 1e-12)
  nlpd   <- -mean(dnorm(y_true, mean = mu, sd = sd_est, log = TRUE))
  
  time_elapsed <- proc.time()[3] - t0
  
  list(mean = mu, lo = lo, hi = hi,
       mse = mse, nlpd = nlpd, coverage = coverage, width = width, time = as.numeric(time_elapsed),
       model = fit)
}
