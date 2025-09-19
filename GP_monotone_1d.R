gp_unconstrained <- function(x, y, xt, y_true,
                             known_noise_var = NULL,  # 改：默认 NULL 表示未知，进入估计
                             nugget.estim = TRUE,
                             var_floor = 1e-6,
                             seed = NULL) {
  if (!requireNamespace("DiceKriging", quietly = TRUE))
    stop("Please install.packages('DiceKriging')")
  if (!is.null(seed)) set.seed(seed)
  t0 <- proc.time()[3]
  x <- as.numeric(x); y <- as.numeric(y)
  xt <- as.numeric(xt); y_true <- as.numeric(y_true)
  
  km_ctrl <- list(trace = FALSE)
  if (is.null(known_noise_var)) {
    fit <- DiceKriging::km(
      formula = ~1, design = data.frame(x = x), response = y,
      covtype = "gauss", control = km_ctrl, nugget.estim = nugget.estim
    )
    sigma2 <- tryCatch(as.numeric(fit@covariance@nugget), error = function(e) NA_real_)
    if (!is.finite(sigma2) || sigma2 < 0) {
      mu_tr <- DiceKriging::predict.km(fit, newdata = data.frame(x=x),
                                       type="UK", se.compute=FALSE)$mean
      sigma2 <- mean((y - mu_tr)^2)
    }
  } else {
    fit <- DiceKriging::km(
      formula = ~1, design = data.frame(x = x), response = y,
      covtype = "gauss", control = km_ctrl, nugget.estim = FALSE,
      noise.var = rep(known_noise_var, length(y))
    )
    sigma2 <- known_noise_var
  }
  sigma2 <- max(sigma2, var_floor)
  
  pr   <- DiceKriging::predict.km(fit, newdata = data.frame(x=xt),
                                  type="UK", se.compute=TRUE, checkNames=FALSE)
  Ef   <- as.numeric(pr$mean)
  Varf <- pmax(as.numeric(pr$sd^2), var_floor)
  
  z <- 1.96
  ci_lower <- Ef - z*sqrt(Varf)   # 函数层面的 95% 带
  ci_upper <- Ef + z*sqrt(Varf)
  
  mse      <- mean((Ef - y_true)^2)
  # 改：函数层面的 NLPD（与前面一致）
  nlpd     <- -mean(dnorm(y_true, mean=Ef, sd=sqrt(Varf), log=TRUE))
  coverage <- mean(y_true >= ci_lower & y_true <= ci_upper)
  width    <- mean(ci_upper - ci_lower)
  
  time <- proc.time()[3] - t0
  list(mse=mse, nlpd=nlpd, coverage=coverage, width=width, time=time,
       Ef=Ef, Varf=Varf, ci_lower=ci_lower, ci_upper=ci_upper,
       sigma2=sigma2, model=fit)
}