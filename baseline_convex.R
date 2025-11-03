## =========================
## 1) Unconstrained GP (SE)
## =========================
## install.packages("DiceKriging")
gp_unconstrained <- function(x, y, xt, y_true,
                             known_noise_var = NULL,
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
  ci_lower <- Ef - z*sqrt(Varf) 
  ci_upper <- Ef + z*sqrt(Varf)
  
  mse      <- mean((Ef - y_true)^2)
  nlpd     <- -mean(dnorm(y_true, mean=Ef, sd=sqrt(Varf), log=TRUE))
  coverage <- mean(y_true >= ci_lower & y_true <= ci_upper)
  width    <- mean(ci_upper - ci_lower)
  
  time <- proc.time()[3] - t0
  list(mse=mse, nlpd=nlpd, coverage=coverage, width=width, time=time,
       mean=Ef, Varf=Varf, lo=ci_lower, hi=ci_upper,
       sigma2=sigma2, model=fit)
}



## =========================================
## 2) Convex Splines (SCAM, convex)
## =========================================
## Convex/Concave Splines via SCAM
## install.packages(c("mgcv","scam"))

convex_spline_scam <- function(x, y, xt, y_true,
                                 shape = c("convex","concave"),
                                 k = 40, m = 2, seed = NULL) {
  suppressPackageStartupMessages({
    if (!require(mgcv)) stop("Please install.packages('mgcv')")
    if (!require(scam)) stop("Please install.packages('scam')")
  })
  if (!is.null(seed)) set.seed(seed)
  t0 <- proc.time()[3]
  
  x  <- as.numeric(x); y <- as.numeric(y)
  xt <- as.numeric(xt); y_true <- as.numeric(y_true)
  stopifnot(length(x) == length(y), length(xt) == length(y_true))
  
  shp   <- match.arg(shape)
  bsTag <- if (shp == "convex") "cx" else "cv"   # convex/concave basis in SCAM
  
  dat  <- data.frame(x = x, y = y)
  form <- as.formula(sprintf("y ~ s(x, bs = '%s', k = %d, m = %d)", bsTag, k, m))
  
  fit <- scam::scam(form, data = dat, family = gaussian())
  
  pr <- predict(fit, newdata = data.frame(x = xt),
                se.fit = TRUE, type = "response")
  
  Ef   <- as.numeric(pr$fit)
  Varf <- pmax(as.numeric(pr$se.fit^2), 1e-12)
  
  z <- 1.96
  ci_lower <- Ef - z * sqrt(Varf)
  ci_upper <- Ef + z * sqrt(Varf)
  
  mse      <- mean((Ef - y_true)^2)
  nlpd     <- -mean(dnorm(y_true, mean = Ef, sd = sqrt(Varf), log = TRUE))
  coverage <- mean(y_true >= ci_lower & y_true <= ci_upper)
  width    <- mean(ci_upper - ci_lower)
  
  time <- proc.time()[3] - t0
  list(mean = Ef, lo = ci_lower, hi = ci_upper,
       mse = mse, nlpd = nlpd, coverage = coverage, width = width, time = time,
       Ef = Ef, Varf = Varf, ci_lower = ci_lower, ci_upper = ci_upper, model = fit)
}


