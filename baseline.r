## =========================
## 1) Unconstrained GP (SE)
## =========================
## install.packages("DiceKriging")
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
       mean=Ef, Varf=Varf, lo=ci_lower, hi=ci_upper,
       sigma2=sigma2, model=fit)
}




## =========================
## 2) Isotonic Regression (PAV)
## =========================
## base R only
monotone_isotonic <- function(
    x, y, xt, y_true,
    increasing = TRUE,     # TRUE=递增，FALSE=递减
    bandwidth = NULL,      # 近似 SE 的核带宽；NULL=自动
    min_neff = 5,          # 每个测试点的最小“有效样本数”
    seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)
  t0 <- proc.time()[3]
  
  x  <- as.numeric(x);  y  <- as.numeric(y)
  xt <- as.numeric(xt); y_true <- as.numeric(y_true)
  stopifnot(length(x) == length(y), length(xt) == length(y_true))
  
  # 递减 => 取反变递增拟合
  y_fit <- if (increasing) y else -y
  
  ord <- order(x)
  iso <- isoreg(x[ord], y_fit[ord])
  
  # 右连续阶梯函数预测器；两端平直延伸
  f_step <- approxfun(x[ord], iso$yf, method = "constant", f = 1, rule = 2)
  
  # 训练点的函数估计，用于估计残差尺度（仅用于 SE 近似，不返回观测层面量）
  Ef_train <- f_step(x) * if (increasing) 1 else -1
  sigma_hat <- sqrt(max(mean((y - Ef_train)^2), 1e-12))
  
  # 测试点的函数预测（确定性）
  Ef <- f_step(xt)
  if (!increasing) Ef <- -Ef
  
  # 带宽自动化（稳定范围）
  if (is.null(bandwidth)) {
    rng <- diff(range(x))
    bw0 <- stats::bw.nrd0(x)
    bandwidth <- max(min(bw0, 0.25 * rng), 0.02 * rng)
  }
  h <- bandwidth
  
  # 局部加权近似标准误：se_i ≈ sigma_hat / sqrt(n_eff,i)
  se <- vapply(xt, function(x0) {
    w <- exp(-0.5 * ((x - x0) / h)^2)
    neff <- sum(w)
    if (!is.finite(neff) || neff < min_neff) neff <- min_neff
    sigma_hat / sqrt(neff)
  }, numeric(1))
  
  Varf <- pmax(se^2, 1e-12)  # 函数层面方差
  z <- 1.96
  ci_lower <- Ef - z * sqrt(Varf)
  ci_upper <- Ef + z * sqrt(Varf)
  
  # 指标（函数层面口径）
  mse      <- mean((Ef - y_true)^2)
  nlpd     <- -mean(dnorm(y_true, mean = Ef, sd = sqrt(Varf), log = TRUE))
  coverage <- mean(y_true >= ci_lower & y_true <= ci_upper)
  width    <- mean(ci_upper - ci_lower)
  
  time <- proc.time()[3] - t0
  
  list(
    mean = Ef,
    lo = ci_lower,
    hi = ci_upper,
    mse = mse, nlpd = nlpd, coverage = coverage, width = width, time = time,
    Ef = Ef, Varf = Varf, ci_lower = ci_lower, ci_upper = ci_upper,
    # 兼容字段：predVar 即函数层面方差
    predVar = Varf,
    bandwidth = h
  )
}



## =========================================
## 3) Monotone Splines (SCAM, monotone incr)
## =========================================
## install.packages("scam")
## Monotone Splines (SCAM, monotone incr/dec)
monotone_spline_scam <- function(x, y, xt, y_true,
                                 monotone = c("increasing","decreasing"),
                                 k = 40, seed = NULL) {
  # 必须 attach 以便公式中能找到 s()
  suppressPackageStartupMessages({
    if (!require(mgcv)) stop("Please install.packages('mgcv')")
    if (!require(scam)) stop("Please install.packages('scam')")
  })
  if (!is.null(seed)) set.seed(seed)
  t0 <- proc.time()[3]
  
  x  <- as.numeric(x); y <- as.numeric(y)
  xt <- as.numeric(xt); y_true <- as.numeric(y_true)
  stopifnot(length(x) == length(y), length(xt) == length(y_true))
  
  mono  <- match.arg(monotone)
  bsTag <- if (mono == "increasing") "mpi" else "mpd"   # scam 的单调基（增/减）
  
  dat <- data.frame(x = x, y = y)
  
  # 用字符串构造公式，避免 s() 被提前求值
  form <- as.formula(sprintf("y ~ s(x, bs = '%s', k = %d)", bsTag, k))
  
  fit <- scam::scam(form, data = dat, family = gaussian())
  
  # 预测
  pr <- predict(fit, newdata = data.frame(x = xt),
                se.fit = TRUE, type = "response")
  
  Ef   <- as.numeric(pr$fit)
  Varf <- pmax(as.numeric(pr$se.fit^2), 1e-12)   # 防 0
  
  z <- 1.96
  ci_lower <- Ef - z * sqrt(Varf)
  ci_upper <- Ef + z * sqrt(Varf)
  
  mse      <- mean((Ef - y_true)^2)
  nlpd     <- -mean(dnorm(y_true, mean = Ef, sd = sqrt(Varf), log = TRUE))  # 改：函数层面
  coverage <- mean(y_true >= ci_lower & y_true <= ci_upper)
  width    <- mean(ci_upper - ci_lower)
  
  time <- proc.time()[3] - t0
  list(mean = Ef, lo = ci_lower, hi = ci_upper, mse = mse, nlpd = nlpd, coverage = coverage, width = width, time = time,
       Ef = Ef, Varf = Varf, ci_lower = ci_lower, ci_upper = ci_upper, model = fit)
}




## =========================================
## 4) Monotone BART (mBART)
## =========================================
monotone_bart <- function(x, y, xt, y_true,
                             ntree = 100, ndpost = 1000, nskip = 200,
                             mgsize = 50, probs = c(0.025, 0.975),
                             printevery = 100) {
  
  t0 <- proc.time()[3]
  
  fit <- monbart(
    x.train = matrix(x, ncol = 1),
    y.train = as.numeric(y),
    x.test  = matrix(xt, ncol = 1),
    ntree   = ntree,
    ndpost  = ndpost,
    nskip   = nskip,
    mgsize  = mgsize,
    probs   = probs,
    printevery = printevery
  )
  
  time_elapsed <- proc.time()[3] - t0
  
  Ef   <- fit$yhat.test.mean
  draws <- fit$yhat.test                # ndpost × nstar
  Varf  <- apply(draws, 2, var)
  lo    <- fit$yhat.test.lower
  hi    <- fit$yhat.test.upper
  
  mse      <- mean((Ef - y_true)^2)
  coverage <- mean(y_true >= lo & y_true <= hi)
  width    <- mean(hi - lo)
  
  # 改：函数层面正态近似的 NLPD
  nlpd <- -mean(dnorm(y_true, mean = Ef, sd = sqrt(pmax(Varf, 1e-12)), log = TRUE))
  
  
  list(
    mean = Ef, lo = lo, hi = hi,
    mse = mse, nlpd = nlpd, coverage = coverage, width = width, time = time_elapsed,
    Ef = Ef, Varf = Varf, ci_lower = lo, ci_upper = hi, fit = fit
  )
}

test_df  <- read.csv("datasets_sine_csv/test_grid.csv")
xt     <- as.numeric(test_df$xt)
y_true <- as.numeric(test_df$y_true)
tr <- read.csv("datasets_sine_csv/train_run19.csv")
x <- as.numeric(tr$x); y <- as.numeric(tr$y)

res    <- gp_unconstrained(x, y, xt, y_true, nugget.estim=TRUE)
#res   <- monotone_isotonic(x, y, xt, y_true)
#res  <- monotone_spline_scam(x, y, xt, y_true, monotone="increasing", k=40)
#res <- monotone_bart(x, y, xt, y_true, ntree=50, ndpost=200, nskip=100, mgsize=20)
plot(xt, y_true, type = "l", lwd = 3, col = col_truth,
     xlab = "x", ylab = "f(x)",
     main = "True f(x) vs Posterior mean & 95% band")
polygon(c(xt, rev(xt)),
        c(res$lo,     rev(res$hi)),
        border = NA, col = col_band)
lines(xt, res$mean, lwd = 2, col = col_mean)
res$mse