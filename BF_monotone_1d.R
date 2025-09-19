# 需要：lineqGPR
library(lineqGPR)

AGP_monotone_SE <- function(x, y, xt, y_true,
                            m = 5,
                            nsim = 1000,
                            maxeval = 200,
                            seed = 7) {
  set.seed(seed)
  x  <- as.numeric(x)
  y  <- as.numeric(y)
  xt <- as.numeric(xt)
  y_true <- as.numeric(y_true)
  stopifnot(length(xt) == length(y_true))
  
  # --- 0) 将 x, xt 统一缩放到 [0,1]（避免 Phi 计算报 NA） ---
  a <- min(c(x, xt))
  b <- max(c(x, xt))
  if (!is.finite(a) || !is.finite(b) || b <= a) {
    stop("Invalid range for scaling to [0,1]. Check x / xt.")
  }
  scale01 <- function(v) (v - a) / (b - a)
  X_train <- matrix(scale01(x), ncol = 1)
  X_test  <- matrix(scale01(xt), ncol = 1)
  
  # --- 1) 建模（单调性约束） ---
  model <- create(class = "lineqAGP",
                  x = X_train,
                  y = y,
                  constrType = "monotonicity",
                  m = m)
  model$kernParam[[1]]$par <- c(1, 2)   # 初值
  model$nugget   <- 1e-3
  # 给一个合理的初始噪声方差；优化器会再估计
  model$varnoise <- max(var(y), 1e-6) * 0.1
  
  t0 <- proc.time()
  model <- lineqGPOptim(
    model,
    x0   = unlist(lapply(model$kernParam, `[[`, "par")),
    eval_f = "logLik",
    additive = TRUE,
    opts = list(algorithm = "NLOPT_LD_MMA",
                print_level = 0,
                ftol_abs = 1e-3,
                maxeval = maxeval,
                check_derivatives = FALSE),
    lb = rep(1e-2, 2),
    ub = c(5, 3),
    estim.varnoise = TRUE,
    bounds.varnoise = c(1e-6, Inf)
  )
  
  # --- 2) 后验抽样（函数层面），得到均值与95%置信带 ---
  sim_obj  <- simulate(model, nsim = nsim, xtest = X_test)
  Phi_test <- sim_obj$Phi.test[[1]]          # ntest x m
  m1 <- model$localParam$m
  Ysim <- Phi_test %*% sim_obj$xiAll.sim[1:m1, , drop = FALSE]  # ntest x nsim
  
  post_mean <- rowMeans(Ysim)
  post_var  <- apply(Ysim, 1, var)
  post_sd   <- sqrt(pmax(post_var, 1e-12))   # 数值稳定
  post_lo   <- apply(Ysim, 1, quantile, probs = 0.025)
  post_hi   <- apply(Ysim, 1, quantile, probs = 0.975)
  
  # --- 3) 指标 ---
  mse <- mean((post_mean - y_true)^2)
  
  # NLPD（函数层面；不含观测噪声）
  nlpd <- -mean(dnorm(y_true, mean = post_mean, sd = post_sd, log = TRUE))
  
  coverage <- mean(y_true >= post_lo & y_true <= post_hi)
  width    <- mean(post_hi - post_lo)
  
  cost_time <- (proc.time() - t0)[["elapsed"]]
  
  
  list(
    mean = post_mean,
    lo = post_lo,
    hi = post_hi,
    mse = mse,
    nlpd = nlpd,
    coverage = coverage,
    width = width,
    time = as.numeric(cost_time)
  )
}

test_df  <- read.csv("datasets_stepwise_csv/test_grid.csv")
xt     <- as.numeric(test_df$xt)
y_true <- as.numeric(test_df$y_true)
tr <- read.csv("datasets_stepwise_csv/train_run01.csv")
x <- as.numeric(tr$x); y <- as.numeric(tr$y)

res   <- AGP_monotone_SE(x, y, xt, y_true, m=5, nsim=1000, maxeval=200, seed=7)
plot(xt, y_true, type = "l", lwd = 3, col = col_truth,
     xlab = "x", ylab = "f(x)",
     main = "True f(x) vs Posterior mean & 95% band")
polygon(c(xt, rev(xt)),
        c(res$lo,     rev(res$hi)),
        border = NA, col = col_band)
lines(xt, res$mean, lwd = 2, col = col_mean)
out <- data.frame(
  x   = xt,
  y_true = y_true,
  lo  = res$lo,
  hi  = res$hi,
  mean = res$mean
)
write.csv(out, "bf_stepwise.csv", row.names = FALSE)
