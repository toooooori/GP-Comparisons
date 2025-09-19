suppressPackageStartupMessages(library(bsamGP))
suppressPackageStartupMessages(library(viridis))

# —— bsamGP 单调模型：训练→在 xt 上预测→返回指标和曲线 —— 
bsar_monotone <- function(x, y, xt, y_true,
                          nbasis = 25,
                          shape  = "Increasing",
                          alpha  = 0.05,
                          seed   = 1) {
  set.seed(seed)
  t0  <- proc.time()[3]
  
  # 拟合
  fit <- bsar(y ~ fs(x), nbasis = nbasis, shape = shape, spm.adequacy = TRUE)
  
  # 预测（注意 newnp 要用 data.frame，并把 alpha 传进去）
  pred <- predict(fit, newnp = data.frame(x = xt), alpha = alpha, HPD = TRUE, type = "mean")
  
  # 取均值与区间（不同版本字段名可能稍有差异，这里按常见的 yhat$mean/lower/upper）
  mu <- as.numeric(pred$yhat$mean)
  lo <- as.numeric(pred$yhat$lower)
  hi <- as.numeric(pred$yhat$upper)
  
  # 指标（函数层面）
  mse      <- mean((mu - y_true)^2)
  coverage <- mean(y_true >= lo & y_true <= hi)
  width    <- mean(hi - lo)
  
  # 用区间反推 sd（正态近似）做 NLPD
  z <- qnorm(1 - alpha/2)
  sd_est <- pmax((hi - lo) / (2*z), 1e-12)
  nlpd   <- -mean(dnorm(y_true, mean = mu, sd = sd_est, log = TRUE))
  
  time_elapsed <- proc.time()[3] - t0
  
  # 返回 list，便于用 $ 访问
  list(mean = mu, lo = lo, hi = hi,
       mse = mse, nlpd = nlpd, coverage = coverage, width = width, time = as.numeric(time_elapsed),
       model = fit)
}

test_df  <- read.csv("datasets_stepwise_csv/test_grid.csv")
xt     <- as.numeric(test_df$xt)
y_true <- as.numeric(test_df$y_true)
tr <- read.csv("datasets_stepwise_csv/train_run01.csv")
x <- as.numeric(tr$x); y <- as.numeric(tr$y)

res  <- bsar_monotone(x, y, xt, y_true, nbasis=50, alpha=0.05, seed=2025)
res$mse
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
write.csv(out, "sr_stepwise.csv", row.names = FALSE)