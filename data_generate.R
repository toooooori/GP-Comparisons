# ----- 可选：安装一次 -----
# install.packages("viridis")  # 不是必须，仅你后续画图会用到

#f_true <- function(x) x + sin(x-10)                                   # linear + sine
#f_true <- function(x) 4 / (1 + exp(-(x - 10)/2 + 4))                 # sigmoid
f_true <- function(x) ifelse(x < 5, x, ifelse(x > 15, x - 10, 5))  # stepwise

make_datasets_csv <- function(R = 25, n = 50, nstar = 100, sigma = 0.5,
                              f_true, out_dir = "datasets_25_csv", seed = 2025) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  set.seed(seed)
  
  # 测试集（所有数据副本共用）
  xt <- seq(0, 10, length.out = nstar)
  y_true <- f_true(xt)
  write.csv(data.frame(xt = xt, y_true = y_true),
            file.path(out_dir, "test_grid.csv"), row.names = FALSE)
  
  for (r in seq_len(R)) {
    x <- runif(n, 0, 10)
    y <- f_true(x) + rnorm(n, 0, sigma)
    write.csv(data.frame(x = x, y = y),
              file.path(out_dir, sprintf("train_run%02d.csv", r)),
              row.names = FALSE)
  }
  
  message("CSV saved under: ", normalizePath(out_dir))
}

# 运行一次生成
make_datasets_csv(R = 25, n = 50, nstar = 100, sigma = 0.5,
                  f_true = f_true, out_dir = "datasets_sin_csv")
