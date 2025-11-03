source("BF_monotone_1d.R", encoding = "UTF-8")  # AGP_monotone_SE
source("SR_monotone_1d.R", encoding = "UTF-8")  # bsar_monotone
source("baseline.R",       encoding = "UTF-8")  # gp_unconstrained / monotone_isotonic / monotone_spline_scam / monotone_bart

root_sine    <- "datasets_sine"
root_sigmoid <- "datasets_sigmoid"
root_shrink  <- "datasets_stepwise_monotone"

out_root <- "results_monotone"
dir.create(out_root, showWarnings = FALSE)

mk_row <- function(func, run, method, res) {
  data.frame(func = func, run = run, method = method,
             MSE = res$mse, NLPD = res$nlpd, Coverage = res$coverage,
             Width = res$width, Time = res$time,
             row.names = NULL, check.names = FALSE)
}

summarize_tbl <- function(df) {
  agg <- aggregate(cbind(MSE,NLPD,Coverage,Width,Time) ~ func + method,
                   data = df,
                   FUN  = function(v) c(mean = mean(v), sd = sd(v)))
  out <- do.call(data.frame, agg)
  names(out) <- c("func","method",
                  "MSE_mean","MSE_sd","NLPD_mean","NLPD_sd",
                  "Coverage_mean","Coverage_sd","Width_mean","Width_sd","Time_mean","Time_sd")
  out
}

## ========== 1) SINE ==========
test_df  <- read.csv(file.path(root_sine, "test_grid.csv"))
xt_sine     <- as.numeric(test_df$xt)
y_true_sine <- as.numeric(test_df$y_true)

rows_sine <- list(); k <- 1L
agp_sine_runs  <- list()
bsam_sine_runs <- list()

for (i in 1:25) {
  tr <- read.csv(file.path(root_sine, sprintf("train_run%02d.csv", i)))
  x <- as.numeric(tr$x); y <- as.numeric(tr$y)
  
  res_agp   <- AGP_monotone_SE(x, y, xt_sine, y_true_sine, m=5, nsim=1000, maxeval=200, seed=7)
  res_gp    <- gp_unconstrained(x, y, xt_sine, y_true_sine, nugget.estim=TRUE)
  res_iso   <- monotone_isotonic(x, y, xt_sine, y_true_sine)
  res_scam  <- monotone_spline_scam(x, y, xt_sine, y_true_sine, monotone="increasing", k=40)
  res_mbart <- monotone_bart(x, y, xt_sine, y_true_sine, ntree=50, ndpost=200, nskip=100, mgsize=20)
  res_bsam  <- bsar_monotone(x, y, xt_sine, y_true_sine, nbasis=50, alpha=0.05, seed=2025)
  
  rows_sine[[k]] <- mk_row("sine", i, "AGP",   res_agp);   k <- k + 1L
  rows_sine[[k]] <- mk_row("sine", i, "GP",    res_gp);    k <- k + 1L
  rows_sine[[k]] <- mk_row("sine", i, "ISO",   res_iso);   k <- k + 1L
  rows_sine[[k]] <- mk_row("sine", i, "SCAM",  res_scam);  k <- k + 1L
  rows_sine[[k]] <- mk_row("sine", i, "MBART", res_mbart); k <- k + 1L
  rows_sine[[k]] <- mk_row("sine", i, "BSAM",  res_bsam);  k <- k + 1L
  
  agp_sine_runs[[i]]  <- data.frame(run=i, MSE=res_agp$mse, NLPD=res_agp$nlpd,
                                    Coverage=res_agp$coverage, Width=res_agp$width, Time=res_agp$time)
  bsam_sine_runs[[i]] <- data.frame(run=i, MSE=res_bsam$mse, NLPD=res_bsam$nlpd,
                                    Coverage=res_bsam$coverage, Width=res_bsam$width, Time=res_bsam$time)
  cat(sprintf("sine    run %02d/25 done\n", i))
}
res_sine        <- do.call(rbind, rows_sine)
agp_sine_table  <- do.call(rbind, agp_sine_runs)
bsam_sine_table <- do.call(rbind, bsam_sine_runs)

## ========== 2) SIGMOID ==========
test_df  <- read.csv(file.path(root_sigmoid, "test_grid.csv"))
xt_sig     <- as.numeric(test_df$xt)
y_true_sig <- as.numeric(test_df$y_true)

rows_sig <- list(); k <- 1L
agp_sig_runs  <- list()
bsam_sig_runs <- list()

for (i in 1:25) {
  tr <- read.csv(file.path(root_sigmoid, sprintf("train_run%02d.csv", i)))
  x <- as.numeric(tr$x); y <- as.numeric(tr$y)
  
  res_agp   <- AGP_monotone_SE(x, y, xt_sig, y_true_sig, m=5, nsim=1000, maxeval=200, seed=7)
  res_gp    <- gp_unconstrained(x, y, xt_sig, y_true_sig, nugget.estim=TRUE)
  res_iso   <- monotone_isotonic(x, y, xt_sig, y_true_sig)
  res_scam  <- monotone_spline_scam(x, y, xt_sig, y_true_sig, monotone="increasing", k=40)
  res_mbart <- monotone_bart(x, y, xt_sig, y_true_sig, ntree=50, ndpost=200, nskip=100, mgsize=20)
  res_bsam  <- bsar_monotone(x, y, xt_sig, y_true_sig, nbasis=50, alpha=0.05, seed=2025)
  
  rows_sig[[k]] <- mk_row("sigmoid", i, "AGP",   res_agp);   k <- k + 1L
  rows_sig[[k]] <- mk_row("sigmoid", i, "GP",    res_gp);    k <- k + 1L
  rows_sig[[k]] <- mk_row("sigmoid", i, "ISO",   res_iso);   k <- k + 1L
  rows_sig[[k]] <- mk_row("sigmoid", i, "SCAM",  res_scam);  k <- k + 1L
  rows_sig[[k]] <- mk_row("sigmoid", i, "MBART", res_mbart); k <- k + 1L
  rows_sig[[k]] <- mk_row("sigmoid", i, "BSAM",  res_bsam);  k <- k + 1L
  
  agp_sig_runs[[i]]  <- data.frame(run=i, MSE=res_agp$mse, NLPD=res_agp$nlpd,
                                   Coverage=res_agp$coverage, Width=res_agp$width, Time=res_agp$time)
  bsam_sig_runs[[i]] <- data.frame(run=i, MSE=res_bsam$mse, NLPD=res_bsam$nlpd,
                                   Coverage=res_bsam$coverage, Width=res_bsam$width, Time=res_bsam$time)
  cat(sprintf("sigmoid run %02d/25 done\n", i))
}
res_sig        <- do.call(rbind, rows_sig)
agp_sig_table  <- do.call(rbind, agp_sig_runs)
bsam_sig_table <- do.call(rbind, bsam_sig_runs)

## ========== 3) SHRINK ==========
test_df  <- read.csv(file.path(root_shrink, "test_grid.csv"))
xt_shk     <- as.numeric(test_df$xt)
y_true_shk <- as.numeric(test_df$y_true)

rows_shk <- list(); k <- 1L
agp_shk_runs  <- list()
bsam_shk_runs <- list()

for (i in 1:25) {
  tr <- read.csv(file.path(root_shrink, sprintf("train_run%02d.csv", i)))
  x <- as.numeric(tr$x); y <- as.numeric(tr$y)
  
  res_agp   <- AGP_monotone_SE(x, y, xt_shk, y_true_shk, m=5, nsim=1000, maxeval=200, seed=7)
  res_gp    <- gp_unconstrained(x, y, xt_shk, y_true_shk, nugget.estim=TRUE)
  res_iso   <- monotone_isotonic(x, y, xt_shk, y_true_shk)
  res_scam  <- monotone_spline_scam(x, y, xt_shk, y_true_shk, monotone="increasing", k=40)
  res_mbart <- monotone_bart(x, y, xt_shk, y_true_shk, ntree=50, ndpost=200, nskip=100, mgsize=20)
  res_bsam  <- bsar_monotone(x, y, xt_shk, y_true_shk, nbasis=50, alpha=0.05, seed=2025)
  
  rows_shk[[k]] <- mk_row("shrink", i, "AGP",   res_agp);   k <- k + 1L
  rows_shk[[k]] <- mk_row("shrink", i, "GP",    res_gp);    k <- k + 1L
  rows_shk[[k]] <- mk_row("shrink", i, "ISO",   res_iso);   k <- k + 1L
  rows_shk[[k]] <- mk_row("shrink", i, "SCAM",  res_scam);  k <- k + 1L
  rows_shk[[k]] <- mk_row("shrink", i, "MBART", res_mbart); k <- k + 1L
  rows_shk[[k]] <- mk_row("shrink", i, "BSAM",  res_bsam);  k <- k + 1L
  
  agp_shk_runs[[i]]  <- data.frame(run=i, MSE=res_agp$mse, NLPD=res_agp$nlpd,
                                   Coverage=res_agp$coverage, Width=res_agp$width, Time=res_agp$time)
  bsam_shk_runs[[i]] <- data.frame(run=i, MSE=res_bsam$mse, NLPD=res_bsam$nlpd,
                                   Coverage=res_bsam$coverage, Width=res_bsam$width, Time=res_bsam$time)
  cat(sprintf("shrink  run %02d/25 done\n", i))
}
res_shk        <- do.call(rbind, rows_shk)
agp_shk_table  <- do.call(rbind, agp_shk_runs)
bsam_shk_table <- do.call(rbind, bsam_shk_runs)


raw_all     <- rbind(res_sine, res_sig, res_shk)
summary_all <- summarize_tbl(raw_all)

write.csv(summary_all, file.path(out_root, "summary_all_mean_sd.csv"), row.names = FALSE)

write.csv(agp_sine_table,  file.path(out_root, "bf_runs_sine.csv"),   row.names = FALSE)
write.csv(agp_sig_table,   file.path(out_root, "bf_runs_sigmoid.csv"),row.names = FALSE)
write.csv(agp_shk_table,   file.path(out_root, "bf_runs_shrink.csv"), row.names = FALSE)

write.csv(bsam_sine_table, file.path(out_root, "sr_runs_sine.csv"),   row.names = FALSE)
write.csv(bsam_sig_table,  file.path(out_root, "sr_runs_sigmoid.csv"),row.names = FALSE)
write.csv(bsam_shk_table,  file.path(out_root, "sr_runs_shrink.csv"), row.names = FALSE)

cat("\n saved：\n -", file.path(out_root, "summary_all_mean_sd.csv"),
    "\n -", file.path(out_root, "bf_runs_sine.csv"),
    "\n -", file.path(out_root, "bf_runs_sigmoid.csv"),
    "\n -", file.path(out_root, "bf_runs_shrink.csv"),
    "\n -", file.path(out_root, "sr_runs_sine.csv"),
    "\n -", file.path(out_root, "sr_runs_sigmoid.csv"),
    "\n -", file.path(out_root, "sr_runs_shrink.csv"), "\n")
