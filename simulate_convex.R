source("BF_convex_1d.R", encoding = "UTF-8")  # AGP_convex_SE
source("SR_convex_1d.R", encoding = "UTF-8")  # bsar_convex
source("baseline_convex.R",       encoding = "UTF-8")  # gp_unconstrained /  convex_spline_scam 

root_cosh    <- "datasets_cosh"
root_parabola <- "datasets_parabola"
root_stepwise_convex  <- "datasets_stepwise_convex"

out_root <- "results_convex"
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

## ========== 1) COSH ==========
test_df  <- read.csv(file.path(root_cosh, "test_grid.csv"))
xt_cosh     <- as.numeric(test_df$xt)
y_true_cosh <- as.numeric(test_df$y_true)

rows_cosh <- list(); k <- 1L
agp_cosh_runs  <- list()
bsam_cosh_runs <- list()

for (i in 1:25) {
  tr <- read.csv(file.path(root_cosh, sprintf("train_run%02d.csv", i)))
  x <- as.numeric(tr$x); y <- as.numeric(tr$y)
  
  res_agp   <- AGP_convex_SE(x, y, xt_cosh, y_true_cosh, m=5, nsim=1000, maxeval=200, seed=7)
  res_gp    <- gp_unconstrained(x, y, xt_cosh, y_true_cosh, nugget.estim=TRUE)
  res_scam  <- convex_spline_scam(x, y, xt_cosh, y_true_cosh, shape="convex", k=40)
  res_bsam  <- bsar_convex(x, y, xt_cosh, y_true_cosh, nbasis=50, alpha=0.05, seed=2025)
  
  rows_cosh[[k]] <- mk_row("cosh", i, "AGP",   res_agp);   k <- k + 1L
  rows_cosh[[k]] <- mk_row("cosh", i, "GP",    res_gp);    k <- k + 1L
  rows_cosh[[k]] <- mk_row("cosh", i, "SCAM",  res_scam);  k <- k + 1L
  rows_cosh[[k]] <- mk_row("cosh", i, "BSAM",  res_bsam);  k <- k + 1L
  
  agp_cosh_runs[[i]]  <- data.frame(run=i, MSE=res_agp$mse, NLPD=res_agp$nlpd,
                                    Coverage=res_agp$coverage, Width=res_agp$width, Time=res_agp$time)
  bsam_cosh_runs[[i]] <- data.frame(run=i, MSE=res_bsam$mse, NLPD=res_bsam$nlpd,
                                    Coverage=res_bsam$coverage, Width=res_bsam$width, Time=res_bsam$time)
  cat(sprintf("cosh    run %02d/25 done\n", i))
}
res_cosh        <- do.call(rbind, rows_cosh)
agp_cosh_table  <- do.call(rbind, agp_cosh_runs)
bsam_cosh_table <- do.call(rbind, bsam_cosh_runs)

## ========== 2) parabola ==========
test_df  <- read.csv(file.path(root_parabola, "test_grid.csv"))
xt_parabola     <- as.numeric(test_df$xt)
y_true_parabola <- as.numeric(test_df$y_true)

rows_parabola <- list(); k <- 1L
agp_parabola_runs  <- list()
bsam_parabola_runs <- list()

for (i in 1:25) {
  tr <- read.csv(file.path(root_parabola, sprintf("train_run%02d.csv", i)))
  x <- as.numeric(tr$x); y <- as.numeric(tr$y)
  
  res_agp   <- AGP_convex_SE(x, y, xt_parabola, y_true_parabola, m=5, nsim=1000, maxeval=200, seed=7)
  res_gp    <- gp_unconstrained(x, y, xt_parabola, y_true_parabola, nugget.estim=TRUE)
  res_scam  <- convex_spline_scam(x, y, xt_parabola, y_true_parabola, shape="convex", k=40)
  res_bsam  <- bsar_convex(x, y, xt_parabola, y_true_parabola, nbasis=50, alpha=0.05, seed=2025)
  
  rows_parabola[[k]] <- mk_row("parabola", i, "AGP",   res_agp);   k <- k + 1L
  rows_parabola[[k]] <- mk_row("parabola", i, "GP",    res_gp);    k <- k + 1L
  rows_parabola[[k]] <- mk_row("parabola", i, "SCAM",  res_scam);  k <- k + 1L
  rows_parabola[[k]] <- mk_row("parabola", i, "BSAM",  res_bsam);  k <- k + 1L
  
  agp_parabola_runs[[i]]  <- data.frame(run=i, MSE=res_agp$mse, NLPD=res_agp$nlpd,
                                   Coverage=res_agp$coverage, Width=res_agp$width, Time=res_agp$time)
  bsam_parabola_runs[[i]] <- data.frame(run=i, MSE=res_bsam$mse, NLPD=res_bsam$nlpd,
                                   Coverage=res_bsam$coverage, Width=res_bsam$width, Time=res_bsam$time)
  cat(sprintf("parabola run %02d/25 done\n", i))
}
res_parabola        <- do.call(rbind, rows_parabola)
agp_parabola_table  <- do.call(rbind, agp_parabola_runs)
bsam_parabola_table <- do.call(rbind, bsam_parabola_runs)

## ========== 3) stepwise_convex ==========
test_df  <- read.csv(file.path(root_stepwise_convex, "test_grid.csv"))
xt_sc     <- as.numeric(test_df$xt)
y_true_sc <- as.numeric(test_df$y_true)

rows_sc <- list(); k <- 1L
agp_sc_runs  <- list()
bsam_sc_runs <- list()

for (i in 1:25) {
  tr <- read.csv(file.path(root_stepwise_convex, sprintf("train_run%02d.csv", i)))
  x <- as.numeric(tr$x); y <- as.numeric(tr$y)
  
  res_agp   <- AGP_convex_SE(x, y, xt_sc, y_true_sc, m=5, nsim=1000, maxeval=200, seed=7)
  res_gp    <- gp_unconstrained(x, y, xt_sc, y_true_sc, nugget.estim=TRUE)
  res_scam  <- convex_spline_scam(x, y, xt_sc, y_true_sc, shape="convex", k=40)
  res_bsam  <- bsar_convex(x, y, xt_sc, y_true_sc, nbasis=50, alpha=0.05, seed=2025)
  
  rows_sc[[k]] <- mk_row("stepwise_convex", i, "AGP",   res_agp);   k <- k + 1L
  rows_sc[[k]] <- mk_row("stepwise_convex", i, "GP",    res_gp);    k <- k + 1L
  rows_sc[[k]] <- mk_row("stepwise_convex", i, "SCAM",  res_scam);  k <- k + 1L
  rows_sc[[k]] <- mk_row("stepwise_convex", i, "BSAM",  res_bsam);  k <- k + 1L
  
  agp_sc_runs[[i]]  <- data.frame(run=i, MSE=res_agp$mse, NLPD=res_agp$nlpd,
                                   Coverage=res_agp$coverage, Width=res_agp$width, Time=res_agp$time)
  bsam_sc_runs[[i]] <- data.frame(run=i, MSE=res_bsam$mse, NLPD=res_bsam$nlpd,
                                   Coverage=res_bsam$coverage, Width=res_bsam$width, Time=res_bsam$time)
  cat(sprintf("stepwise_convex  run %02d/25 done\n", i))
}
res_sc        <- do.call(rbind, rows_sc)
agp_sc_table  <- do.call(rbind, agp_sc_runs)
bsam_sc_table <- do.call(rbind, bsam_sc_runs)


raw_all     <- rbind(res_cosh, res_parabola, res_sc)
summary_all <- summarize_tbl(raw_all)

write.csv(summary_all, file.path(out_root, "summary_all_mean_sd.csv"), row.names = FALSE)

write.csv(agp_cosh_table,  file.path(out_root, "bf_runs_cosh.csv"),   row.names = FALSE)
write.csv(agp_parabola_table,   file.path(out_root, "bf_runs_parabola.csv"),row.names = FALSE)
write.csv(agp_sc_table,   file.path(out_root, "bf_runs_stepwise_convex.csv"), row.names = FALSE)

write.csv(bsam_cosh_table, file.path(out_root, "sr_runs_cosh.csv"),   row.names = FALSE)
write.csv(bsam_parabola_table,  file.path(out_root, "sr_runs_parabola.csv"),row.names = FALSE)
write.csv(bsam_sc_table,  file.path(out_root, "sr_runs_stepwise_convex.csv"), row.names = FALSE)

cat("\n saved：\n -", file.path(out_root, "convex_summary_all_mean_sd.csv"),
    "\n -", file.path(out_root, "bf_runs_cosh.csv"),
    "\n -", file.path(out_root, "bf_runs_parabola.csv"),
    "\n -", file.path(out_root, "bf_runs_stepwise_convex.csv"),
    "\n -", file.path(out_root, "sr_runs_cosh.csv"),
    "\n -", file.path(out_root, "sr_runs_parabola.csv"),
    "\n -", file.path(out_root, "sr_runs_stepwise_convex.csv"), "\n")
