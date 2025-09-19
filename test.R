library(dplyr); library(tidyr); library(purrr)
library(rstatix); library(PMCMRplus); library(broom)

metrics <- c("MSE","NLPD","Coverage","Width","Time")
funcs   <- c("sine","sigmoid","stepwise")
out_dir <- "results_stats"
if(!dir.exists(out_dir)) dir.create(out_dir)

read_func <- function(func_name){
  files <- c(
    IP  = paste0("results_csv/ip_runs_",  func_name, ".csv"),
    CGP = paste0("results_csv/cgp_runs_", func_name, ".csv"),
    SR  = paste0("results_csv/sr_runs_",  func_name, ".csv"),
    BF  = paste0("results_csv/bf_runs_",  func_name, ".csv")
  )
  dfl <- lapply(names(files), function(m){
    read.csv(files[[m]], stringsAsFactors = FALSE) |>
      mutate(method = m, func = func_name)
  })
  bind_rows(dfl) |>
    mutate(method = factor(method, levels=c("IP","CGP","SR","BF")))
}

paired_pairwise <- function(df, metric){
  wide <- df |>
    select(run, method, !!rlang::sym(metric)) |>
    pivot_wider(names_from = method, values_from = !!rlang::sym(metric)) |>
    drop_na()
  meths <- c("IP","CGP","SR","BF")
  combs <- combn(meths, 2, simplify = FALSE)
  map_dfr(combs, function(cc){
    a <- wide[[cc[1]]]; b <- wide[[cc[2]]]
    diff <- a - b
    p_shap <- tryCatch(shapiro.test(diff)$p.value, error=function(e) NA_real_)
    if(!is.na(p_shap) && p_shap > 0.05){
      tt <- t.test(a, b, paired = TRUE)
      eff <- mean(diff)/sd(diff)
      tibble::tibble(metric=metric, pair=paste(cc, collapse=" vs "),
                     test="paired t", p=tt$p.value,
                     effect=eff, effect_name="Cohen_dz")
    } else {
      wt <- wilcox.test(a, b, paired = TRUE, exact = FALSE)
      eff_r <- rstatix::wilcox_effsize(
        data.frame(a=a,b=b) |>
          mutate(id=row_number()) |>
          pivot_longer(c(a,b), names_to="grp", values_to="val") |>
          mutate(grp=factor(grp, levels=c("a","b"))),
        val ~ grp, paired=TRUE
      )$effsize[1]
      tibble::tibble(metric=metric, pair=paste(cc, collapse=" vs "),
                     test="Wilcoxon signed-rank", p=wt$p.value,
                     effect=eff_r, effect_name="rank_biserial_r")
    }
  }) |>
    mutate(p_holm = p.adjust(p, method = "holm")) |>
    arrange(p_holm)
}

run_friedman <- function(df, metric){
  wide <- df |>
    select(run, method, !!rlang::sym(metric)) |>
    pivot_wider(names_from = method, values_from = !!rlang::sym(metric)) |>
    drop_na()
  mat <- as.matrix(wide[, c("IP","CGP","SR","BF")])
  fr  <- friedman.test(mat)
  tibble::tibble(metric = metric, stat = unname(fr$statistic),
                 df = unname(fr$parameter), p = fr$p.value)
}

for (f in funcs){
  cat("\n====================\nFunction:", f, "\n====================\n")
  dff <- read_func(f)
  
  # (1) 描述统计
  
  # (3) Pairwise
  pairwise_tab <- map_dfr(metrics, ~ paired_pairwise(dff, .x))
  write.csv(pairwise_tab, file.path(out_dir, paste0("pairwise_", f, ".csv")), row.names=FALSE)
  
  # (5) Optional: Binomial tests
  overall_cov_ge95 <- dff %>%
    group_by(method) %>%
    summarise(
      n        = n(),
      mean_cov = mean(Coverage),
      sd_cov   = sd(Coverage),
      
      # 单样本 t 检验：H0: mu <= 0.95 vs H1: mu > 0.95
      t_obj  = list(t.test(Coverage, mu = 0.95, alternative = "greater")),
      # 单样本 Wilcoxon：H0: median <= 0.95 vs H1: median > 0.95
      p_w_gt = tryCatch(
        wilcox.test(Coverage, mu = 0.95, alternative = "greater", exact = FALSE)$p.value,
        error = function(e) NA_real_
      ),
      .groups = "drop"
    ) %>%
    mutate(
      t_stat   = purrr::map_dbl(t_obj, ~ .x$statistic %||% NA_real_),
      p_t_gt   = purrr::map_dbl(t_obj, ~ .x$p.value   %||% NA_real_),
      # 取出双侧 95%CI 仅作参考（若下界 > 0.95，更强支撑 ≥95%）
      ci_lo    = purrr::map_dbl(t_obj, ~ .x$conf.int[1] %||% NA_real_),
      ci_hi    = purrr::map_dbl(t_obj, ~ .x$conf.int[2] %||% NA_real_),
      
      # 同一函数内做多重校正（方法间）
      p_t_gt_holm = p.adjust(p_t_gt, method = "holm"),
      p_w_gt_holm = p.adjust(p_w_gt, method = "holm")
    ) %>%
    select(method, n, mean_cov, sd_cov, ci_lo, ci_hi,
           t_stat, p_t_gt, p_t_gt_holm, p_w_gt, p_w_gt_holm)
  
  # 保存
  write.csv(overall_cov_ge95,
            file.path(out_dir, paste0("cov_overall_ge95_", f, ".csv")),
            row.names = FALSE)
}