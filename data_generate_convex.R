f_parabola <- function(x) 5 * (x/25)^2 - 2                              # parabola
f_cosh <- function(x) (1/1050) * cosh(x) - 0.55                    # cosh
f_stepwise_convex <- function(x) ifelse(x < -5, - x - 5, ifelse(x > 5, x - 5, 0)) # stepwise


make_datasets_csv <- function(R = 25, n = 50, nstar = 100, sigma = 0.5,
                              f_true, out_dir = "datasets_25_csv", seed = 42,
                              edge_k = 4,
                              stratified = TRUE,
                              frac_rep = 0.10,
                              rep_times = 3
){
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  set.seed(seed)
  
  xr <- c(-10, 10)
  
  xt <- seq(xr[1], xr[2], length.out = nstar)
  y_true <- f_true(xt)
  write.csv(data.frame(xt = xt, y_true = y_true),
            file.path(out_dir, "test_grid.csv"), row.names = FALSE)
  
  draw_x <- function(n) {
    k <- min(edge_k, floor(n/5))
    n_core <- n - 2*max(k, 0)
    x_edgeL <- if (k > 0) runif(k, xr[1], mean(xr)) else numeric(0)
    x_edgeR <- if (k > 0) runif(k, mean(xr), xr[2]) else numeric(0)
    
    if (!stratified || n_core <= 0) {
      x_core <- runif(max(n_core, 0), xr[1], xr[2])
    } else {
      breaks <- seq(xr[1], xr[2], length.out = n_core + 1)
      x_core <- vapply(seq_len(n_core), function(i) runif(1, breaks[i], breaks[i+1]), numeric(1))
    }
    
    sort(c(x_edgeL, x_core, x_edgeR))
  }
  
  for (r in seq_len(R)) {
    x <- draw_x(n)
    
    if (frac_rep > 0 && rep_times > 1) {
      n_rep <- max(0, floor(frac_rep * length(x)))
      if (n_rep > 0) {
        id_rep <- sample(seq_along(x), n_rep)
        x_rep  <- rep(x[id_rep], each = rep_times - 1)
        x <- sort(c(x, x_rep))
      }
    }
    
    y <- f_true(x) + rnorm(length(x), 0, sigma)
    
    write.csv(data.frame(x = x, y = y),
              file.path(out_dir, sprintf("train_run%02d.csv", r)),
              row.names = FALSE)
  }
  
  message("CSV saved under: ", normalizePath(out_dir))
}

make_datasets_csv(R = 25, n = 50, nstar = 100, sigma = 0.3,
                  f_true = f_parabola, out_dir = "datasets_parabola",
                  seed = 42,
                  edge_k = 4,
                  stratified = TRUE,
                  frac_rep = 0.1,
                  rep_times = 3)

make_datasets_csv(
  R = 25,
  n = 60,
  nstar = 100,
  sigma = 0.2,
  f_true = f_cosh,
  out_dir = "datasets_cosh",
  seed = 42,
  edge_k = 4,
  stratified = TRUE,
  frac_rep = 0.1,
  rep_times = 3
)




