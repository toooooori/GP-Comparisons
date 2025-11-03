library("lineqGPR")
library("DiceDesign")
library("plot3D")
library("viridis")
library("Matrix")

rm(list=ls())
set.seed(7)
n  <- 50
x1 <- runif(n)
x2 <- runif(n)
X  <- cbind(x1, x2)

#sigm <- function(z) 1/(1 + exp(-z))
#f    <- function(x1, x2) sigm(10*(x1 - 0.4)) + sigm(10*(x2 - 0.6))
f    <- function(x1, x2) exp(x1) + x2 + sin(20*x2-10)/20
y    <- f(x1, x2) + rnorm(n, sd = 0.5)
d <- 2

model <- create(class = "lineqAGP",
                x = X, y = y,
                constrType = c("none", "monotonicity"),
                m = 10)

for (k in 1:d) model$kernParam[[k]]$par <- c(1, 2)
model$localParam$sampler <- "HMC"
model$nugget   <- 1e-7
model$varnoise <- 0.25

t0 <- proc.time()
model <- lineqGPOptim(
  model,
  x0   = unlist(purrr::map(model$kernParam, "par")),
  eval_f = "logLik",
  additive = TRUE,
  opts = list(algorithm = "NLOPT_LD_MMA", print_level = 3,
              ftol_abs = 1e-3, maxeval = 1e2, check_derivatives = FALSE),
  lb = rep(1e-2, 2*d),
  ub = rep(c(5, 3), d),
  estim.varnoise = TRUE,
  bounds.varnoise = c(1e-4, Inf)
)

ntest <- 100
tgrid <- seq(0, 1, length = ntest)
xtest <- matrix(tgrid, nrow = ntest, ncol = d)
Z_true <- outer(tgrid, tgrid, function(a, b) f(a, b))

nsim <- 1000
sim.model <- simulate(model, nsim = nsim, xtest = xtest)

pred <- predict(model, xtest)

PhiAll.test <- cbind(
  pred$Phi.test[[1]][rep(1:ntest, times = ntest), ],
  pred$Phi.test[[2]][rep(1:ntest, each  = ntest), ]
)

Z_post_mean <- matrix(
  rowMeans(PhiAll.test %*% sim.model$xiAll.sim),  # ntest^2 行
  nrow = ntest, ncol = ntest
)

mse_test <- mean((c(Z_post_mean) - c(Z_true))^2)
cost_time <- (proc.time() - t0)[["elapsed"]]

colormap <- rev(viridis(100))

par(mfrow = c(1,1), mar = c(2,2,2,1))

persp3D(
  x = tgrid, y = tgrid, z = Z_true,
  main = "True surface (mesh) + MCMC posterior mean",
  xlab = "x1", ylab = "x2", zlab = "y",
  facets = FALSE,
  col = "grey85", border = "grey40",
  phi = 20, theta = -30, colkey = FALSE
)

persp3D(
  x = tgrid, y = tgrid, z = Z_post_mean,
  add = TRUE,
  col = colormap, alpha = 0.6,
  border = NA, colkey = TRUE
)

df_export <- data.frame(
  x = rep(tgrid, times = ntest),
  y = rep(tgrid, each  = ntest),
  z_true = as.vector(Z_true),
  z_post_mean = as.vector(Z_post_mean)
)

write.csv(df_export, "bf_2D_results.csv", row.names = FALSE)
