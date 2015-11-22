library(readr)
library(dplyr)
library(tidyr)
library(rstan)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

fit1 <- lm(mpg ~ am+wt+hp+disp+cyl,data=mtcars)
nrow(mtcars)

fit1$coefficients

x <- mtcars %>%
  #select(cyl:carb) %>%
  select(am,wt,hp,disp,cyl) %>%
  as.matrix()

dim(x)

dat <- list(
  "N" = nrow(mtcars),
  "N_x" = dim(x)[[2]],
  "x" = x,
  "y" = mtcars$mpg)

stan.model <- "
data {
  int<lower=0> N;
  int<lower=1> N_x;
  matrix[N,N_x] x;
  vector[N] y;
}
parameters {
  real alpha;
  vector[N_x] beta;
  real<lower=0> sigma;
} 
model {
  y ~ normal(alpha + x*beta, sigma);
}
"
fit2 <- stan(model_code = stan.model, 
            model_name = "Stan2", 
            iter=2500, warmup=500,
            thin=2, chains=4, seed=252014,
            data = dat)
print(fit2, pars=c("alpha", "beta", 'sigma'), probs=c(0.5, 0.75, 0.95))
fit$coefficients

traceplot(fit2, pars=c("alpha", "beta", 'sigma'))
stan_dens(fit, pars=c("beta1", "beta2", "theta1", "theta2"), 
          fill="skyblue")

stan.optim1 <- "
data {
  int<lower=0> N;
  int<lower=1> N_x;
  matrix[N,N_x] x;
  vector[N] y;
}
parameters {
  real alpha;
  vector[N_x] beta;
} 
transformed parameters {
  real<lower=0> squared_error;
  //squared_error <- dot_self(alpha + x*beta - y);
  squared_error <- dot_self(y - alpha + x*beta);
}
model {
  increment_log_prob(-squared_error);
}
"

stan.optim2 <- "
data {
  int<lower=0> N;
  int<lower=1> N_x;
  matrix[N,N_x] x;
  vector[N] y;
}
parameters {
  real alpha;
  vector[N_x] beta;
} 
transformed parameters {
  real<lower=0> squared_error;
  vector[N] y_hat;
  y_hat <- alpha + x*beta;
  squared_error <- dot_self(y-y_hat);
  squared_error <- dot_self(y_hat)-y;
}
model {
  increment_log_prob(-squared_error);
}
"

fit3 <- stan(model_code = stan.optim2, 
             model_name = "Stan3", 
             iter=2500, warmup=1500,
             thin=2, chains=4, seed=252014,
             data = dat)

print(fit2, pars=c("alpha", "beta", 'sigma'), probs=c(0.5, 0.75, 0.95))
print(fit3, pars=c("alpha", "beta", 'squared_error'), probs=c(0.5, 0.75, 0.95))
