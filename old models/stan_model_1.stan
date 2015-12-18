data {
  int<lower=0> N;
  vector [N] y;
  vector [N] covar1;
  vector [N] covar2;
  vector [N] y_m2;
  vector [N] y_m1;
  
  //one-dimensional array of size N containing real values
  //real y[N];
  //real y_hat[N];
  //vector[N] y_hat;
}
parameters {
  //intercept
  real alpha;
  
  //regression
  real<lower=-1,upper=1> beta1;
  real<lower=-1,upper=1> beta2;
  
  //moving average
  real<lower=0,upper=1> theta1;
  
  real<lower=0> sigma;
}
transformed parameters {
  real<lower=0,upper=1> theta2;
  theta2 <- 1-theta1;
}
model {
  for (n in 1:N)
    y[n] ~ normal(alpha + 
                    beta1*covar1[n] + beta2*covar2[n] + 
                    theta1*y_m1[n] + theta2*y_m2[n],
                  sigma);
}