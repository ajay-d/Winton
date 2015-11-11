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
  simplex[2] theta;
  
  real<lower=0> sigma;
}
model {
  vector[N] y_hat;
  vector[N] epsilon;
  
  for (n in 1:N) {
    y_hat[n] <- alpha + beta1*covar1[n] + beta2*covar2[n] + theta[1]*y_m1[n] + theta[2]*y_m2[n];
    epsilon[n] <- y[n] - y_hat[n];
  }
  // priors
  alpha ~ normal(0,10);
  beta1 ~ normal(0,2);
  beta2 ~ normal(0,2);
  
  theta ~ normal(0,2);
  
  sigma ~ cauchy(0,5);
  
  // likelihood
  epsilon ~ normal(0,sigma);
  
  //p33
  //exp(epsilon) ~ normal(0,sigma);
  //increment_log_prob(epsilon);
  
  //p268
  //increment_log_prob(normal_log(epsilon, 0, sigma));

}