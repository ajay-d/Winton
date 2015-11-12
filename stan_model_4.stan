data {
  int<lower=0> N;
  vector [N] y;
  matrix [N, 25] covar;
  vector [N] y_m2;
  vector [N] y_m1;
  
  vector [N] weights;
  
  //one-dimensional array of size N containing real values
  //real y[N];
  //real y_hat[N];
  //vector[N] y_hat;
}
parameters {
  //intercept
  //vector [N] alpha;
  real alpha;
  
  vector[N] epsilon;
  
  //regression
  vector [25] beta;
  
  //moving average
  simplex[2] theta;
  //vector[2] theta;
  
  real<lower=0> sigma;
}
transformed parameters{
  //matrix [N,2] covar;
  matrix [N,2] y_m;
  
  //average return over two days
  vector [N] y_m1_m2;
  y_m1_m2 <- (y_m1 + 1) .* (y_m2 + 1);
  for (n in 1:N)
    y_m1_m2[n] <- pow(y_m1_m2[n], .5) - 1;
  
  
  //covar <- append_col(covar1, covar2);
  y_m <- append_col(y_m1, y_m2);
  //y_m <- append_col(y_m, y_m1_m2);
  
  //print("cols in y=", cols(y_m));
  
}
model {
  vector[N] y_hat;
  //vector[N] epsilon;
  
  y_hat <- alpha + covar * beta + y_m * theta;
  epsilon <- (y - y_hat) .* weights;
  
  // priors
  alpha ~ normal(0,10);
  beta ~ normal(0,2);
  
  theta ~ normal(cols(y_m),2);
  
  //sigma ~ cauchy(0,5);
  sigma ~ normal(0,.1);
  
  // likelihood
  epsilon ~ normal(0,sigma);
  
  //p33
  //exp(epsilon) ~ normal(0,sigma);
  //increment_log_prob(epsilon);
  
  //p268
  //increment_log_prob(normal_log(epsilon, 0, sigma));
  
}