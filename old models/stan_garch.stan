data {
  int<lower=0> N;
  vector [N] y;
  matrix [N, 25] covar;
  vector [N] y_m2;
  vector [N] y_m1;
  
  vector [N] weights;
}
parameters {
  
  vector [N] mu;
  vector<lower=0> [N] alpha0;
  vector<lower=0,upper=1> [N] alpha1;

}
transformed parameters {
  vector<lower=0> [N] sigma;
  for (n in 1:N)
    sigma[n] <- fabs(y[n] - y_m1[n]) * weights[n];
  //sigma <- (y - y_m1) .* weights;
} 
model {
  y ~ normal(mu,sigma);
}