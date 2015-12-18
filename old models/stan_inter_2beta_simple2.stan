data {
  int<lower=0> N;
  
  //Two days to forecast
  matrix[N,2] y;
  
  int<lower=0> D; //levels of segmentation
  
  matrix[N,25] covar;
  vector[N] x_m2;
  vector[N] x_m1;
  
  //group (level)
  //vector [N] ll;
  int ll[N];
  
  vector [N] weights;
  
}

transformed data{
  
  matrix [N,25] covar_sq;
  //matrix[N_new, 25] covar_new_sq;
  
  vector [N] norm;
  
  for (i in 1:N)
    for (j in 1:25)
      covar_sq[i,j] <- pow(covar[i,j], 2);
    
//   for (i in 1:N_new)
//     for (j in 1:25)
//       covar_new_sq[i,j] <- pow(covar_new[i,j], 2);

  norm <- weights ./ (sum(weights)/rows(weights));
      
}

parameters {
  //intercept
  real alpha[D,2];
  
  //regression
  vector[25] beta;
  
  //moving average
  simplex[2] theta;
  
  real<lower=0> sigma;
}

model {
  matrix[N,2] y_hat;
  vector[N] epsilon;
  vector [N] weighted_err;
  
  for (n in 1:N) {
    y_hat[n,1] <- alpha[ll[n], 1] + row(covar, n) * beta + theta[1]*x_m1[n] + theta[2]*x_m2[n];
    y_hat[n,2] <- alpha[ll[n], 2] + row(covar, n) * beta + theta[1]*x_m1[n] + theta[2]*x_m2[n];
    
    epsilon[n] <- y[n,1] - y_hat[n,1] + y[n,2] - y_hat[n,2];
  }
  
  weighted_err <- epsilon .* norm;
  
  // priors
  for(d in 1:D)
    for(i in 1:2)
      alpha[d,i] ~ normal(0,2);

  theta ~ normal(0,2);
  
  //sigma ~ cauchy(0,5);
  sigma ~ normal(0,5);
  
  // likelihood
  epsilon ~ normal(0,sigma);
  weighted_err ~ normal(0,5);
  
  //p33
  //exp(epsilon) ~ normal(0,sigma);
  //increment_log_prob(epsilon);
  
  //p268
  //increment_log_prob(normal_log(epsilon, 0, sigma));

}