data {
  int<lower=0> N;
  
  //Two days to forecast
  matrix [N, 2] y;
  
  int<lower=0> D; //levels of segmentation
  
  matrix [N, 25] covar;
  vector [N] x_m2;
  vector [N] x_m1;
  
}

transformed data{
  
  matrix [N, 25] covar_sq;
  //matrix[N_new, 25] covar_new_sq;
  
  for (i in 1:N)
    for (j in 1:25)
      covar_sq[i,j] <- pow(covar[i,j], 2);
    
//   for (i in 1:N_new)
//     for (j in 1:25)
//       covar_new_sq[i,j] <- pow(covar_new[i,j], 2);
      
}

parameters {
  //intercept
  real alpha;
  
  //regression
  simplex[25] beta;
  
  //moving average
  simplex[2] theta;
  
  real<lower=0> sigma;
}

model {
  vector[N] y_hat;
  vector[N] epsilon;
  
  for (n in 1:N) {
    y_hat[n] <- alpha + covar[n] * beta + theta[1]*x_m1[n] + theta[2]*x_m2[n];
    //y_hat[n] <- alpha + theta[1]*x_m1[n] + theta[2]*x_m2[n];
    epsilon[n] <- y[n,1] - y_hat[n];
  }
  // priors
  alpha ~ normal(0,10);

  theta ~ normal(0,2);
  
  //sigma ~ cauchy(0,5);
  sigma ~ normal(0,5);
  
  // likelihood
  epsilon ~ normal(0,sigma);
  
  //p33
  //exp(epsilon) ~ normal(0,sigma);
  //increment_log_prob(epsilon);
  
  //p268
  //increment_log_prob(normal_log(epsilon, 0, sigma));

}