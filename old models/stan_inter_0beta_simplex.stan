data {
  int<lower=0> N;
  
  //Two days to forecast
  matrix[N,2] y;
  
  int<lower=0> D; //levels of segmentation
  
  matrix[N,25] covar;
  vector [N] x_m2;
  vector [N] x_m1;
  //combo intra-day returns and two prior day returns
  vector [N] x_intra;
  
  //group (level)
  //vector [N] ll;
  int ll[N];
  
  vector [N] weights;
  
}

transformed data{
  
  matrix [N,25] covar_sq;
  vector [N] norm;
  matrix [N,3] x;
  
  for (i in 1:N)
    for (j in 1:25)
      covar_sq[i,j] <- pow(covar[i,j], 2);
    
  norm <- weights ./ (sum(weights)/rows(weights));
  
  x <- append_col(append_col(x_m2, x_m1), x_intra);
}

parameters {
  //intercept
  real alpha[D,2];
  
  //regression
  //simplex[25] beta[D,2];
  //simplex[25] beta_sq[D,2];
  
  //moving average
  //vector[3] theta[D,2];
  simplex[3] theta[D,2];
  
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
}

model {
  matrix[N,2] y_hat;
  
  matrix[N,2] epsilon;
  matrix[N,2] weighted_err;
  
  for (n in 1:N) {
    y_hat[n,1] <- alpha[ll[n], 1] + 
                  //covar[n] * beta[ll[n], 1] + 
                  //covar_sq[n] * beta_sq[ll[n], 1] + 
                  x[n] * theta[ll[n], 1];
                  
    y_hat[n,2] <- alpha[ll[n], 2] + 
                  //covar[n] * beta[ll[n], 2] + 
                  //covar_sq[n] * beta_sq[ll[n], 2] + 
                  x[n] * theta[ll[n], 2];
    
    epsilon[n,1] <- y[n,1] - y_hat[n,1];
    epsilon[n,2] <- y[n,2] - y_hat[n,2];
  }
  
  weighted_err <- append_col(col(epsilon,1) .* norm, col(epsilon,2) .* norm);
  
  // priors
  for(d in 1:D)
    for(i in 1:2) {
      alpha[d,i] ~ normal(0,2);
      theta[d,i] ~ normal(0,2);
      
      //beta[d,i] ~ normal(0,2);
      //beta_sq[d,i] ~ normal(0,2);
    }
  
  //sigma ~ cauchy(0,5);
  sigma_1 ~ normal(0,1)T[0,];
  sigma_2 ~ normal(0,1)T[0,];
  
  // likelihood
  for(i in 1:N) {
    epsilon[i] ~ normal(0,sigma_1);
    weighted_err[i] ~ normal(0,sigma_2);
  }


}