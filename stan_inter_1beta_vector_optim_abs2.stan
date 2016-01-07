data {
  int<lower=0> N;
  
  //Two days to forecast
  matrix[N,2] y;
  
  int<lower=0> D; //levels of segmentation
  
  //matrix[N,25] covar;
  row_vector[25] covar[N];
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
  
  vector [N] norm;
  matrix [N,3] x;
    
  norm <- weights ./ (sum(weights)/rows(weights));
  
  x <- append_col(append_col(x_m2, x_m1), x_intra);
}

parameters {
  //intercept
  real alpha[D,2];
  
  //regression
  vector[25] beta[D,2];
  //simplex[25] beta_sq[D,2];
  
  //moving average
  //vector[3] theta[D,2];
  vector[3] theta[D,2];
  
  //real<lower=0> sigma_1;
  //real<lower=0> sigma_2;
}

model {
  matrix[N,2] y_hat;
  
  matrix[N,2] epsilon;
  matrix[N,2] weighted_err;
  //row_vector[N] squared_error;
  row_vector[N] abs_error;
  
  for (n in 1:N) {
    y_hat[n,1] <- alpha[ll[n], 1] + 
                  covar[n] * beta[ll[n], 1] + 
                  //covar_sq[n] * beta_sq[ll[n], 1] + 
                  x[n] * theta[ll[n], 1];
                  
    y_hat[n,2] <- alpha[ll[n], 2] + 
                  covar[n] * beta[ll[n], 2] + 
                  //covar_sq[n] * beta_sq[ll[n], 2] + 
                  x[n] * theta[ll[n], 2];
    
    epsilon[n,1] <- y[n,1] - y_hat[n,1];
    epsilon[n,2] <- y[n,2] - y_hat[n,2];
  }
  
  weighted_err <- append_col(col(epsilon,1) .* norm, col(epsilon,2) .* norm);
  
  //for(i in 1:N)
    //squared_error[i] <- dot_self(weighted_err[i]);
  for(i in 1:N)
    abs_error[i] <- fabs(weighted_err[i,1]) + fabs(weighted_err[i,2]);
  
  // priors
  for(d in 1:D)
    for(i in 1:2) {
      alpha[d,i] ~ normal(0, 1);
      theta[d,i] ~ normal(1, 1);
      
      beta[d,i] ~ normal(0, .1);
      //beta_sq[d,i] ~ normal(0,2);
    }
  
  //sigma ~ cauchy(0,5);
  //sigma_1 ~ normal(0,1)T[0,];
  //sigma_2 ~ normal(0,1)T[0,];
  
  // likelihood
  for(i in 1:N) {
    //epsilon[i] ~ normal(0, .01);
    //weighted_err[i] ~ normal(0, .01);
    increment_log_prob(-abs_error[i]);
  }


}