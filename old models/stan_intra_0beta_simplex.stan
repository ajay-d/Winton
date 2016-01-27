data {
  
  int<lower=0> T; //time periods
  int<lower=0> N; //training obs
  
  //Forecast period
  matrix [N,60] y;
  
  int<lower=0> D; //levels of segmentation
  
  int<lower=0> Q; //lags
  
  matrix[N,25] covar;
  
  //intra-day returns
  matrix [N,T] x_intra;
  
  //group (level)
  //vector [N] ll;
  int ll[N];
  
  vector [N] weights;
  
}

transformed data{
  
  matrix [N,25] covar_sq;
  vector [N] norm;
  matrix [N,Q] x_returns;

  for (i in 1:N)
    for (j in 1:25)
      covar_sq[i,j] <- pow(covar[i,j], 2);
  
  for (n in 1:N)
    for (q in 1:Q)
      x_returns[n,q] <- x_intra[n,(T-q+1)];
  
  norm <- weights ./ (sum(weights)/rows(weights));
}

parameters {
  //intercept
  real alpha[D,60];
  
  //regression
  //simplex[25] beta[D,2];
  //simplex[25] beta_sq[D,2];
  
  //moving average
  //vector[3] theta[D,2];
  simplex[Q] theta[D,60];
  
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
}

model {
  matrix[N,60] y_hat;
  
  matrix[N,60] epsilon;
  matrix[N,60] weighted_err;
  
  for (n in 1:N) {
    for(j in 1:60) {
      y_hat[n,j] <- alpha[ll[n], j] + 
                    //covar[n] * beta[ll[n], j] + 
                    //covar_sq[n] * beta_sq[ll[n], j] + 
                    x_returns[n] * theta[ll[n], j];
                  
      epsilon[n,j] <- y[n,j] - y_hat[n,j];
    }
  }
  
  weighted_err <- epsilon .* rep_matrix(norm, 60);
  
  // priors
  for(d in 1:D)
    for(j in 1:60) {
      alpha[d,j] ~ normal(0,2);
      theta[d,j] ~ normal(0,2);
      
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