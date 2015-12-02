data {
  
  int<lower=0> T; //time periods
  int<lower=0> N; //training obs
  //int<lower=0> N_new; //test obs
  
  int<lower=0> D; //levels of segmentation
  
  int<lower=0> Q; //lags
  
  matrix [N, 60] y;
  matrix [N, 25] covar;
  
  //intra-day returns
  matrix [N, T] x_intra;
  
  vector [N] weights;
}

transformed data{
  
  matrix [N, 25] covar_sq;
  
  for (i in 1:N)
    for (j in 1:25)
      covar_sq[i,j] <- pow(covar[i,j], 2);
}

parameters {
  
  //intercept
  vector [60] alpha;
  
  //regression on features
  //vector[25] beta[D];
  //vector[25] beta_sq[D];
  matrix [25, 60] beta;
  matrix [25, 60] beta_sq;
  
  //Thetas for all prior returns, and combined returns
  //vector[Q] theta_1[D];
  matrix [Q, 60] theta;
  
  vector [60] sigma;
  
}

transformed parameters{
  
  matrix [N, 60] epsilon;    // error term at time t
  matrix [N, 60] weighted_err;

  matrix [N, 60] mu;
  matrix [N, Q] x_returns;
  
  vector [N] norm;
  
  for (n in 1:N)
    for (q in 1:Q)
      x_returns[n,q] <- x_intra[n,(T-q+1)];

//   for (t in 1:60)
//     col(mu,t) <- alpha[t] + 
//       covar * col(beta,t) + 
//       covar_sq * col(beta_sq,t) +
//       x_returns * col(theta,t) + 
//       sigma[t];
  
  for (n in 1:N)
    mu[n] <- alpha' + 
      covar[n] * beta + 
      covar_sq[n] * beta_sq +
      x_returns[n] * theta + 
      sigma';
    
  for (n in 1:N)
    for (t in 1:60)
      epsilon[n,t] <- y[n,t] - mu[n,t]; //error for each forcast
  
  norm <- weights ./ (sum(weights)/rows(weights));
  
  //abs??
  weighted_err <- epsilon .* rep_matrix(norm, 60);
  
}

model {
  
  // priors
  alpha ~ normal(0,2);
  
  for (j in 1:60){
    col(beta,j) ~ normal(0,2);
    col(beta_sq,j) ~ normal(0,2);
	  
	  col(theta,j) ~ normal(0,2);
    }

  //increment_log_prob(-sum(weighted_err));
  for(i in 1:60)
    col(weighted_err,i) ~ normal(0,2);
  
}