data {
  
  int<lower=0> T; //time periods
  int<lower=0> N; //obs
  int<lower=0> D; //levels of segmentation
  
  int<lower=0> Q; //lags
  
  vector[N] y;
  matrix[N,25] covar;
  
  //intra-day returns
  matrix[N,T] y_intra;
  
  //group (level)
  //vector [N] ll;
  int ll[N];
  
  vector[N] weights;
  
  //one-dimensional array of size N containing real values
  //real y[N];
  //real y_hat[N];
  //vector[N] y_hat;
  
  //Add non-linear effects from both Features and returns
  //y = A + B*feat + T*returns + B*f(feat) + T*f(returns) / feat^2,  ln(returns)
  //ln(y) = A + B*feat + T*returns
}

transformed data{
  
  matrix[N,25] covar_sq;
  
  for (i in 1:N)
    for (j in 1:25)
      covar_sq[i,j] <- pow(covar[i,j], 2);
    
}

parameters {
  //intercept
  //vector [N] alpha;
  //real alpha;
  real alpha[D];
  
  //real epsilon[D];
  //vector [N] epsilon;
  //real epsilon;
  
  //regression
  vector[25] beta[D];
  vector[25] beta_sq[D];
  
  //Thetas for all prior returns, and combined returns
  vector[Q] theta;
  
  vector<lower=0>[N] sigma;
  
}

transformed parameters{
  
  matrix[N,T] epsilon;    // error term at time t
  //vector[N] weighted_err;
  
  vector[N] mu;
  
  for (n in 1:N)
    mu[n] <- alpha[ll[n]] + 
      row(covar, n) * beta[ll[n]] +
      row(covar_sq, n) * beta_sq[ll[n]];
  
  for (n in 1:N)
    for (t in 1:T) {
      epsilon[n,t] <- y_intra[n,t] - mu[n];
      for (q in 1:min(t-1,Q))
        epsilon[n,t] <- epsilon[n,t] - theta[q] * epsilon[n,t - q];
    }
  
  //weighted_err <- (y - y_hat) .* weights;
  
}

model {
  
  matrix[N,T] eta;
  
  // priors
  alpha ~ normal(0,2);
  
  for(d in 1:D){
    beta[d] ~ normal(0,2);
    beta_sq[d] ~ normal(0,2);
  }
  
  theta ~ normal(0,2);
  
  //sigma ~ cauchy(0,5);
  //sigma ~ normal(0,1) T[0,];
  
  for(n in 1:N)
    sigma[n] ~ cauchy(0,1) T[0,];
  
  // likelihood
  //weighted_err ~ normal(0,sigma);
  //weighted_err ~ student_t(2,0,sigma);
  
  for (n in 1:N)
    for (t in 1:T) {
      eta[n,t] <- mu[n];
      for (q in 1:min(t-1,Q))
        eta[n,t] <- eta[n,t] + theta[q] * epsilon[n,t - q];
    }
  
  //for(n in 1:N)
    //for (t in 1:T)
      //y_intra[n,t] ~ normal(eta[n,t],sigma[n]);
  
  for(n in 1:N)    
    row(y_intra,n) ~ normal(row(eta,n),sigma[n]);
}