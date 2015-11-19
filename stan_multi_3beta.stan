data {
  int<lower=0> N;
  int<lower=0> D;
  vector [N] y;
  matrix [N, 25] covar;
  
  //intra-day returns
  matrix [N, 119] intra_day_1;
  vector [N] y_m2;
  vector [N] y_m1;
  
  //calculated returns for first half of day
  vector [N] y_intra;
  
  //group (level)
  //vector [N] ll;
  int ll[N];
  
  vector [N] weights;
  
  //one-dimensional array of size N containing real values
  //real y[N];
  //real y_hat[N];
  //vector[N] y_hat;
  
  //Add non-linear effects from both Features and returns
  //y = A + B*feat + T*returns + B*f(feat) + T*f(returns) / feat^2,  ln(returns)
  //ln(y) = A + B*feat + T*returns
}

transformed data{
  
  matrix [N,2] y_temp1;
  matrix [N,3] y_temp2;
  
  matrix [N,5] y_m;
  matrix [N,5] y_m_ln;
  
  vector [N] y_m1_m2;
  //vector [N] y_intra;
  
  matrix [N, 25] covar_ln;
  matrix [N, 25] covar_sq;
  
  //log of 1+covars
  covar_ln <- log(covar+1);
  
  for (i in 1:N)
    for (j in 1:25)
      covar_sq[i,j] <- pow(covar[i,j], 2);
  
  //average return over two days
  y_m1_m2 <- (y_m1 + 1) .* (y_m2 + 1);
  for (n in 1:N)
    y_m1_m2[n] <- pow(y_m1_m2[n], .5) - 1;
  
  //Return history matrix is:
  //Ret_MinusTwo, Ret_MinusTwo, Average over 1+2, Average of first half of intra-day, Average of 2.5/3 days
  y_temp1 <- append_col(y_m2, y_m1);
  y_temp2 <- append_col(y_temp1, y_m1_m2);

  //average return over three days
  y_m1_m2 <- (y_m1 + 1) .* (y_m2 + 1) .* (y_intra + 1);
  for (n in 1:N)
    y_m1_m2[n] <- pow(y_m1_m2[n], inv(3)) - 1;
  
  y_m <- append_col(append_col(y_temp2, y_intra), y_m1_m2);
  
  //log of 1+returns
  y_m_ln <- log(y_m+1);
  
}

parameters {
  //intercept
  //vector [N] alpha;
  //real alpha;
  real alpha[D];
  
  //real epsilon[D];
  vector [N] epsilon;
  //real epsilon;
  
  //regression
  vector [25] beta[D];
  vector [25] beta_ln[D];
  vector [25] beta_sq[D];
  
  //Thetas for all prior returns, and combined returns
  vector[5] theta[D];
  vector[5] theta_ln[D];
  
  real<lower=0> sigma;
  
}

transformed parameters{
  
  vector[N] y_hat;
  vector[N] weighted_err;
  
  for (n in 1:N)
    y_hat[n] <- alpha[ll[n]] + 
                row(covar, n) * beta[ll[n]] + 
                //row(covar_ln, n) * beta_ln[ll[n]] + 
                row(covar_sq, n) * beta_sq[ll[n]] + 
                row(y_m, n) * theta[ll[n]] + 
                row(y_m_ln, n) * theta_ln[ll[n]] + 
                epsilon[n];
  
  weighted_err <- (y - y_hat) .* weights;
  
}

model {
  
  // priors
  alpha ~ normal(0,2);
  epsilon ~ normal(0,.5);
  
  for(d in 1:D){
    beta[d] ~ normal(0,2);
    beta_ln[d] ~ normal(0,2);
    beta_sq[d] ~ normal(0,2);
  }
  
  for(d in 1:D){
    theta[d] ~ normal(0,2);
    theta_ln[d] ~ normal(0,2);
  }
  
  //sigma ~ cauchy(0,5);
  sigma ~ normal(0,1) T[0,];
  
  // likelihood
  weighted_err ~ normal(0,sigma);
  //weighted_err ~ student_t(2,0,sigma);
  
  //y ~ normal(y_hat, sigma);
  
}