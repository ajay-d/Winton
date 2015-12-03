data {
  
  int<lower=0> N; //training obs
  int<lower=0> N_new; //test obs
  
  int<lower=0> D; //levels of segmentation
  
  matrix [N, 25] covar;
  vector [N] x_m2;
  vector [N] x_m1;
  //combo intra-day returns and two prior day returns
  vector [N] x_intra;
  
  //Two days to forecast
  matrix [N, 2] y;
  
  //group (level)
  //vector [N] ll;
  int ll[N];
  
  vector [N] weights;
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
  //real alpha;
  real alpha[D, 2];
  
  //regression on features
  //2 for the two forecast days
  matrix [25, 2] beta[D];
  matrix [25, 2] beta_sq[D];
  
  //Thetas for all prior returns, and combined returns
  //2 for the two forecast days
  //matrix [Q, 2] theta_m2[D];
  //matrix [Q, 2] theta_m1[D];
  //matrix [Q, 2] theta_intra[D];
  
  real theta_m2[D, 2];
  real theta_m1[D, 2];
  real theta_intra[D, 2];
  
  real sigma_1;
  real sigma_2;
  
  //real<lower=0> sd_1;
  //real<lower=0> sd_2;
  
}

transformed parameters{
  
  matrix [N, 2] weighted_err;
  matrix [N, 2] epsilon;    // error
  
  vector [N] y_hat_P1;
  vector [N] y_hat_P2;
  
  vector [N] norm;
  
  for (i in 1:N) {
    y_hat_P1[i] <- alpha[ll[i], 1] + 
      row(covar, i) * col(beta[ll[i]], 1) +
      row(covar_sq, i) * col(beta_sq[ll[i]], 1) +
      x_m2[i] * theta_m2[ll[i], 1] +
      x_m1[i] * theta_m1[ll[i], 1] +
      x_intra[i] * theta_intra[ll[i], 1] + 
      sigma_1;
    
    y_hat_P2[i] <- alpha[ll[i], 2] + 
      row(covar, i) * col(beta[ll[i]], 2) +
      row(covar_sq, i) * col(beta_sq[ll[i]], 2) +
      x_m2[i] * theta_m2[ll[i], 2] +
      x_m1[i] * theta_m1[ll[i], 2] +
      x_intra[i] * theta_intra[ll[i], 2] + 
      sigma_2;
  }
      
  for (n in 1:N)
    for (t in 1:2)
      epsilon[n,t] <- y[n,t] - append_col(y_hat_P1, y_hat_P2)[n,t]; //error for each forcast
    
  norm <- weights ./ (sum(weights)/rows(weights));
  //abs??
  weighted_err <- epsilon .* rep_matrix(norm, 2);
    
}

model {

  // priors
  
  //sd_1 ~ normal(0,2.5) T[0,];
  //sd_2 ~ cauchy(0,2.5) T[0,];
  
  //sigma_1 ~ normal(0,sd_1);
  //sigma_2 ~ normal(0,.1);
  
  //real sigma;
  //sigma ~ cauchy(0,2.5);
  
  
  for(d in 1:D)
    for(i in 1:2) {
      
      alpha[d,i] ~ normal(0,2);
      theta_m2[d,i] ~ normal(0,2);
      theta_m1[d,i] ~ normal(0,2);
      theta_intra[d,i] ~ normal(0,2);
      
      //col(theta_m2[d],i) ~ normal(0,2);
      //col(theta_m1[d],i) ~ normal(0,2);
      //col(theta_intra[d],i) ~ normal(0,2);
      
      col(beta[d],i) ~ normal(0,2);
      col(beta_sq[d],i) ~ normal(0,2);
    }
  
  //increment_log_prob(-sum(weighted_err));
  //for(i in 1:2)
    //col(weighted_err,i) ~ normal(0,2);
  
  for (n in 1:N)
    weighted_err[n] ~ normal(0,2);
  
}
