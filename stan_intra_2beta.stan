data {
  
  int<lower=0> T; //time periods
  int<lower=0> N; //training obs
  //int<lower=0> N_new; //test obs
  
  int<lower=0> D; //levels of segmentation
  
  int<lower=0> Q; //lags
  
  matrix[N, 60] y;
  matrix[N, 25] covar;
  
  //intra-day returns
  matrix[N, T] x_intra;
  
  //group (level)
  //vector [N] ll;
  int ll[N];
  
  vector[N] weights;
  
  //one-dimensional array of size N containing real values
  //real y[N];
  //real y_hat[N];
  //vector[N] y_hat;
  
  //intra-day returns for prediction
//   matrix[N_new, T] x_intra_new;
//   matrix[N_new, 25] covar_new;
//   int ll_new[N_new];
  
  //Add non-linear effects from both Features and returns
  //y = A + B*feat + T*returns + B*f(feat) + T*f(returns) / feat^2,  ln(returns)
  //ln(y) = A + B*feat + T*returns
}

transformed data{
  
  matrix[N, 25] covar_sq;
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
  real alpha[D];
  
  //real epsilon[D];
  //vector [N] epsilon;
  //real epsilon;
  
  //row_vector<lower=0>[60] sigma;
  
  //regression on features
  //vector[25] beta[D];
  //vector[25] beta_sq[D];
  matrix[25, 60] beta[D];
  matrix[25, 60] beta_sq[D];
  
  //Thetas for all prior returns, and combined returns
  //vector[Q] theta_1[D];
  matrix[Q, 60] theta_1[D];
  
  vector [60] sigma_1;
  
  real<lower=0> sd_1;
  real<lower=0> sd_2;
  
  //Thetas on error terms
  //matrix[N, 60] theta_2;
  
}

transformed parameters{
  
  matrix[N, 60] epsilon;    // error term at time t
  //vector[N] weighted_err;
  
  matrix[N, 60] weighted_err;
  //vector[N] mu;
  matrix[N, 60] mu;
  matrix[N, Q] x_returns;
  //matrix[N, Q] x_returns_new;
  
  vector[N] norm;
  
  for (n in 1:N)
    for (q in 1:Q) {
      x_returns[n,q] <- x_intra[n,(T-q+1)];
	    //x_returns_new[n,q] <- x_intra_new[n,(T-q+1)];
    } 

//  for (n in 1:N)
//    mu[n] <- alpha[ll[n]] + 
//      row(covar, n) * beta[ll[n]] +
//      row(covar_sq, n) * beta_sq[ll[n]] +
//      row(x_returns, n) * theta_1[ll[n]];

  for (i in 1:N)
    for (j in 1:60)
      mu[i,j] <- alpha[ll[i]] + 
        row(covar, i) * col(beta[ll[i]], j) +
        row(covar_sq, i) * col(beta_sq[ll[i]], j) +
        row(x_returns, i) * col(theta_1[ll[i]], j) + 
        sigma_1[j];
  
  for (n in 1:N)
    for (t in 1:60)
      epsilon[n,t] <- y[n,t] - mu[n,t]; //error for each forcast
  
  norm <- weights ./ (sum(weights)/rows(weights));
  
  //abs??
  weighted_err <- epsilon .* rep_matrix(norm, 60);
  
}

model {
  
  //matrix[N, 60] y_hat;
  
//   for (j in 1:60){
//     col(mu,j) ~ normal(0,2);
//     
//     col(epsilon,j) ~ normal(0,1);
// 	
// 	//Should increase through time?
//   col(theta_2,j) ~ normal(0,2);
//   }
  
  sd_1 ~ normal(0,2.5) T[0,];
  sd_2 ~ cauchy(0,2.5) T[0,];
  
  for(i in 1:60)
    sigma_1[i] ~ normal(0,sd_1);
    //sigma_1[i] ~ normal(0,.1);
  
  // priors
  alpha ~ normal(0,2);
  
  for(d in 1:D)
    for (j in 1:60){
      col(beta[d],j) ~ normal(0,2);
      col(beta_sq[d],j) ~ normal(0,2);
	  
	  col(theta_1[d],j) ~ normal(0,2);
    }

  //increment_log_prob(-sum(weighted_err));
  for(i in 1:60)
    col(weighted_err,i) ~ normal(0,sd_2);
  
  //for (n in 1:N)
    //for (t in 1:60)
      //y_hat[n,t] <- mu[n,t] + theta_2[n,t] * epsilon[n,t];
  
  //for(t in 1:60)
    //col(y,t) ~ normal(col(y_hat,t), sigma[t]);
    //col(y,t) ~ normal(col(mu,t), sigma[t]);
  
}

// generated quantities {
//   
//   matrix[N_new, 60] mu_new;
//   matrix[N_new, 60] y_pred;
//   
//   for (i in 1:N_new)
//     for (j in 1:60)
//       mu_new[i,j] <- alpha[ll_new[i]] +
//         row(covar_new, i) * col(beta[ll_new[i]], j);
//         //row(covar_new_sq, i) * col(beta_sq[ll_new[i]], j) +
//         //row(x_returns_new, i) * col(theta_1[ll_new[i]], j);
//   
//   print("cols in x=", cols(x_returns_new));
//   print("rows in x=", rows(x_returns_new));
//   
//   for (i in 1:N_new)
//     for (j in 1:60)
//       y_pred[i,j] <- normal_rng(mu_new[i,j], sigma[j]);
// 
//   //for(t in 1:60)
//   //  col(y_pred,t) <- normal_rng(col(mu_new,t), sigma[t]);
//   
// }