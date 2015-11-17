data {
  int<lower=0> N;
  int<lower=0> D;
  vector [N] y;
  matrix [N, 25] covar;
  
  //intra-day returns
  matrix [N, 119] intra_day_1;
  vector [N] y_m2;
  vector [N] y_m1;
  
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
  matrix [N,4] y_m;
  matrix [N,4] ln_y_m;
  
  vector [N] y_m1_m2;
  vector [N] y_intra;
  
  matrix [N, 25] ln_covar;
  
  //log of 1+covars
  ln_covar <- log(covar+1);
  
  //average return over two days
  y_m1_m2 <- (y_m1 + 1) .* (y_m2 + 1);
  for (n in 1:N)
    y_m1_m2[n] <- pow(y_m1_m2[n], .5) - 1;
  
  //Return history matrix is:
    //Ret_MinusTwo, Ret_MinusTwo, Average over 1+2, Average of first half of intra-day
  y_temp1 <- append_col(y_m2, y_m1);
  y_temp2 <- append_col(y_temp1, y_m1_m2);
  
  //print("cols in y=", cols(y_m));
  
  //average return over first half of intra day
  //y_intra <- intra_day_1[,1] + 1;
  y_intra <- col(intra_day_1, 1) + 1;
  for (j in 2:119)
    y_intra <- y_intra .* (col(intra_day_1, j) + 1);
  
  //for (i in 1:N)
    //  for (j in 2:119)
      //    print("cols in y=", intra_day_1[i,j]);
  
  for (n in 1:N)
    y_intra[n] <- pow(y_intra[n], inv(119)) - 1;
  
  //y_m <- append_col(append_col(y_m1, y_m2), y_intra);
  y_m <- append_col(y_temp2, y_intra);
  
  //log of 1+returns
  ln_y_m <- log(y_m+1);
  
}

parameters {
  //intercept
  //vector [N] alpha;
  //real alpha;
  real alpha[D];
  
  //regression
  vector [25] beta;
  
  //Thetas for all prior returns, and combined returns
  vector[4] theta;
  
  real<lower=0> sigma;
  
}

transformed parameters{
  //matrix [N,2] covar;
  //matrix [N,3] y_m;
  vector [N] y_gr;
  
  //error terms
  matrix [N,2] epsilon;
  
  //convert to growth rates
  y_gr <- y + 1;
  
  epsilon <- append_col(y - y_m1, y - y_m2);
  
}

model {
  vector[N] y_hat;
  vector[N] weighted_err;
  
  for (n in 1:N)
    y_hat[n] <- alpha[ll[n]+1] + row(covar, n) * beta + row(y_m, n) * theta;

  weighted_err <- (y - y_hat) .* weights;
  
  // priors
  alpha ~ normal(0,10);
  beta ~ normal(0,2);
  
  theta ~ normal(1,2);
  
  //sigma ~ cauchy(0,5);
  sigma ~ normal(0,1) T[0,];
  
  // likelihood
  //weighted_err ~ normal(0,sigma);
  weighted_err ~ student_t(2,0,sigma);
  
  //y ~ normal(y_hat, sigma);
  
}