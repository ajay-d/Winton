rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(rstan)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

test.full <- read_csv("data/test.csv.zip")
sample.submission <- read_csv("data/sample_submission.csv.zip")

intra.ret <- test.full %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

intra.ret <- intra.ret+1
prod <- apply(intra.ret, 1, prod, na.rm=TRUE)

n.returns <- apply(!is.na(intra.ret), 1, sum)

df <- data_frame(total.gr.intra = prod,
                 n.returns.intra = n.returns) %>%
  mutate(return.intra = total.gr.intra^(1/n.returns)-1)

test.imp <- cbind(test.full, df) %>%
  mutate(Feature_3 = Feature_3 %||% Feature_4,
         Feature_4 = Feature_4 %||% Feature_3,
         Feature_6 = Feature_6 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_6,
         Feature_18 = Feature_18 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_18) %>%
  #8 level: +++, ---, ++-, --+, -++, +--, +-+, -+-
  mutate(level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            1 == sign(return.intra), 1, 0),
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            -1 == sign(return.intra), 2, level_8),
         # ++-
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            -1 == sign(return.intra), 3, level_8),
         # --+
         level_8 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            1 == sign(return.intra), 4, level_8),
         # -++
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            1 == sign(return.intra), 5, level_8),
         # +--
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) == sign(return.intra) & 
                            -1 == sign(return.intra), 6, level_8),
         # +-+
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            1 == sign(return.intra), 7, level_8),
         # -+-
         level_8 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 
                            sign(Ret_MinusOne) != sign(return.intra) & 
                            -1 == sign(return.intra), 8, level_8)) %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0))

load("Stan_inter_0beta.RData")
inter_0beta <- extract(fit, pars=c("alpha", "theta"), permuted = TRUE)

names(inter_0beta)
dim(inter_0beta$alpha)
dim(inter_0beta$theta)

alpha <- apply(inter_0beta$alpha, 2:3, median)
median(inter_0beta$alpha[,1,1])
median(inter_0beta$alpha[,7,2])

theta <- apply(inter_0beta$theta, 2:4, median)

#####0 Beta ####
#inter day returns (0 beta):
y.pred.intra <- NULL
for(d in sort(unique(test.imp$level_8))) {
  
  t.d <- filter(test.imp, level_8==d)
  x.d <- t.d %>% select(Ret_MinusTwo, Ret_MinusOne, return.intra)
  
  theta.d <- theta[d,,]
  alpha.d <- alpha[d,]
  
  y.pred.d <- as.matrix(x.d) %*% as.matrix(t(theta.d)) + alpha.d
  
  y.pred.d <- t.d %>% 
    select(Id) %>%
    bind_cols(as.data.frame(y.pred.d)) %>%
    setNames(c('Id','61','62'))
  
  y.pred.intra <- bind_rows(y.pred.intra, y.pred.d)
    
}

y.pred.intra <- y.pred.intra %>%
  gather(Id.y, Predicted, -Id) %>%
  mutate(Id.merge = paste0(Id, "_", Id.y)) %>%
  select(Id=Id.merge, Predicted) %>%
  arrange(Id)


load("Stan_inter_1beta.RData")
inter_1beta <- extract(fit, pars=c("alpha", "theta", "beta"), permuted = TRUE)

alpha <- apply(inter_1beta$alpha, 2:3, median)
theta <- apply(inter_1beta$theta, 2:4, median)
beta <- apply(inter_1beta$beta, 2:4, median)

#####1 Beta ####
#inter day returns (1 beta):
y.pred.intra <- NULL
for(d in sort(unique(test.imp$level_8))) {
  
  t.d <- filter(test.imp, level_8==d)
  x.d <- t.d %>% select(Ret_MinusTwo, Ret_MinusOne, return.intra)
  covar.d <- t.d %>% select(matches("Feature"))
  
  theta.d <- theta[d,,]
  alpha.d <- alpha[d,]
  #beta.d <- beta[d,,]
  
  y.pred.d <- as.matrix(x.d) %*% as.matrix(t(theta.d)) + alpha.d
  
  y.pred.d <- t.d %>% 
    select(Id) %>%
    bind_cols(as.data.frame(y.pred.d)) %>%
    setNames(c('Id','61','62'))
  
  y.pred.intra <- bind_rows(y.pred.intra, y.pred.d)
    
}

y.pred.intra <- y.pred.intra %>%
  gather(Id.y, Predicted, -Id) %>%
  mutate(Id.merge = paste0(Id, "_", Id.y)) %>%
  select(Id=Id.merge, Predicted) %>%
  arrange(Id)
