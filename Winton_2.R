rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train.full <- read_csv("data/train.csv.zip")
test.full <- read_csv("data/test.csv.zip")
sample_submission <- read_csv("data/sample_submission.csv.zip")

names(train.full)

features <- train.full %>%
  select(matches("Feature"))

f.cor <- cor(features, use="pairwise.complete.obs")

f.cor <- apply(f.cor, 2, function (x) ifelse(x==1,0,x))
f.cor.max <- apply(f.cor, 1, max)
f.cor.max

# impute missing features
train.imp <- train.full %>%
  mutate(Feature_3 = Feature_3 %||% Feature_4,
         Feature_4 = Feature_4 %||% Feature_3,
         Feature_6 = Feature_6 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_6,
         Feature_18 = Feature_18 %||% Feature_21,
         Feature_21 = Feature_21 %||% Feature_18)


train.sample <- train.imp %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
  sample_n(5000)

intra.ret <- train.sample %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

intra.ret[is.na(intra.ret)] <- 0

sum(is.na(intra.ret))

features <- train.sample %>%
  select(matches("Feature")) %>%
  as.matrix()

train.full %>%
  mutate(na.var = ifelse(is.na(Ret_MinusTwo), 1, 0)) %>%
  count(na.var)

dat <- list('N' = dim(train.sample)[[1]],
            'covar' = features,
            'intra_day_1' = intra.ret,
            "y_m2" = train.sample$Ret_MinusTwo,
            "y_m1" = train.sample$Ret_MinusOne,
            'y' = train.sample$Ret_PlusOne,
            'weights' = train.sample$Weight_Daily)

fit <- stan('stan_model_5.stan',  
            model_name = "Stan1", 
            iter=1500, warmup=500,
            thin=2, chains=1, seed=252014,
            data = dat)

print(fit, pars=c("beta", "theta", "sigma"), probs=c(0.5, 0.75, 0.95))
traceplot(fit, pars=c("beta", "theta", 'sigma'))
