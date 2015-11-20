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

train.full <- read_csv("data/train.csv.zip")
test.full <- read_csv("data/test.csv.zip")

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
         Feature_21 = Feature_21 %||% Feature_18) %>%
  #used for multi-level
  #4 level: ++, --, +-, -+
  mutate(level_2 = sign(Ret_MinusOne) == sign(Ret_MinusTwo),
         level_4 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & 1 == sign(Ret_MinusTwo), 1, 0),
         level_4 = ifelse(sign(Ret_MinusOne) == sign(Ret_MinusTwo) & -1 == sign(Ret_MinusTwo), 2, level_4),
         level_4 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & 1 == sign(Ret_MinusTwo), 3, level_4),
         level_4 = ifelse(sign(Ret_MinusOne) != sign(Ret_MinusTwo) & -1 == sign(Ret_MinusTwo), 4, level_4))

intra.ret <- train.full %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

intra.ret <- intra.ret+1
prod <- apply(intra.ret, 1, prod, na.rm=TRUE)

n.returns <- apply(!is.na(intra.ret), 1, sum)

df <- data_frame(total.gr.intra = prod,
                 n.returns.intra = n.returns) %>%
  mutate(return.intra = total.gr.intra^(1/n.returns)-1)

train.imp <- cbind(train.imp, df) %>%
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
                            -1 == sign(return.intra), 8, level_8))

#train.imp %>% filter(Ret_MinusOne < 0, Ret_MinusTwo <0, return.intra <0)

table(train.imp$level_2, useNA = 'ifany')
table(train.imp$level_4, useNA = 'ifany')
table(train.imp$level_8, useNA = 'ifany')

plot(density(rnorm(1000,0,2)))
plot(density(rt(1000,2)))

quantile(rt(1000,1))
quantile(rnorm(1000,0,2))

train.sample <- train.imp %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
  sample_n(5000)

intra.ret <- train.sample %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

dim(intra.ret)

for(i in 1:nrow(intra.ret)) {
  for(j in 1:ncol(intra.ret)) {
      if(is.na(intra.ret[i,j]))
        intra.ret[i,j] <- train.sample[[i,'return.intra']]
    }
  }

#intra.ret[is.na(intra.ret)] <- 0

sum(is.na(intra.ret))

features <- train.sample %>%
  select(matches("Feature")) %>%
  as.matrix()

train.full %>%
  mutate(na.var = ifelse(is.na(Ret_MinusTwo), 1, 0)) %>%
  count(na.var)

ggplot(train.sample) + geom_histogram(aes(Weight_Intraday))
ggplot(train.sample) + geom_histogram(aes(x=Weight_Intraday, fill=as.factor(level_8)))

##########

dat <- list('Q' = 3, #number of lags for MA
            'D' = length(sort(unique(train.sample$level_8))),
            'll' = train.sample$level_8,
            'N' = dim(train.sample)[[1]], #obs
            'T' = dim(intra.ret)[[2]], #time periods
            'covar' = features,
            'y_intra' = intra.ret,
            #'y' = train.sample$Ret_120,
            'y' = train.sample$Ret_121,
            'weights' = train.sample$Weight_Intraday)

fit <- stan('stan_intra_2beta.stan',
            model_name = "Stan_intra", 
            iter=300, warmup=200,
            thin=2, chains=1, seed=252014,
            data = dat)

print(fit, pars=c("beta", "theta", 'sigma', 'epsilon', 'gamma'), probs=c(0.5, 0.75, 0.95))
traceplot(fit, pars=c("beta", "theta", 'sigma', 'epsilon', 'gamma'))
