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

################################################################################
#' Prepare Training data

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

setdiff(names(train.imp), names(train.full))

################################################################################
#' Prepare Test data for forecasting predictions

# impute missing features
test.imp <- test.full %>%
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

intra.ret <- test.full %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

intra.ret <- intra.ret+1
prod <- apply(intra.ret, 1, prod, na.rm=TRUE)

n.returns <- apply(!is.na(intra.ret), 1, sum)

df <- data_frame(total.gr.intra = prod,
                 n.returns.intra = n.returns) %>%
  mutate(return.intra = total.gr.intra^(1/n.returns)-1)

test.imp <- cbind(test.imp, df) %>%
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

x_intra_new <- test.imp %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

dim(x_intra_new)

ind <- which(is.na(x_intra_new), arr.ind=TRUE)
x_intra_new[ind] <- test.imp[ind[,1],'return.intra']

sum(is.na(x_intra_new))
tail(x_intra_new[ind])
tail(test.imp[ind[,1],'return.intra'])

# for(i in 1:nrow(x_intra_new)) {
#   for(j in 1:ncol(x_intra_new)) {
#       if(is.na(x_intra_new[i,j]))
#         x_intra_new[i,j] <- test.imp[[i,'return.intra']]
#     }
#   }

sum(is.na(x_intra_new))

features_new <- test.imp %>%
  select(matches("Feature")) %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
  as.matrix()


#################################################################################
#' Sample Training data
train.sample <- train.imp %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
  sample_n(5000)

intra.ret <- train.sample %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

dim(intra.ret)

# for(i in 1:nrow(intra.ret)) {
#   for(j in 1:ncol(intra.ret)) {
#       if(is.na(intra.ret[i,j]))
#         intra.ret[i,j] <- train.sample[[i,'return.intra']]
#     }
#   }

#intra.ret[is.na(intra.ret)] <- 0

ind <- which(is.na(intra.ret), arr.ind=TRUE)
intra.ret[ind] <- train.sample[ind[,1],'return.intra']

sum(is.na(intra.ret))

features <- train.sample %>%
  select(matches("Feature")) %>%
  as.matrix()

setdiff(names(train.imp), names(test.full))

y <- train.sample %>%
  select(Ret_121:Ret_180) %>%
  as.matrix()

dim(y)

ggplot(train.sample) + geom_histogram(aes(Weight_Intraday))
ggplot(train.sample) + geom_histogram(aes(x=Weight_Intraday, fill=as.factor(level_8)))

##########

dat <- list('T' = dim(intra.ret)[[2]], #time periods we have returns for
            'N' = dim(train.sample)[[1]], #number of obs
            'Q' = 3, #number of lags for MA
            'D' = length(sort(unique(train.sample$level_8))), #number of stratification levels
            'll' = train.sample$level_8, #level indicator
            'covar' = features,
            'x_intra' = intra.ret,
            'y' = y, #training 
            'weights' = train.sample$Weight_Intraday,
            'N_new' = 60000,
            'x_intra_new' = x_intra_new, #new covars for predictions
            'covar_new' = features_new, #new returns for predictions
            'll_new' = test.imp$level_8 #level of new obs
            )

fit <- stan('stan_intra_2beta.stan',
            model_name = "Stan_intra", 
            iter=3000, warmup=2000,
            thin=2, chains=5, seed=252014,
            data = dat)

print(fit, pars=c("alpha", "beta", 'mu'), probs=c(0.5, 0.75, 0.95))
print(fit, pars=c("theta", 'sigma', 'epsilon'), probs=c(0.5, 0.75, 0.95))
traceplot(fit, pars=c("alpha", "theta"))
