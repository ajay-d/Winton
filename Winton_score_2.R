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

test.full <- read_csv("data/test_2.csv.zip")
sample.submission <- read_csv("data/sample_submission_2.csv.zip")

intra.ret <- test.full %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

intra.ret <- intra.ret+1
prod <- apply(intra.ret, 1, prod, na.rm=TRUE)

n.returns <- apply(!is.na(intra.ret), 1, sum)

df <- data_frame(total.gr.intra = prod,
                 n.returns.intra = n.returns) %>%
  mutate(return.intra = total.gr.intra^(1/n.returns)-1) %>%
  mutate(return.intra.day = (return.intra+1)^420-1)

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

#Replace any returns with intra calc:
x_intra_new <- test.imp %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

dim(x_intra_new)

ind <- which(is.na(x_intra_new), arr.ind=TRUE)
x_intra_new[ind] <- test.imp[ind[,1],'return.intra']

sum(is.na(x_intra_new))
tail(x_intra_new[ind])
tail(test.imp[ind[,1],'return.intra'])

intra.names <- test.imp %>%
  select(Ret_2:Ret_120) %>%
  names

x_intra_new <- x_intra_new %>%
  as.data.frame() %>%
  setNames(intra.names)

test.imp <- test.imp %>%
  select(-(Ret_2:Ret_120)) %>%
  bind_cols(x_intra_new)

load("Stan_inter_1beta.RData")
inter_1beta <- extract(fit, pars=c("alpha", "theta", "beta"), permuted = TRUE)

cat(get_stancode(fit))

alpha <- apply(inter_1beta$alpha, 2:3, median)
theta <- apply(inter_1beta$theta, 2:4, median)
beta <- apply(inter_1beta$beta, 2:4, median)

#####1 Beta ####
#inter day returns (1 beta):
y.pred.inter <- NULL
for(d in sort(unique(test.imp$level_8))) {
  
  t.d <- filter(test.imp, level_8==d)
  x.d <- t.d %>% select(Ret_MinusTwo, Ret_MinusOne, return.intra)
  covar.d <- t.d %>% select(matches("Feature"))
  
  theta.d <- theta[d,,]
  alpha.d <- alpha[d,]
  beta.d <- beta[d,,]
  
  y.pred.d <- as.matrix(x.d) %*% as.matrix(t(theta.d)) + alpha.d +
    as.matrix(covar.d) %*% as.matrix(t(beta.d))
  
  y.pred.d <- t.d %>% 
    select(Id) %>%
    bind_cols(as.data.frame(y.pred.d)) %>%
    setNames(c('Id','61','62'))
  
  y.pred.inter <- bind_rows(y.pred.inter, y.pred.d)
    
}

y.pred.inter <- y.pred.inter %>%
  gather(Id.y, Predicted, -Id) %>%
  mutate(Id.merge = paste0(Id, "_", Id.y)) %>%
  select(Id=Id.merge, Predicted) %>%
  arrange(Id)


load("Stan_intra_0beta_3Q_500.RData")
intra_0beta <- extract(fit, pars=c("alpha", "theta"), permuted = TRUE)

alpha <- apply(intra_0beta$alpha, 2:3, median)
theta <- apply(intra_0beta$theta, 2:4, median)

#####0 Beta ####
#intra day returns (0 beta):
y.pred.intra <- NULL
for(d in sort(unique(test.imp$level_8))) {
  
  t.d <- filter(test.imp, level_8==d)
  x.d <- t.d %>% select(matches('Ret_'))
  
  theta.d <- theta[d,,]
  alpha.d <- alpha[d,]
  
  #Q: number of lags used
  dim(theta.d)[[2]]
  old.lag <- 120 - dim(theta.d)[[2]] + 1
  
  x.d <- x.d %>% select(num_range('Ret_', old.lag:120))
  
  y.pred.d <- as.matrix(x.d) %*% as.matrix(t(theta.d)) + alpha.d
  
  y.pred.d <- t.d %>% 
    select(Id) %>%
    bind_cols(as.data.frame(y.pred.d)) %>%
    setNames(c('Id', 1:60))
  
  y.pred.intra <- bind_rows(y.pred.intra, y.pred.d)
    
}

y.pred.intra <- y.pred.intra %>%
  gather(Id.y, Predicted, -Id) %>%
  mutate(Id.merge = paste0(Id, "_", Id.y)) %>%
  select(Id=Id.merge, Predicted)

#try intras all 0
y.pred.intra <- y.pred.intra %>%
  mutate(Predicted=0)

y.pred <- y.pred.intra %>%
  bind_rows(y.pred.inter) %>%
  arrange(Id)

#Check names and for any NAs
sum(is.na(y.pred[,2]))
#sample.submission %>% anti_join(y.pred, by='Id')

file <- paste0("winton-inter_1beta-intra_0", ".csv.gz")
write.csv(y.pred, gzfile(file), row.names=FALSE)

summary(y.pred$Predicted)
ggplot(y.pred) + geom_histogram(aes(Predicted), binwidth=.01)

###################################################################

simple <- test.imp %>%
  select(Id, Ret_MinusTwo, Ret_MinusOne, return.intra, return.intra.day)

sample.submission <- sample.submission %>%
  separate(Id, c('Id2', 'Day'), remove=FALSE, convert=TRUE)

ggplot(test.imp) + geom_histogram(aes(Ret_MinusTwo), binwidth=.01)
ggplot(test.imp) + geom_histogram(aes(Ret_MinusOne), binwidth=.01)

ggplot(test.imp) + geom_histogram(aes(Ret_2), binwidth=.01)
ggplot(test.imp) + geom_histogram(aes(Ret_3), binwidth=.01)
ggplot(test.imp) + geom_histogram(aes(Ret_119), binwidth=.01)
ggplot(test.imp) + geom_histogram(aes(Ret_120), binwidth=.01)

summary(test.imp$Ret_MinusTwo)
summary(test.imp$Ret_MinusOne)
summary(test.imp$Ret_2)

test <- sample.submission %>%
  left_join(simple, by=c('Id2'='Id'))

test.intra <- test %>%
  mutate(Predicted = ifelse(Day<=60, return.intra, 0)) %>%
  select(Id, Predicted)

file <- paste0("winton-intra_avg", ".csv.gz")
write.csv(test.intra, gzfile(file), row.names=FALSE)

test.intra2 <- test %>%
  mutate(Predicted = ifelse(Day==1, return.intra, 0)) %>%
  select(Id, Predicted)

file <- paste0("winton-intra2_avg", ".csv.gz")
write.csv(test.intra2, gzfile(file), row.names=FALSE)

test.inter <- test %>%
  mutate(Predicted = ifelse(Day>60, return.intra.day, 0)) %>%
  select(Id, Predicted)

file <- paste0("winton-inter_avg", ".csv.gz")
write.csv(test.inter, gzfile(file), row.names=FALSE)
