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
                            -1 == sign(return.intra), 8, level_8))

load("Stan_inter_0beta.RData")
inter_0beta <- extract(fit, pars=c("alpha", "theta"), permuted = TRUE)

names(inter_0beta)
dim(inter_0beta$alpha)
dim(inter_0beta$theta)

alpha <- apply(inter_0beta$alpha, 2:3, median)
median(inter_0beta$alpha[,1,1])
median(inter_0beta$alpha[,7,2])

theta <- apply(inter_0beta$theta, 2:4, median)

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

