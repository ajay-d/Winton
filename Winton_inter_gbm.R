rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(Matrix)
library(ggplot2)
library(xgboost)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train.full <- read_csv("data/train.csv.zip")
test.full <- read_csv("data/test_2.csv.zip")
sample.submission <- read_csv("data/sample_submission_2.csv.zip")

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
  mutate(return.intra = total.gr.intra^(1/n.returns)-1) %>%
  mutate(return.intra.day = (return.intra+1)^420-1)

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

summary(train.imp$Ret_MinusTwo)
summary(train.imp$Ret_MinusOne)

summary(train.imp$Ret_2)
summary(train.imp$Ret_119)

summary(train.imp$return.intra)
summary(train.imp$return.intra.day)

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
  mutate(return.intra = total.gr.intra^(1/n.returns)-1) %>%
  mutate(return.intra.day = (return.intra+1)^420-1)

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

# features_new <- test.imp %>%
#   select(matches("Feature")) %>%
#   replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
#                   Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
#                   Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
#   as.matrix()

test.data <- test.imp %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0)) %>%
  select(matches("Feature"), Ret_MinusTwo, Ret_MinusOne, total.gr.intra, n.returns.intra, return.intra, return.intra.day, level_8) %>%
  as.matrix()

#################################################################################
#' Sample Training data
train.sample <- train.imp %>%
  replace_na(list(Feature_1=0,Feature_2=0,Feature_3=0,Feature_4=0,Feature_5=0,Feature_6=0,Feature_7=0,Feature_8=0,Feature_9=0,Feature_10=0,
                  Feature_11=0,Feature_12=0,Feature_13=0,Feature_14=0,Feature_15=0,Feature_16=0,Feature_17=0,Feature_18=0,Feature_19=0,Feature_20=0,
                  Feature_21=0,Feature_22=0,Feature_23=0,Feature_24=0,Feature_25=0))

intra.ret <- train.sample %>%
  select(Ret_2:Ret_120) %>%
  as.matrix()

dim(intra.ret)

ind <- which(is.na(intra.ret), arr.ind=TRUE)
intra.ret[ind] <- train.sample[ind[,1],'return.intra']

sum(is.na(intra.ret))

intra.names <- train.imp %>%
  select(Ret_2:Ret_120) %>%
  names

x_intra <- intra.ret %>%
  as.data.frame() %>%
  setNames(intra.names)

y <- train.sample %>%
  select(Ret_PlusOne:Ret_PlusTwo) %>%
  as.matrix()

dim(y)

set.seed(12345)

train.data.model <- train.sample %>%
  select(Id, matches("Feature"), Ret_MinusTwo, Ret_MinusOne, total.gr.intra, n.returns.intra, return.intra, return.intra.day, level_8, Ret_PlusOne:Ret_PlusTwo) %>%
  sample_frac(.75) %>%
  arrange(Id)

setdiff(names(train.imp), names(test.full))

oos.test <- train.sample %>%
  anti_join(train.data.model %>%
              select(Id), by='Id') %>%
  select(Id, matches("Feature"), Ret_MinusTwo, Ret_MinusOne, total.gr.intra, n.returns.intra, return.intra, return.intra.day, level_8, Ret_PlusOne:Ret_PlusTwo) %>%
  arrange(Id)

train.data.watch <- train.data.model %>%
  sample_frac(.5)

train.data.watch.y <- train.data.watch %>%
  select(Ret_PlusOne:Ret_PlusTwo) %>%
  as.matrix

train.data.model <- train.data.model %>% 
  anti_join(train.data.watch %>%
              select(Id), by='Id') %>%
  arrange(Id)

train.data.model.y <- train.data.model %>%
  select(Ret_PlusOne:Ret_PlusTwo) %>%
  as.matrix

oos.test.y <- oos.test %>%
  select(Ret_PlusOne:Ret_PlusTwo) %>%
  as.matrix

train.data.model <- train.data.model %>% select(-Id) %>% as.matrix
oos.test <- oos.test %>% select(-Id) %>% as.matrix
train.data.watch <- train.data.watch %>% select(-Id) %>% as.matrix

dtrain <- xgb.DMatrix(data = train.data.model, label = train.data.model.y[,1])
dtest <- xgb.DMatrix(data = train.data.watch, label = train.data.watch.y[,1])

dtrain.all <- xgb.DMatrix(data = rbind(train.data.model, train.data.watch), 
                          label = rbind(train.data.model.y[,1], train.data.watch.y[,1]))

#dtest <- xgb.DMatrix(data = train.data.watch, label = train.data.watch.y)

reg.1 <- xgb.train(data=dtrain, max.depth=10, eta=.05, nround=200, 
                   eval.metric = "rmse",
                   nthread = 8, objective = "reg:linear")


watchlist <- list(train=dtrain, test=dtest)
##best
reg.2 <- xgb.train(data=dtrain, max.depth=20, eta=.05, nround=200, 
                   watchlist = watchlist,
                   eval.metric = "rmse",
                   nthread = 8, objective = "reg:linear")

reg.3 <- xgb.train(data=dtrain, max.depth=20, eta=.01, nround=500, 
                   watchlist = watchlist,
                   eval.metric = "rmse",
                   nthread = 8, objective = "reg:linear")

reg.4 <- xgb.train(data=dtrain, max.depth=50, eta=.01, nround=500, 
                   watchlist = watchlist,
                   eval.metric = "rmse",
                   nthread = 8, objective = "reg:linear")

reg.3 <- xgb.train(data=dtrain, max.depth=20, nround=500, 
                   eval.metric = "rmse",
                   nthread = 8, objective = "reg:linear")

pred <- predict(reg.4, oos.test)

pred <- predict(reg.3, oos.test)

pred <- predict(reg.2, oos.test)

pred.1 <- pred %>%
  cbind(oos.test.y[,1]) %>%
  as.data.frame() %>%
  setNames(c('pred', 'y')) %>%
  mutate(error = pred-y)

sum(abs(pred.1$error))

pred.test.1 <- predict(reg.2, test.data)

sum(abs(pred.1$error))

dtrain <- xgb.DMatrix(data = train.data, label = y[,2])
reg.2 <- xgb.train(data=dtrain, max.depth=20, eta=.5, nround=200, 
                   eval.metric = "merror", eval.metric = "mlogloss",
                   nthread = 8, objective = "reg:linear")

pred <- predict(reg.2, oos.test)
pred.test.2 <- predict(reg.2, test.data)

pred.2 <- pred %>%
  cbind(oos.test.y[,2]) %>%
  as.data.frame() %>%
  setNames(c('pred', 'y')) %>%
  mutate(error = pred-y)

sum(abs(pred.2$error))

pred.test <- cbind(test.imp$Id, pred.test.1, pred.test.2) %>%
  as.data.frame() %>%
  setNames(c('Id','61','62'))

summary(train.imp$Ret_MinusTwo)
summary(train.imp$Ret_MinusOne)

summary(pred.test$`62`)
summary(pred.test$`61`)

y.pred.inter <- pred.test %>%
  gather(Id.y, Predicted, -Id) %>%
  mutate(Id.merge = paste0(Id, "_", Id.y)) %>%
  select(Id=Id.merge, Predicted.inter = Predicted) %>%
  arrange(Id)

submission <- sample.submission %>%
  separate(Id, c('Id2', 'Day'), remove=FALSE, convert=TRUE) %>%
  left_join(y.pred.inter, by=c('Id'='Id'))

submission <- submission %>%
  mutate(Predicted = ifelse(Day>60, Predicted.inter, 0)) %>%
  select(Id, Predicted)

file <- paste0("winton-inter_gbm", ".csv.gz")
write.csv(submission, gzfile(file), row.names=FALSE)
