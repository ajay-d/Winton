library(readr)
library(dplyr)
library(tidyr)
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

train.full <- read_csv("data/train.csv.zip")
test.full <- read_csv("data/test.csv.zip")
sample_submission <- read_csv("data/sample_submission.csv.zip")

names(train.full)

ggplot(train.full) + geom_histogram(aes(Ret_MinusTwo), binwidth=.01)
ggplot(train.full) + geom_histogram(aes(Ret_MinusOne), binwidth=.01)

summary(train.full$Ret_MinusTwo)
summary(train.full$Ret_MinusOne)

ggplot(train.full) + geom_histogram(aes(Feature_2), binwidth=.01)
ggplot(train.full) + geom_histogram(aes(Feature_3), binwidth=.01)
ggplot(train.full) + geom_histogram(aes(Feature_4), binwidth=.01)