rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
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

intra <- test.full %>%
  select(Ret_2:Ret_120) %>%
  by_row(lift_vl(median), na.rm=TRUE, .labels=FALSE, .to='intra.median', .collate = "rows") %>%
  bind_cols(test.full %>% select(Id))

sample.submission <- sample.submission %>%
  separate(Id, c('Id2', 'Day'), remove=FALSE, convert=TRUE)

test <- sample.submission %>%
  left_join(intra, by=c('Id2'='Id'))

test.intra <- test %>%
  mutate(Predicted = ifelse(Day<=60, intra.median, 0)) %>%
  select(Id, Predicted)

file <- paste0("winton-intra_median", ".csv.gz")
write.csv(test.intra, gzfile(file), row.names=FALSE)


