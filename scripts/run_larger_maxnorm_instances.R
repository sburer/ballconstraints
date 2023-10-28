library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(scales)
library(xtable)

# Set tolerances

tol_rel_gap <- 1.0e-4
tol_eval_ratio <- 10^4

################################################################################

df <- read_csv("../results/run_larger_maxnorm_instances.csv")

tmp <- df %>%
    group_by(n, m) %>%
    summarize(mean_time = mean(time)) %>%
    pivot_wider(names_from = m, values_from = mean_time) %>%
    mutate(n = as.integer(n)) %>%
    xtable(digits = 1) %>%
    print(include.rownames=FALSE)

#fit <- lm(data = df, time ~ poly(n, m, degree = 3))
#print(summary(fit))
