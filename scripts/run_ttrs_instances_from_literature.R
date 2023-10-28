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

df <- read_csv("../results/run_ttrs_instances_from_literature.csv")

tmp <- df %>%
    group_by(n) %>%
    summarize(mean_time = mean(time))

print(tmp)
