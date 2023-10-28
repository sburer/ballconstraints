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

#
# linear
#

# Read data

# df <- read_csv("../python/csv/test_instances.csv") %>%
#   filter(type == "linear")
# 
# # Prepare joint "number solved" and "total timings" table
# 
# df %>%
#   mutate(solved = (rel_gap < tol_rel_gap & eval_ratio > tol_eval_ratio)) %>%
#   mutate(m = 2) %>%
#   group_by(relaxation, n, m) %>%
#   summarize(num_instances = n(), num_solved = sum(solved), total_time = sum(time)) %>%
#   pivot_wider(names_from = relaxation, values_from = c("num_solved", "total_time")) %>%
#   mutate(num_instances = comma(num_instances), num_solved_beta = comma(num_solved_beta), n = as.integer(n), m = as.integer(m)) %>%
#   select(n, m, num_instances, num_solved_kron, num_solved_beta, total_time_shor, total_time_kron, total_time_beta) %>%
#   xtable(digits = 1) %>%
#   print(include.rownames=FALSE)

################################################################################

#
# Martinez
#

# Read data

df <- read_csv("../results/run_instances.csv") %>%
  filter(type == "martinez")

# Prepare joint "number solved" and "total timings" table

df %>%
  mutate(solved = (rel_gap < tol_rel_gap & eval_ratio > tol_eval_ratio)) %>%
  mutate(m = 2) %>%
  group_by(relaxation, n, m) %>%
  summarize(num_instances = n(), num_solved = sum(solved), total_time = sum(time)) %>%
  pivot_wider(names_from = relaxation, values_from = c("num_solved", "total_time")) %>%
  mutate(num_instances = comma(num_instances), num_solved_beta = comma(num_solved_beta), n = as.integer(n), m = as.integer(m)) %>%
  select(n, m, num_instances, num_solved_kron, num_solved_beta, total_time_shor, total_time_kron, total_time_beta) %>%
  xtable(digits = 1) %>%
  print(include.rownames=FALSE)

# Determine gap closed

df %>%
  mutate(m = 2) %>%
  mutate(solved = (rel_gap < tol_rel_gap & eval_ratio > tol_eval_ratio)) %>%
  select(relaxation, n, m, seed, pval, dval, solved) %>%
  pivot_wider(names_from = relaxation, values_from = c("pval", "dval", "solved")) %>%
  mutate(pval_best = pmin(pval_shor, pval_kron, pval_beta)) %>%
  select(-pval_shor, -pval_kron, -pval_beta) %>%
  mutate(gap_closed_kron = 1 - (pval_best - dval_kron)/(pval_best - dval_shor)) %>%
  mutate(gap_closed_kron = pmax(0, pmin(1, gap_closed_kron))) %>%
  select(-dval_kron) %>%
  mutate(gap_closed_beta = 1 - (pval_best - dval_beta)/(pval_best - dval_shor)) %>%
  mutate(gap_closed_beta = pmax(0, pmin(1, gap_closed_beta))) %>%
  select(-dval_beta) %>%
  select(-dval_shor, -pval_best) %>%
  # filter(solved_kron == FALSE) %>%
  # group_by(n, m, solved_kron, solved_beta) %>%
  group_by(solved_kron, solved_beta) %>%
  summarize(instances = n(), kron = mean(gap_closed_kron), beta = mean(gap_closed_beta)) %>%
  # mutate(n = as.character(n), m = as.character(m), instances = as.character(instances)) %>%
  mutate(instances = as.character(instances)) %>%
  mutate(kron = percent(kron, 1), beta = percent(beta, 1)) %>%
  mutate(solved_kron = ifelse(solved_kron, "solved", "unsolved")) %>%
  mutate(solved_beta = ifelse(solved_beta, "solved", "unsolved")) %>%
  xtable(digits = 2) %>%
  print(include.rownames=FALSE)


################################################################################

#
# maxnorm
#

df <- read_csv("../results/run_instances.csv") %>%
  filter(type == "maxnorm")

# Prepare joint "number solved" and "total timings" table

df[complete.cases(df), ] %>%
  mutate(solved = (rel_gap < tol_rel_gap & eval_ratio > tol_eval_ratio)) %>%
  mutate(m = extra_balls + 1) %>%
  group_by(relaxation, n, m) %>%
  summarize(num_instances = n(), num_solved = sum(solved), total_time = sum(time)) %>%
  pivot_wider(names_from = relaxation, values_from = c("num_solved", "total_time")) %>%
  mutate(num_instances = comma(num_instances), n = as.integer(n), m = as.integer(m)) %>%
  select(n, m, num_instances, num_solved_kron, num_solved_beta, total_time_shor, total_time_kron, total_time_beta) %>%
  xtable(digits = 1) %>%
  print(include.rownames=FALSE)

# Determine gap closed

df[complete.cases(df), ] %>%
  mutate(m = extra_balls + 1) %>%
  mutate(solved = (rel_gap < tol_rel_gap & eval_ratio > tol_eval_ratio)) %>%
  select(relaxation, n, m, seed, pval, dval, solved) %>%
  pivot_wider(names_from = relaxation, values_from = c("pval", "dval", "solved")) %>%
  mutate(pval_best = pmin(pval_shor, pval_kron, pval_beta)) %>%
  select(-pval_shor, -pval_kron, -pval_beta) %>%
  mutate(gap_closed_kron = 1 - (pval_best - dval_kron)/(pval_best - dval_shor)) %>%
  mutate(gap_closed_kron = pmax(0, pmin(1, gap_closed_kron))) %>%
  select(-dval_kron) %>%
  mutate(gap_closed_beta = 1 - (pval_best - dval_beta)/(pval_best - dval_shor)) %>%
  mutate(gap_closed_beta = pmax(0, pmin(1, gap_closed_beta))) %>%
  select(-dval_beta) %>%
  select(-dval_shor, -pval_best) %>%
  # filter(solved_kron == FALSE) %>%
  # group_by(n, m, solved_kron, solved_beta) %>%
  group_by(solved_kron, solved_beta) %>%
  summarize(instances = n(), kron = mean(gap_closed_kron), beta = mean(gap_closed_beta)) %>%
  # mutate(n = as.character(n), m = as.character(m), instances = as.character(instances)) %>%
  mutate(instances = as.character(instances)) %>%
  mutate(kron = percent(kron, 1), beta = percent(beta, 1)) %>%
  mutate(solved_kron = ifelse(solved_kron, "solved", "unsolved")) %>%
  mutate(solved_beta = ifelse(solved_beta, "solved", "unsolved")) %>%
  xtable(digits = 2) %>%
  print(include.rownames=FALSE)
