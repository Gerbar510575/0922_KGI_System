# --- 套件 ---
library(tidyverse)
library(tidymodels)
library(themis)   # SMOTE
library(vip)
library(knitr)
library(kableExtra)
#library(jsonlite)
#library(workflows)
#library(parsnip)
#library(hardhat)
#library(glmnet)

# --- 訓練 / 測試切分 ---
set.seed(123)

# --- 1) 資料讀取 ---
df_raw <- read_csv("train.csv") %>%
  mutate(across(where(is.character), as.factor),
         Loan_Status = factor(Loan_Status, levels = c("N", "Y")))   # 強制 factor

# Kaggle 刪除 Loan_ID (你檔案已刪)
# df_raw <- df_raw %>% select(-Loan_ID)

target <- "Loan_Status"

# --- 2) 缺失值處理 ---
# Python 用 KNNImputer (數值) + most_frequent (類別)
# tidymodels 沒有現成 KNN imputer，這裡先用 median/mode 近似 (若要完全一致需寫 custom step)
mode_fun <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}

df_raw <- df_raw %>%
  mutate(
    Gender        = fct_explicit_na(Gender, na_level = as.character(mode_fun(Gender))),
    Married       = fct_explicit_na(Married, na_level = as.character(mode_fun(Married))),
    Dependents    = fct_explicit_na(Dependents, na_level = as.character(mode_fun(Dependents))),
    Self_Employed = fct_explicit_na(Self_Employed, na_level = as.character(mode_fun(Self_Employed))),
    Credit_History = ifelse(is.na(Credit_History),
                            median(Credit_History, na.rm = TRUE),
                            Credit_History),
    LoanAmount = ifelse(is.na(LoanAmount),
                        median(LoanAmount, na.rm = TRUE),
                        LoanAmount),
    Loan_Amount_Term = ifelse(is.na(Loan_Amount_Term),
                              median(Loan_Amount_Term, na.rm = TRUE),
                              Loan_Amount_Term)
  )

# --- 3) 特徵工程 (與 Kaggle 一致) ---
df_raw <- df_raw %>%
  mutate(
    TotalIncome = ApplicantIncome + CoapplicantIncome,
    Loan_Monthly_Paid = LoanAmount / Loan_Amount_Term,
    Income_After_Loan = TotalIncome - LoanAmount,
    Income_to_LoanRatio = TotalIncome / LoanAmount,
    
    log_ApplicantIncome   = log(ApplicantIncome + 1),
    log_LoanAmount        = log(LoanAmount + 1),
    log_TotalIncome       = log(TotalIncome + 1),
    log_Loan_Monthly_Paid = log(Loan_Monthly_Paid + 1),
    log_Income_After_Loan = log(Income_After_Loan + 1),
    log_Income_to_LoanRatio = log(Income_to_LoanRatio + 1)
  ) %>%
  # 刪掉原始數值欄位，保留 log 特徵 (與作者一致)
  select(-ApplicantIncome, -CoapplicantIncome, -LoanAmount,
         -TotalIncome, -Loan_Monthly_Paid, -Income_After_Loan, -Income_to_LoanRatio)

# --- 4) 特徵選擇 (依 Kaggle importance 移除) ---
df_raw <- df_raw %>%
  select(-Education, -Self_Employed, -Gender, -Married)

# --- 5) 訓練 / 測試切分 ---
split <- initial_split(df_raw, prop = 0.8, strata = !!sym(target))
train_df <- training(split)
test_df  <- testing(split)

# --- 6) 配方 (含 SMOTETomek 替代方案) ---
rec <- recipe(as.formula(paste(target, "~ .")), data = train_df) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_lincomb(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.95) #%>%
# step_smote(all_outcomes(), neighbors = 5)

# --- 7a) LASSO Logistic Regression ---
lasso_spec <- logistic_reg(
  penalty = tune(),
  mixture = 1
) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lasso_wf <- workflow() %>% 
  add_model(lasso_spec) %>% 
  add_recipe(rec)

cv_folds <- vfold_cv(train_df, v = 5, strata = !!sym(target))
grid <- grid_regular(penalty(), levels = 40)

# --- metrics_set ---
metrics_set <- metric_set(
  roc_auc,
  accuracy,
  sens,
  spec,
  f_meas
)

lasso_tuned <- tune_grid(
  lasso_wf,
  resamples = cv_folds,
  grid = grid,
  metrics = metrics_set,
  control = control_grid(save_pred = TRUE)
)

lasso_best <- select_best(lasso_tuned, metric = "roc_auc")
final_lasso_wf <- finalize_workflow(lasso_wf, lasso_best)
final_lasso_fit <- fit(final_lasso_wf, data = train_df)

# --- 找最佳 threshold (Max F1) ---
all_preds <- collect_predictions(lasso_tuned)

pr_obj <- pr_curve(all_preds, truth = !!sym(target), .pred_Y, event_level = "second")

f1_scores <- pr_obj %>%
  as_tibble() %>%
  mutate(f1 = 2 * (precision * recall) / (precision + recall)) %>%
  filter(!is.na(f1))

best_thresh <- f1_scores %>%
  slice_max(order_by = f1, n = 1) %>%
  pull(.threshold)

cat("最佳 threshold (Max F1):", best_thresh, "\n")

# --- 測試集預測 (用最佳 threshold) ---
lasso_test_pred <- predict(final_lasso_fit, test_df, type = "prob") %>%
  mutate(.pred_class = factor(if_else(.pred_Y >= best_thresh, "Y", "N"),
                              levels = c("N", "Y"))) %>%
  bind_cols(test_df %>% select(!!sym(target)) %>%
              mutate(Loan_Status = factor(Loan_Status, levels = c("N", "Y"))))

# --- 效能 ---
lasso_perf <- metrics_set(
  lasso_test_pred,
  truth = !!sym(target),
  estimate = .pred_class,
  .pred_Y,
  event_level = "second"
)
print(lasso_perf)

# --- LASSO 特徵重要性 ---
glmnet_fit <- extract_fit_engine(final_lasso_fit)
coefs <- coef(glmnet_fit, s = lasso_best$penalty) %>%
  as.matrix() %>% as.data.frame()
coefs$feature <- rownames(coefs)
colnames(coefs)[1] <- "coef"

imp_features <- coefs %>%
  filter(coef != 0, feature != "(Intercept)") %>%
  arrange(desc(abs(coef)))

top_features <- imp_features %>% slice_max(order_by = abs(coef), n = 10)

top_features %>%
  mutate(coef = round(coef, 3)) %>%
  kable("html", caption = "Loan_Status 預測模型 - LASSO 重要特徵 Top 10") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover"))

ggplot(top_features, aes(x = reorder(feature, abs(coef)), y = coef, fill = coef > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Loan_Status 預測模型 - LASSO 重要特徵 Top 10",
       x = "特徵", y = "LASSO 迴歸係數") +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato")) +
  theme_minimal(base_size = 14)

# =========================
# 12) 存檔 (供 API 使用)
# =========================
trained_wf <- final_lasso_wf %>% fit(data = train_df)

model_bundle <- list(
  workflow   = trained_wf,
  threshold  = best_thresh,   # ★ 用最佳 threshold
  method     = "Max F1 (from CV)",
  trained_levels = list(
    outcome_levels = levels(train_df[[target]])
  ),
  timestamp = Sys.time()
)

saveRDS(model_bundle, file = "crisis_model_bundle.rds")
message("Saved: crisis_model_bundle.rds")
