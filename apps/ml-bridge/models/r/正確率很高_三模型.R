# --- 套件 ---
library(tidyverse)
library(tidymodels)
library(themis)       # SMOTE / Tomek
#library(smotefamily)  # 若需要 SMOTETomek，可透過 themis::step_smote() + 自訂調整
library(vip)
library(kableExtra)

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
# tidymodels 目前 only 支援 SMOTE，不支援 SMOTETomek
# 若要完全 replicate，需額外包 smotefamily::SMOTE + TomekLink
# 這裡先近似：step_smote()
rec <- recipe(as.formula(paste(target, "~ .")), data = train_df) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_lincomb(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.95) #%>%
  #step_smote(all_outcomes(), neighbors = 5)

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

# --- 修正的 metrics_set ---
metrics_set <- metric_set(
  roc_auc,        # 直接使用內建
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

lasso_test_pred <- predict(final_lasso_fit, test_df, type = "prob") %>%
  mutate(.pred_class = factor(if_else(.pred_Y >= 0.5, "Y", "N"),
                              levels = c("N", "Y"))) %>%
  bind_cols(test_df %>% select(!!sym(target)) %>%
              mutate(Loan_Status = factor(Loan_Status, levels = c("N", "Y"))))  # 保證 factor

lasso_perf <- metrics_set(
  lasso_test_pred,
  truth = !!sym(target),
  estimate = .pred_class,
  .pred_Y,
  event_level = "second"   # **這裡指定正類為 Y**
)

#print(lasso_perf)




# --- 7b) XGBoost ---
xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(rec)

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  min_n(),
  sample_size = sample_prop(),
  finalize(mtry(), train_df),
  size = 20
)

xgb_tuned <- tune_grid(
  xgb_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = metrics_set,
  control = control_grid(save_pred = TRUE)
)

xgb_best <- select_best(xgb_tuned, metric = "roc_auc")
final_xgb_wf <- finalize_workflow(xgb_wf, xgb_best)
final_xgb_fit <- fit(final_xgb_wf, data = train_df)

xgb_test_pred <- predict(final_xgb_fit, test_df, type = "prob") %>%
  mutate(.pred_class = factor(if_else(.pred_Y >= 0.5, "Y", "N"),
                              levels = c("N", "Y"))) %>%
  bind_cols(test_df %>% select(!!sym(target)) %>%
              mutate(Loan_Status = factor(Loan_Status, levels = c("N", "Y"))))

xgb_perf <- metrics_set(
  xgb_test_pred,
  truth = !!sym(target),
  estimate = .pred_class,
  .pred_Y,
  event_level = "second"
)
#print(xgb_perf)

# --- 8) 繪圖 ROC / PR (XGBoost) ---
roc_obj <- roc_curve(xgb_test_pred, truth = !!sym(target), .pred_Y, event_level = "second")
autoplot(roc_obj)

pr_obj <- pr_curve(xgb_test_pred, truth = !!sym(target), .pred_Y, event_level = "second")
autoplot(pr_obj)


# --- 7c) Support Vector Machine (SVM, RBF kernel) ---
svm_spec <- svm_rbf(
  cost = tune(),          # C
  rbf_sigma = tune()      # gamma
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_wf <- workflow() %>%
  add_model(svm_spec) %>%
  add_recipe(rec)

# --- 超參數搜尋空間 (與 XGBoost 相同使用拉丁超立方隨機搜尋) ---
svm_grid <- grid_latin_hypercube(
  cost(),
  rbf_sigma(),
  size = 20
)

# --- 調參 ---
svm_tuned <- tune_grid(
  svm_wf,
  resamples = cv_folds,
  grid = svm_grid,
  metrics = metrics_set,
  control = control_grid(save_pred = TRUE)
)

# --- 選最佳參數 ---
svm_best <- select_best(svm_tuned, metric = "roc_auc")
final_svm_wf <- finalize_workflow(svm_wf, svm_best)
final_svm_fit <- fit(final_svm_wf, data = train_df)

# --- 測試集預測 ---
svm_test_pred <- predict(final_svm_fit, test_df, type = "prob") %>%
  mutate(.pred_class = factor(if_else(.pred_Y >= 0.5, "Y", "N"),
                              levels = c("N", "Y"))) %>%
  bind_cols(test_df %>% select(!!sym(target)) %>%
              mutate(Loan_Status = factor(Loan_Status, levels = c("N", "Y"))))

# --- 效能輸出 ---
svm_perf <- metrics_set(
  svm_test_pred,
  truth = !!sym(target),
  estimate = .pred_class,
  .pred_Y,
  event_level = "second"
)
#print(svm_perf)

# --- 視覺化 ROC / PR (SVM) ---
svm_roc <- roc_curve(svm_test_pred, truth = !!sym(target), .pred_Y, event_level = "second")
autoplot(svm_roc)

svm_pr <- pr_curve(svm_test_pred, truth = !!sym(target), .pred_Y, event_level = "second")
autoplot(svm_pr)


# --- 整理三個模型的效能比較 ---
model_comparison <- bind_rows(
  lasso_perf %>% mutate(Model = "LASSO Logistic"),
  xgb_perf   %>% mutate(Model = "XGBoost"),
  svm_perf   %>% mutate(Model = "SVM (RBF)")
) %>%
  select(Model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# --- 輸出表格 ---
model_comparison %>%
  mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  kable("html", caption = "三種模型效能比較") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover"))


# --- LASSO 特徵重要性 ---
lasso_fit_obj <- extract_fit_engine(final_lasso_fit)
lasso_coefs <- coef(lasso_fit_obj, s = lasso_best$penalty) %>%
  as.matrix() %>% as.data.frame()
lasso_coefs$feature <- rownames(lasso_coefs)
colnames(lasso_coefs)[1] <- "coef"

lasso_imp <- lasso_coefs %>%
  filter(feature != "(Intercept)", coef != 0) %>%
  mutate(Importance = abs(coef)) %>%
  arrange(desc(Importance)) %>%
  mutate(Model = "LASSO") %>%
  select(Model, Feature = feature, Importance)

# --- XGBoost 特徵重要性 ---
xgb_fit_obj <- extract_fit_engine(final_xgb_fit)
xgb_imp <- xgboost::xgb.importance(model = xgb_fit_obj) %>%
  as_tibble() %>%
  select(Feature = Feature, Importance = Gain) %>%
  arrange(desc(Importance)) %>%
  mutate(Model = "XGBoost")

# --- SVM 特徵重要性 (Permutation Importance) ---
set.seed(123)

# 自訂 wrapper，輸出 Y 的機率
svm_pred_wrapper <- function(object, newdata) {
  predict(object, newdata, type = "prob")$.pred_Y
}

svm_imp <- vip::vi_permute(
  object = final_svm_fit,
  feature_names = setdiff(names(train_df), target), # 全部特徵
  train = train_df,
  target = target,
  metric = "roc_auc",
  pred_wrapper = svm_pred_wrapper,
  reference_class = "Y",   # 正類
  nsim = 10,               # 次數可調大以更穩定
  event_level = "second"
) %>%
  as_tibble() %>%
  arrange(desc(Importance)) %>%
  mutate(Model = "SVM (Permutation)") %>%
  select(Model, Feature = Variable, Importance)


# --- 合併三種模型的重要性 ---
all_importance <- bind_rows(
  lasso_imp,
  xgb_imp,
  svm_imp
)

# --- 顯示前 15 個特徵 ---
all_importance %>%
  group_by(Model) %>%
  slice_max(order_by = Importance, n = 15) %>%
  ungroup() %>%
  mutate(Importance = round(Importance, 3)) %>%
  kable("html", caption = "三種模型的特徵重要性比較 (前 15)") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed"))

