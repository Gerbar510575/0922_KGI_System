# Rscript predict.R < stdin JSON
suppressWarnings(suppressMessages({
  library(jsonlite)
  library(workflows)
  library(parsnip)
  library(hardhat)
  library(glmnet)
}))

safe_stop <- function(msg) {
  cat(toJSON(list(error = msg), auto_unbox = TRUE))
  quit(save = "no", status = 1)
}

# 從 stdin 讀 JSON (忽略最後換行警告)
input_text <- tryCatch({
  paste(readLines(file("stdin"), warn = FALSE), collapse = "\n")
}, error = function(e) safe_stop(paste("failed to read stdin:", e$message)))

if (nchar(input_text) == 0) {
  safe_stop("empty input")
}

payload <- tryCatch({
  fromJSON(input_text)
}, error = function(e) safe_stop(paste("invalid JSON input:", e$message)))

# 確保輸入轉成 data.frame
newdata <- tryCatch({
  as.data.frame(payload)
}, error = function(e) safe_stop(paste("cannot convert to data.frame:", e$message)))

model_bundle <- tryCatch({
  readRDS("/app/models/r/crisis_model_bundle.rds")
}, error = function(e) safe_stop(paste("failed to load model bundle:", e$message)))

if (is.null(model_bundle$workflow)) safe_stop("bundle missing workflow")
if (is.null(model_bundle$threshold)) safe_stop("bundle missing threshold")
if (is.null(model_bundle$trained_levels)) safe_stop("bundle missing trained_levels")

model <- model_bundle$workflow
best_thresh <- model_bundle$threshold
trained_levels <- model_bundle$trained_levels$outcome_levels

# 預測機率
prob <- tryCatch({
  predict(model, newdata, type = "prob")
}, error = function(e) safe_stop(paste("prediction failed:", e$message)))

if (ncol(prob) == 0) safe_stop("no probability columns in prediction output")

# 找 Y (正類) 機率
target_col <- intersect(colnames(prob), c("1","Y","Yes","Positive","TRUE"))
p <- tryCatch({
  if (length(target_col) >= 1) as.numeric(prob[[target_col[1]]]) else as.numeric(prob[[1]])
}, error = function(e) safe_stop(paste("cannot extract probability:", e$message)))

# 套用 bundle 的 threshold
pred <- tryCatch({
  ifelse(p >= best_thresh, trained_levels[2], trained_levels[1])
}, error = function(e) safe_stop(paste("thresholding failed:", e$message)))

# 輸出 JSON
cat(toJSON(list(
  predictions = as.list(pred),
  prob = as.list(p),
  threshold = best_thresh,
  method = model_bundle$method
), auto_unbox = TRUE))


