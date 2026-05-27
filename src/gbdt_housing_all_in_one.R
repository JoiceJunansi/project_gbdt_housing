#!/usr/bin/env Rscript

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults

  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    if (length(kv) == 2 && nzchar(kv[1])) out[[kv[1]]] <- kv[2]
  }

  out
}

to_bool <- function(x) {
  tolower(x) %in% c("1", "true", "yes", "y")
}

mape <- function(actual, pred) {
  mean(abs((actual - pred) / pmax(abs(actual), 1e-8))) * 100
}

safe_dir_create <- function(path) {
  dir_path <- dirname(path)
  if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
}

prepare_preprocessor <- function(df, target_col) {
  feature_cols <- setdiff(names(df), target_col)
  numeric_cols <- feature_cols[sapply(df[feature_cols], is.numeric)]
  categorical_cols <- setdiff(feature_cols, numeric_cols)

  medians <- list()
  levels_map <- list()

  for (col in numeric_cols) {
    x <- suppressWarnings(as.numeric(df[[col]]))
    med <- suppressWarnings(stats::median(x, na.rm = TRUE))
    if (!is.finite(med)) med <- 0
    medians[[col]] <- med
  }

  for (col in categorical_cols) {
    x <- as.character(df[[col]])
    x[is.na(x) | trimws(x) == ""] <- "UNKNOWN"
    lvls <- sort(unique(x))
    if (!("UNKNOWN" %in% lvls)) lvls <- c(lvls, "UNKNOWN")
    levels_map[[col]] <- lvls
  }

  list(
    target_col = target_col,
    numeric_cols = numeric_cols,
    categorical_cols = categorical_cols,
    medians = medians,
    levels_map = levels_map
  )
}

apply_preprocessor <- function(df, prep, include_target = FALSE) {
  out <- df

  for (col in prep$numeric_cols) {
    if (!(col %in% names(out))) out[[col]] <- NA_real_
    x <- suppressWarnings(as.numeric(out[[col]]))
    x[is.na(x)] <- prep$medians[[col]]
    out[[col]] <- x
  }

  for (col in prep$categorical_cols) {
    if (!(col %in% names(out))) out[[col]] <- "UNKNOWN"
    x <- as.character(out[[col]])
    x[is.na(x) | trimws(x) == ""] <- "UNKNOWN"

    known_levels <- prep$levels_map[[col]]
    x[!(x %in% known_levels)] <- "UNKNOWN"

    final_levels <- known_levels
    if (!("UNKNOWN" %in% final_levels)) final_levels <- c(final_levels, "UNKNOWN")
    out[[col]] <- factor(x, levels = final_levels)
  }

  if (include_target && prep$target_col %in% names(out)) {
    out[[prep$target_col]] <- suppressWarnings(as.numeric(out[[prep$target_col]]))
  }

  out
}

train_gbdt <- function(data_path, target_col, test_size, seed, model_out, metrics_out, pred_test_out) {
  if (!file.exists(data_path)) stop(sprintf("File data tidak ditemukan: %s", data_path))

  cat("Membaca data training...\n")
  t0 <- Sys.time()
  df <- read.csv(data_path, stringsAsFactors = FALSE)

  if (!(target_col %in% names(df))) stop(sprintf("Kolom target '%s' tidak ditemukan.", target_col))

  df[[target_col]] <- suppressWarnings(as.numeric(df[[target_col]]))
  df <- df[!is.na(df[[target_col]]), , drop = FALSE]

  set.seed(seed)
  n <- nrow(df)
  idx_test <- sample(seq_len(n), size = floor(n * test_size))
  test_raw <- df[idx_test, , drop = FALSE]
  train_raw <- df[-idx_test, , drop = FALSE]

  set.seed(seed + 1L)
  n_train <- nrow(train_raw)
  idx_val <- sample(seq_len(n_train), size = floor(n_train * 0.2))
  val_raw <- train_raw[idx_val, , drop = FALSE]
  tr_raw <- train_raw[-idx_val, , drop = FALSE]

  prep <- prepare_preprocessor(tr_raw, target_col)
  tr <- apply_preprocessor(tr_raw, prep, include_target = TRUE)
  val <- apply_preprocessor(val_raw, prep, include_target = TRUE)
  train_full <- apply_preprocessor(train_raw, prep, include_target = TRUE)
  test <- apply_preprocessor(test_raw, prep, include_target = TRUE)

  grid <- expand.grid(
    n.trees = c(300, 600, 1000),
    interaction.depth = c(2, 3, 5),
    shrinkage = c(0.03, 0.05, 0.1),
    n.minobsinnode = c(10, 20),
    bag.fraction = c(0.8),
    stringsAsFactors = FALSE
  )

  form <- as.formula(paste(target_col, "~ ."))
  best_idx <- NA_integer_
  best_mape <- Inf

  cat(sprintf("Tuning %d kombinasi...\n", nrow(grid)))

  for (i in seq_len(nrow(grid))) {
    p <- grid[i, ]

    fit <- gbm::gbm(
      formula = form,
      data = tr,
      distribution = "gaussian",
      n.trees = p$n.trees,
      interaction.depth = p$interaction.depth,
      shrinkage = p$shrinkage,
      n.minobsinnode = p$n.minobsinnode,
      bag.fraction = p$bag.fraction,
      train.fraction = 1.0,
      n.cores = 1,
      verbose = FALSE
    )

    pred_val <- predict(fit, newdata = val, n.trees = p$n.trees)
    cur_mape <- mape(val[[target_col]], pred_val)
    cat(sprintf("Grid %d/%d | MAPE validasi: %.4f%%\n", i, nrow(grid), cur_mape))

    if (cur_mape < best_mape) {
      best_mape <- cur_mape
      best_idx <- i
    }
  }

  best_params <- grid[best_idx, , drop = FALSE]
  cat("Training final model...\n")

  final_model <- gbm::gbm(
    formula = form,
    data = train_full,
    distribution = "gaussian",
    n.trees = best_params$n.trees,
    interaction.depth = best_params$interaction.depth,
    shrinkage = best_params$shrinkage,
    n.minobsinnode = best_params$n.minobsinnode,
    bag.fraction = best_params$bag.fraction,
    train.fraction = 1.0,
    n.cores = 1,
    verbose = FALSE
  )

  pred_test <- predict(final_model, newdata = test, n.trees = best_params$n.trees)
  actual_test <- test[[target_col]]

  mae <- mean(abs(actual_test - pred_test))
  rmse <- sqrt(mean((actual_test - pred_test)^2))
  sse <- sum((actual_test - pred_test)^2)
  sst <- sum((actual_test - mean(actual_test))^2)
  r2 <- 1 - (sse / sst)
  mape_test <- mape(actual_test, pred_test)
  runtime_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  metrics <- data.frame(
    model = "GBDT (gbm)",
    mae = mae,
    rmse = rmse,
    r2 = r2,
    mape_percent = mape_test,
    best_n_trees = best_params$n.trees,
    best_interaction_depth = best_params$interaction.depth,
    best_shrinkage = best_params$shrinkage,
    best_n_minobsinnode = best_params$n.minobsinnode,
    best_bag_fraction = best_params$bag.fraction,
    runtime_seconds = runtime_sec,
    stringsAsFactors = FALSE
  )

  bundle <- list(
    model = final_model,
    preprocessor = prep,
    target_col = target_col,
    best_params = best_params
  )

  safe_dir_create(model_out)
  safe_dir_create(metrics_out)
  safe_dir_create(pred_test_out)

  saveRDS(bundle, model_out)
  write.csv(metrics, metrics_out, row.names = FALSE)

  out_test <- test_raw
  out_test$prediction <- pred_test
  write.csv(out_test, pred_test_out, row.names = FALSE)

  cat("\n=== HASIL TRAINING ===\n")
  cat(sprintf("MAE      : %.4f\n", mae))
  cat(sprintf("RMSE     : %.4f\n", rmse))
  cat(sprintf("R2       : %.4f\n", r2))
  cat(sprintf("MAPE     : %.4f%%\n", mape_test))
  cat(sprintf("Runtime  : %.2f detik\n", runtime_sec))
  cat(sprintf("Model    : %s\n", normalizePath(model_out, winslash = "/", mustWork = FALSE)))
  cat(sprintf("Metrik   : %s\n", normalizePath(metrics_out, winslash = "/", mustWork = FALSE)))
  cat(sprintf("Pred Test: %s\n", normalizePath(pred_test_out, winslash = "/", mustWork = FALSE)))

  invisible(bundle)
}

predict_gbdt <- function(model_obj, model_path, input_path, output_path) {
  bundle <- model_obj
  if (is.null(bundle)) {
    if (!file.exists(model_path)) stop(sprintf("File model tidak ditemukan: %s", model_path))
    bundle <- readRDS(model_path)
  }

  if (!file.exists(input_path)) stop(sprintf("File input tidak ditemukan: %s", input_path))

  new_df <- read.csv(input_path, stringsAsFactors = FALSE)
  prep <- bundle$preprocessor
  model <- bundle$model
  best_params <- bundle$best_params
  target_col <- bundle$target_col

  processed <- apply_preprocessor(new_df, prep, include_target = FALSE)
  feature_cols <- c(prep$numeric_cols, prep$categorical_cols)
  processed <- processed[, feature_cols, drop = FALSE]

  preds <- predict(model, newdata = processed, n.trees = best_params$n.trees)

  out <- new_df
  out$prediction <- preds

  safe_dir_create(output_path)
  write.csv(out, output_path, row.names = FALSE)

  cat("\n=== HASIL PREDIKSI DATA BARU ===\n")
  cat(sprintf("Output: %s\n", normalizePath(output_path, winslash = "/", mustWork = FALSE)))

  if (target_col %in% names(new_df)) {
    actual <- suppressWarnings(as.numeric(new_df[[target_col]]))
    idx <- !is.na(actual)
    if (sum(idx) > 0) {
      m <- mape(actual[idx], preds[idx])
      cat(sprintf("MAPE pada data input: %.4f%%\n", m))
    }
  }
}

main <- function() {
  defaults <- list(
    mode = "both",
    auto_install = "false",
    data = "data/housing.csv",
    target = "median_house_value",
    test_size = "0.2",
    seed = "42",
    model_out = "models/gbdt_housing_model.rds",
    metrics_out = "reports/metrics_gbdt_r.csv",
    pred_test_out = "reports/predictions_test_gbdt_r.csv",
    predict_input = "data/housing.csv",
    predict_out = "reports/predictions_new_data_gbdt_r.csv"
  )

  args <- parse_args(defaults)
  mode <- tolower(args$mode)
  model_path <- if (!is.null(args$model)) args$model else args$model_out

  if (to_bool(args$auto_install) && !requireNamespace("gbm", quietly = TRUE)) {
    install.packages("gbm", repos = "https://cran.rstudio.com")
  }
  if (!requireNamespace("gbm", quietly = TRUE)) {
    stop("Package 'gbm' belum terpasang. Jalankan: install.packages('gbm')")
  }

  test_size <- as.numeric(args$test_size)
  seed <- as.integer(args$seed)

  trained_bundle <- NULL

  if (mode %in% c("train", "both")) {
    trained_bundle <- train_gbdt(
      data_path = args$data,
      target_col = args$target,
      test_size = test_size,
      seed = seed,
      model_out = model_path,
      metrics_out = args$metrics_out,
      pred_test_out = args$pred_test_out
    )
  }

  if (mode %in% c("predict", "both")) {
    predict_gbdt(
      model_obj = trained_bundle,
      model_path = model_path,
      input_path = args$predict_input,
      output_path = args$predict_out
    )
  }
}

main()
