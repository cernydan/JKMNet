# ---------------------------
# IMPORT LIBRARIES
# ---------------------------
library(data.table)
# ---------------------------

# ---------------------------
# CREATE DEFAULT SETTING 
# ---------------------------
make_defaults <- function() {
  list(
    data = list(
      data_file = "data/inputs/data_all_daily.csv",
      id        = "94206029",
      id_col    = "ID",
      columns   = c("T1","T2","T3","moisture"),
      timestamp = "date"
    ),
    model = list(
      trainer       = "batch",
      architecture  = c(8, 6, 2),
      input_numbers = c(0, 0, 1, 2),
      activation    = "RELU",
      weight_init   = "RANDOM"
    ),
    training = list(
      ensemble_runs = 3,
      max_iterations = 500L,
      max_error      = 0.002,
      learning_rate  = 0.001,
      shuffle        = TRUE,
      seed           = 42L,
      batch_size     = 30L,
      train_fraction = 0.8,
      split_shuffle  = FALSE,
      transform      = "MINMAX",
      transform_alpha = 0.015,
      exclude_last_col_from_transform = FALSE,
      remove_na_before_calib          = FALSE
    ),
    paths = list(
      out_dir = "data/outputs/",
      calib_mat = "data/outputs/calib_mat.csv",
      weights_csv_init = "data/outputs/weights/weights_init.csv",
      weights_bin_init = "data/outputs/weights/weights_init.bin",
      weights_vec_csv_init = "data/outputs/weights/weights_init_vector.csv",
      weights_vec_bin_init = "data/outputs/weights/weights_init_vector.bin",
      weights_csv = "data/outputs/weights/weights_final.csv",
      weights_bin = "data/outputs/weights/weights_final.bin",
      weights_vec_csv = "data/outputs/weights/weights_final_vector.csv",
      weights_vec_bin = "data/outputs/weights/weights_final_vector.bin",
      real_calib = "data/outputs/calib_real.csv",
      pred_calib = "data/outputs/calib_pred.csv",
      real_valid = "data/outputs/valid_real.csv",
      pred_valid = "data/outputs/valid_pred.csv",
      metrics_cal = "data/outputs/metrics/calib_metrics.csv",
      metrics_val = "data/outputs/metrics/valid_metrics.csv",
      run_info = "data/outputs/metrics/run_info.csv",
      errors_csv  = "data/outputs/metrics/errors.csv"
    )
  )
}
# ---------------------------

# ---------------------------
# RENDER TEXT FROM THE FILE
# ---------------------------
render_ini_text <- function(cfg) {
  fmt_bool <- function(x) tolower(as.character(as.logical(x)))
  fmt_vec  <- function(x) paste(x, collapse = ", ")
  fmt_num  <- function(x) if (is.numeric(x)) sprintf("%g", x) else sub(",", ".", as.character(x), fixed = TRUE)
  
  lines <- c(
    "; config_model.ini",
    "",
    "; [data]",
    paste0("data_file = ", cfg$data$data_file, "  ; hourly data for the run"),
    paste0("id = ", cfg$data$id),
    paste0("id_col = ", cfg$data$id_col),
    paste0("columns = ", fmt_vec(cfg$data$columns)),
    paste0("timestamp = ", cfg$data$timestamp, "  ; date (daily data), hour_start (hourly data)"),
    "",
    "; [model]",
    paste0("trainer = ", cfg$model$trainer),
    paste0("architecture = ", fmt_vec(cfg$model$architecture)),
    paste0("input_numbers = ", fmt_vec(cfg$model$input_numbers),
           "  ; length must match number of columns (", paste(cfg$data$columns, collapse=","), ")"),
    paste0("activation = ", cfg$model$activation),
    paste0("weight_init = ", cfg$model$weight_init),
    "",
    "; [training]",
    paste0("ensemble_runs = ", cfg$training$ensemble_runs),
    paste0("max_iterations = ", cfg$training$max_iterations),
    paste0("max_error = ", fmt_num(cfg$training$max_error)),
    paste0("learning_rate = ", fmt_num(cfg$training$learning_rate)),
    paste0("shuffle = ", fmt_bool(cfg$training$shuffle)),
    paste0("seed = ", cfg$training$seed),
    paste0("batch_size = ", cfg$training$batch_size),
    paste0("train_fraction = ", fmt_num(cfg$training$train_fraction)),
    paste0("split_shuffle = ", fmt_bool(cfg$training$split_shuffle)),
    paste0("transform = ", cfg$training$transform),
    paste0("transform_alpha = ", fmt_num(cfg$training$transform_alpha)),
    paste0("exclude_last_col_from_transform = ", fmt_bool(cfg$training$exclude_last_col_from_transform)),
    paste0("remove_na_before_calib = ", fmt_bool(cfg$training$remove_na_before_calib)),
    "",
    "; [paths]",
    paste0("out_dir = ", cfg$paths$out_dir),
    paste0("calib_mat = ", cfg$paths$calib_mat),
    paste0("weights_csv_init = ", cfg$paths$weights_csv_init),
    paste0("weights_bin_init = ", cfg$paths$weights_bin_init),
    paste0("weights_vec_csv_init = ", cfg$paths$weights_vec_csv_init),
    paste0("weights_vec_bin_init = ", cfg$paths$weights_vec_bin_init),
    paste0("weights_csv = ", cfg$paths$weights_csv),
    paste0("weights_bin = ", cfg$paths$weights_bin),
    paste0("weights_vec_csv = ", cfg$paths$weights_vec_csv),
    paste0("weights_vec_bin = ", cfg$paths$weights_vec_bin),
    paste0("real_calib = ", cfg$paths$real_calib),
    paste0("pred_calib = ", cfg$paths$pred_calib),
    paste0("real_valid = ", cfg$paths$real_valid),
    paste0("pred_valid = ", cfg$paths$pred_valid),
    paste0("metrics_cal = ", cfg$paths$metrics_cal),
    paste0("metrics_val = ", cfg$paths$metrics_val),
    paste0("run_info = ", cfg$paths$run_info),
    paste0("errors_csv = ", cfg$paths$errors_csv)
  )
  paste(lines, collapse = "\n")
}
# ---------------------------

normalize_dir <- function(p) sub("[/\\\\]+$", "", p)
ensure_dir <- function(p) if (!dir.exists(p)) dir.create(p, recursive = TRUE)
# ---------------------------

# ---------------------------
# GENERATE CONGIFURATION FILE
# ---------------------------
generate_config_case <- function(
    root_dir,
    id,
    epoch,
    data_source,
    bin_path = "software/JKMNet/bin/JKMNet",
    copy_data_into_inputs = TRUE,
    overwrite_inputs = TRUE,
    filter_input_by_id = TRUE,
    select_only_model_columns = TRUE,
    
    # optional overrides
    trainer = NULL, architecture = NULL, input_numbers = NULL,
    activation = NULL, weight_init = NULL,
    ensemble_runs = NULL,
    learning_rate = NULL, seed = NULL, split_shuffle = NULL,
    columns = NULL, timestamp = NULL, data_file = NULL,
    max_error = NULL, shuffle = NULL, batch_size = NULL,
    train_fraction = NULL, transform = NULL, transform_alpha = NULL,
    exclude_last_col_from_transform = NULL, remove_na_before_calib = NULL
) {
  cfg <- make_defaults()
  
  # basic updates
  cfg$data$id <- as.character(id)
  cfg$training$max_iterations <- as.integer(epoch)
  
  # apply overrides if given
  if (!is.null(columns))       cfg$data$columns <- columns
  if (!is.null(timestamp))     cfg$data$timestamp <- timestamp
  if (!is.null(data_file))     cfg$data$data_file <- data_file
  
  if (!is.null(trainer))       cfg$model$trainer <- trainer
  if (!is.null(architecture))  cfg$model$architecture <- architecture
  if (!is.null(input_numbers)) cfg$model$input_numbers <- input_numbers
  if (!is.null(activation))    cfg$model$activation <- activation
  if (!is.null(weight_init))   cfg$model$weight_init <- weight_init
  
  if (!is.null(ensemble_runs)) cfg$training$ensemble_runs <- as.integer(ensemble_runs)
  if (!is.null(learning_rate)) cfg$training$learning_rate <- as.numeric(learning_rate)
  if (!is.null(seed))          cfg$training$seed <- as.integer(seed)
  if (!is.null(split_shuffle)) cfg$training$split_shuffle <- isTRUE(split_shuffle)
  if (!is.null(max_error))     cfg$training$max_error <- as.numeric(max_error)
  if (!is.null(shuffle))       cfg$training$shuffle <- isTRUE(shuffle)
  if (!is.null(batch_size))    cfg$training$batch_size <- as.integer(batch_size)
  if (!is.null(train_fraction))cfg$training$train_fraction <- as.numeric(train_fraction)
  if (!is.null(transform))     cfg$training$transform <- transform
  if (!is.null(transform_alpha)) cfg$training$transform_alpha <- as.numeric(transform_alpha)
  if (!is.null(exclude_last_col_from_transform))
    cfg$training$exclude_last_col_from_transform <- isTRUE(exclude_last_col_from_transform)
  if (!is.null(remove_na_before_calib))
    cfg$training$remove_na_before_calib <- isTRUE(remove_na_before_calib)
  
  # folders: <root>/<id>/iter<epoch>/
  case_dir  <- file.path(normalize_dir(root_dir), as.character(cfg$data$id))
  iter_dir  <- file.path(case_dir, paste0("iter", cfg$training$max_iterations))
  settings_dir <- file.path(iter_dir, "settings"); ensure_dir(settings_dir)
  inputs_dir   <- file.path(iter_dir, "data", "inputs"); ensure_dir(inputs_dir)
  outputs_dir  <- file.path(iter_dir, "data", "outputs"); ensure_dir(outputs_dir)
  ensure_dir(file.path(outputs_dir, "weights"))
  ensure_dir(file.path(outputs_dir, "metrics"))
  
  # copy & filter data
  if (isTRUE(copy_data_into_inputs)) {
    if (requireNamespace("data.table", quietly = TRUE)) {
      DT <- data.table::fread(data_source, showProgress = FALSE)
      id_col <- cfg$data$id_col
      if (!(id_col %in% names(DT))) {
        warning("Skipping ID ", cfg$data$id, ": column ", id_col, " not found.")
        return(invisible(NULL))
      }
      if (isTRUE(filter_input_by_id)) {
        DT <- DT[as.character(DT[[id_col]]) == as.character(cfg$data$id)]
        if (nrow(DT) == 0L) {
          warning("Skipping ID ", cfg$data$id, ": no rows in dataset.")
          return(invisible(NULL))
        }
        # check for all-NA in modelling columns
        model_cols <- intersect(cfg$data$columns, names(DT))
        if (length(model_cols) > 0) {
          DT <- DT[rowSums(!is.na(DT[, ..model_cols])) > 0]
          if (nrow(DT) == 0L) {
            warning("Skipping ID ", cfg$data$id, ": all rows are NA in model columns.")
            return(invisible(NULL))
          }
        }
      }
      if (isTRUE(select_only_model_columns)) {
        keep <- unique(c(cfg$data$id_col, cfg$data$timestamp, cfg$data$columns))
        keep <- intersect(keep, names(DT))
        DT <- DT[, ..keep]
      }
      data.table::fwrite(DT, file.path(inputs_dir, "data_all_daily.csv"))
    }
    cfg$data$data_file <- "data/inputs/data_all_daily.csv"
  }
  
  # rewrite outputs paths
  cfg$paths$out_dir <- "data/outputs/"
  cfg$paths$calib_mat <- "data/outputs/calib_mat.csv"
  cfg$paths$weights_csv_init <- "data/outputs/weights_init.csv"
  cfg$paths$weights_bin_init <- "data/outputs/weights_init.bin"
  cfg$paths$weights_vec_csv_init <- "data/outputs/weights_init_vector.csv"
  cfg$paths$weights_vec_bin_init <- "data/outputs/weights_init_vector.bin"
  cfg$paths$weights_csv <- "data/outputs/weights_final.csv"
  cfg$paths$weights_bin <- "data/outputs/weights_final.bin"
  cfg$paths$weights_vec_csv <- "data/outputs/weights_final_vector.csv"
  cfg$paths$weights_vec_bin <- "data/outputs/weights_final_vector.bin"
  cfg$paths$real_calib <- "data/outputs/calib_real.csv"
  cfg$paths$pred_calib <- "data/outputs/calib_pred.csv"
  cfg$paths$real_valid <- "data/outputs/valid_real.csv"
  cfg$paths$pred_valid <- "data/outputs/valid_pred.csv"
  cfg$paths$metrics_cal <- "data/outputs/calib_metrics.csv"
  cfg$paths$metrics_val <- "data/outputs/valid_metrics.csv"
  cfg$paths$run_info <- "data/outputs/run_info.csv"
  cfg$paths$errors_csv <- "data/outputs/errors.csv"
  
  # write config file
  out_path <- file.path(settings_dir, "config_model.ini")
  writeLines(render_ini_text(cfg), out_path, useBytes = TRUE)
  
  # copy binary
  file.copy(bin_path, iter_dir, overwrite = TRUE)
  
  invisible(out_path)
}

# ---------------------------

# ---------------------------
# RUN THE FUNCTION TO GENERATE CONFIG FILES AND FOLDERS FOR ALL IDs
# ---------------------------
# source data file
#data_path <- "software/JKMNet/data/inputs/data_all_hourly.csv"  # MetaVO
data_path <- "data/inputs/data_all_hourly.csv"  # MJ local
data_file <- read.csv(data_path)
setDT(data_file)

# select IDs
ids <- unique(data_file$ID)
ids <- sample(ids, size = 2, replace = FALSE)   # randomly select ids, set size=50

# set epochs
# TEST
epochs <- c(500, 1000)
# REAL
# basic epochs
#epochs <- seq(1, 500, by = 100)
# advanced epochs
# epochs <- c(
#     1:50,
#     seq(55, 100, by = 5),
#     seq(200, 1000, by = 100),
#     seq(1000, 5000, by = 500)
#   )

for (id in ids) {
  for (ep in epochs) {
      generate_config_case(
        #root_dir = "home/michalajakubcova/JKMNet_run/epochs",  # MetaVO
        root_dir = "JKMNet_run/",  # MJ local
        id = id,
        epoch = ep,
        data_source = data_path,
        # bin_path = "software/JKMNet/bin/JKMNet"  # MetaVO
        bin_path = "bin/JKMNet",  # MJ local
        
        # overrides
        timestamp = "hour_start",
        activation = "LEAKYRELU",
        weight_init = "RANDOM",
        trainer = "batch",
        ensemble_runs = 25,
        architecture = c(200,48),
        input_numbers = c(0,0,0,168),
        max_error = 0,
        seed = 0,
      )
  }
  cat(sprintf("Generating %s completed\n", id)); flush.console()
}
cat("Task completed\n"); flush.console()
# ---------------------------
