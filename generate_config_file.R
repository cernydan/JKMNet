
library(data.table)

# generate config files for CJKMNet

# make default blueprint

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
      trainer       = "batch",   # online, batch, online_batch, batch_epoch
      architecture  = c(8, 6, 2),
      input_numbers = c(0, 0, 1, 2),
      activation    = "RELU",  # RELU, SIGMOID, LINEAR, TANH, GAUSSIAN, ...
      weight_init   = "RANDOM"   # RANDOM, LHS, LHS2
    ),
    training = list(
      ensemble_runs = 3,
      max_iterations = 500L,
      max_error      = 0.002,
      learning_rate  = 0.001,
      shuffle        = TRUE,
      seed           = 42L,
      batch_size     = 30L,      # only for batch trainer
      train_fraction = 0.8,
      split_shuffle  = FALSE,    # false = chrono split, true = random train/validation split
      transform      = "MINMAX", # NONE, MINMAX, NONLINEAR, ZSCORE
      transform_alpha = 0.015,   # param
      exclude_last_col_from_transform = FALSE,
      remove_na_before_calib          = TRUE
    ),
    paths = list(  
      out_dir = "data/outputs/",
      calib_mat = "data/outputs/calib_mat.csv",
      weights_csv_init = "data/outputs/weights_init.csv",
      weights_bin_init = "data/outputs/weights_init.bin",
      weights_vec_csv_init = "data/outputs/weights_init_vector.csv",
      weights_vec_bin_init = "data/outputs/weights_init_vector.bin",
      weights_csv = "data/outputs/weights_final.csv",
      weights_bin = "data/outputs/weights_final.bin",
      weights_vec_csv = "data/outputs/weights_final_vector.csv",
      weights_vec_bin = "data/outputs/weights_final_vector.bin",
      real_calib = "data/outputs/calib_real.csv",
      pred_calib = "data/outputs/calib_pred.csv",
      real_valid = "data/outputs/valid_real.csv",
      pred_valid = "data/outputs/valid_pred.csv",
      metrics_cal = "data/outputs/calib_metrics.csv",
      metrics_val = "data/outputs/valid_metrics.csv",
      run_info = "data/outputs/run_info.csv",
      errors_csv  = "data/outputs/errors.csv"
    )
  )
}

# creates a list with default values

cfg <- make_defaults()

# render the config as text from list
render_ini_text <- function(cfg) {
  fmt_bool <- function(x) tolower(as.character(as.logical(x)))
  fmt_vec  <- function(x) paste(x, collapse = ", ")
  fmt_num  <- function(x) if (is.numeric(x)) sprintf("%g", x) else sub(",", ".", as.character(x), fixed = TRUE)
  
  lines <- c(
    "; config_model.ini",
    "",
    "; [data]",
    paste0("data_file = ", cfg$data$data_file, "  ; hourly data for th run"),
    paste0("id = ", cfg$data$id),
    paste0("id_col = ", cfg$data$id_col),
    paste0("columns = ", fmt_vec(cfg$data$columns)),
    paste0("timestamp = ", cfg$data$timestamp, "  ; date (daily data), hour_start (hourly data)"),
    "",
    "; [model]",
    paste0("trainer = ", cfg$model$trainer, "  ; online, batch"),
    paste0("architecture = ", fmt_vec(cfg$model$architecture)),
    paste0("input_numbers = ", fmt_vec(cfg$model$input_numbers),
           "  ; length must match number of columns (", paste(cfg$data$columns, collapse=","), ")"),
    paste0("activation = ", cfg$model$activation, "  ; RELU, SIGMOID, LINEAR, TANH, GAUSSIAN, ... "),
    paste0("weight_init = ", cfg$model$weight_init, "  ; RANDOM, LHS, LHS2"),
    "",
    "; [training]",
    paste0("ensemble_runs = ", cfg$training$ensemble_runs),
    paste0("max_iterations = ", cfg$training$max_iterations),
    paste0("max_error = ", fmt_num(cfg$training$max_error)),
    paste0("learning_rate = ", fmt_num(cfg$training$learning_rate)),
    paste0("shuffle = ", fmt_bool(cfg$training$shuffle)),
    paste0("seed = ", cfg$training$seed),
    paste0("batch_size = ", cfg$training$batch_size, "  ; used only for batch trainer"),
    paste0("train_fraction = ", fmt_num(cfg$training$train_fraction)),
    paste0("split_shuffle = ", fmt_bool(cfg$training$split_shuffle), "  ; false = chronological, true = random train/validation split"),
    paste0("transform = ", cfg$training$transform, "     ; NONE, MINMAX, NONLINEAR, ZSCORE"),
    paste0("transform_alpha = ", fmt_num(cfg$training$transform_alpha), "  ; param"),
    paste0("exclude_last_col_from_transform = ", fmt_bool(cfg$training$exclude_last_col_from_transform)),
    paste0("remove_na_before_calib = ", fmt_bool(cfg$training$remove_na_before_calib)),
    "",
    "; [paths] ",
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

normalize_dir <- function(p) sub("[/\\\\]+$", "", p)
ensure_dir <- function(p) if (!dir.exists(p)) dir.create(p, recursive = TRUE)

# generate the cases based on selected ids
generate_config_case <- function(
    root_dir,
    id,
    epoch,                               # max_iterations
    data_source = NULL,                  # path to data source (if NULL, use data_file from defaults)
    copy_data_into_inputs = TRUE,        # save data to data/inputs
    overwrite_inputs = TRUE,             # overwrite existing inputs if copy_data_into_inputs = TRUE
    
    # filter data before copying into inputs
    filter_input_by_id = TRUE,           # filter by ID
    select_only_model_columns = TRUE,    # only model columns
    
    # model parameters
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
  
  # update the template
  cfg$data$id <- as.character(id)
  cfg$training$max_iterations <- as.integer(epoch)
  
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
  
  # # sanity
  # stopifnot(length(cfg$data$columns) > 0)
  # if (length(cfg$model$input_numbers) != length(cfg$data$columns))
  #   stop("`input_numbers` musí mít stejnou délku jako `columns`.")
  # if (cfg$training$max_iterations <= 0) stop("`epoch`/`max_iterations` musí být > 0.")
  # if (!(cfg$training$train_fraction > 0 && cfg$training$train_fraction <= 1))
  #   stop("`train_fraction` musí být v (0, 1].")
  
  # folders structure
  root <- normalize_dir(root_dir); ensure_dir(root)
  case_dir  <- file.path(root, paste0("case_", cfg$data$id), fsep = "/"); ensure_dir(case_dir)
  epoch_dir <- file.path(case_dir, paste0("epoch_", cfg$training$max_iterations), fsep = "/"); ensure_dir(epoch_dir)
  data_dir  <- file.path(epoch_dir, "data", fsep = "/"); ensure_dir(data_dir)
  inputs_dir  <- file.path(data_dir, "inputs", fsep = "/"); ensure_dir(inputs_dir)
  outputs_dir <- file.path(data_dir, "outputs", fsep = "/"); ensure_dir(outputs_dir)
  set_dir   <- file.path(epoch_dir, "set",  fsep = "/"); ensure_dir(set_dir)
  
  src  <- if (!is.null(data_source)) data_source else cfg$data$data_file
  dest <- file.path(inputs_dir, basename(src), fsep = "/")
  
  # copy original or use filtered data
  if (isTRUE(copy_data_into_inputs)) {
    if (!file.exists(src)) stop("No data file", src)
    
    if (isTRUE(filter_input_by_id)) {
      if (requireNamespace("data.table", quietly = TRUE)) {
        DT <- data.table::fread(src, showProgress = FALSE)
        id_col <- cfg$data$id_col
        if (!(id_col %in% names(DT))) stop("ID column is missing ", id_col)
        # filter
        DT <- DT[as.character(DT[[id_col]]) == as.character(cfg$data$id)]
        if (nrow(DT) == 0L) stop("Selected ID:", cfg$data$id, " has no rows.")
        # cols all or only model specified
        if (isTRUE(select_only_model_columns)) {
          keep <- unique(c(cfg$data$id_col, cfg$data$timestamp, cfg$data$columns))
          keep <- intersect(keep, names(DT))
          DT <- DT[, ..keep]
        }
        data.table::fwrite(DT, dest)
      } else {
        df <- read.csv(src, stringsAsFactors = FALSE)
        id_col <- cfg$data$id_col
        if (!(id_col %in% names(df))) stop("V datech chybí sloupec ID: ", id_col)
        df <- df[as.character(df[[id_col]]) == as.character(cfg$data$id), , drop = FALSE]
        if (nrow(df) == 0L) stop("Pro ID ", cfg$data$id, " nejsou v datech žádné řádky.")
        if (isTRUE(select_only_model_columns)) {
          keep <- unique(c(cfg$data$id_col, cfg$data$timestamp, cfg$data$columns))
          keep <- keep[keep %in% names(df)]
          df <- df[, keep, drop = FALSE]
        }
        write.csv(df, dest, row.names = FALSE)
      }
    } else {
      # all data, no filter
      ok <- file.copy(src, dest, overwrite = isTRUE(overwrite_inputs))
      if (!ok) stop("Coping of data failed: ", dest)
    }
    cfg$data$data_file <- dest
  } else {

    cfg$data$data_file <- src
  }
  
  # outputs -> data/outputs
  cfg$paths$out_dir             <- paste0(normalize_dir(outputs_dir), "/")
  cfg$paths$calib_mat           <- file.path(outputs_dir, "calib_mat.csv", fsep = "/")
  cfg$paths$weights_csv_init    <- file.path(outputs_dir, "weights_init.csv", fsep = "/")
  cfg$paths$weights_bin_init    <- file.path(outputs_dir, "weights_init.bin", fsep = "/")
  cfg$paths$weights_vec_csv_init<- file.path(outputs_dir, "weights_init_vector.csv", fsep = "/")
  cfg$paths$weights_vec_bin_init<- file.path(outputs_dir, "weights_init_vector.bin", fsep = "/")
  cfg$paths$weights_csv         <- file.path(outputs_dir, "weights_final.csv", fsep = "/")
  cfg$paths$weights_bin         <- file.path(outputs_dir, "weights_final.bin", fsep = "/")
  cfg$paths$weights_vec_csv     <- file.path(outputs_dir, "weights_final_vector.csv", fsep = "/")
  cfg$paths$weights_vec_bin     <- file.path(outputs_dir, "weights_final_vector.bin", fsep = "/")
  cfg$paths$real_calib          <- file.path(outputs_dir, "calib_real.csv", fsep = "/")
  cfg$paths$pred_calib          <- file.path(outputs_dir, "calib_pred.csv", fsep = "/")
  cfg$paths$real_valid          <- file.path(outputs_dir, "valid_real.csv", fsep = "/")
  cfg$paths$pred_valid          <- file.path(outputs_dir, "valid_pred.csv", fsep = "/")
  cfg$paths$metrics_cal         <- file.path(outputs_dir, "calib_metrics.csv", fsep = "/")
  cfg$paths$metrics_val         <- file.path(outputs_dir, "valid_metrics.csv", fsep = "/")
  cfg$paths$run_info            <- file.path(outputs_dir, "run_info.csv", fsep = "/")
  cfg$paths$errors_csv          <- file.path(outputs_dir, "errors.csv", fsep = "/")
  
  out_path <- file.path(set_dir, "config_model.ini", fsep = "/")
  writeLines(render_ini_text(cfg), out_path, useBytes = TRUE)
  invisible(out_path)
}

# # one at a time
# generate_config_case_epoch(
#   root_dir = "configs",
#   id = "93254125",
#   epoch = 300,
#   data_source = "C:/Users/kuzelkova/Desktop/C/JKMNet/data/inputs/data_all_daily.csv",
#   activation = "RELU",
#   weight_init = "LHS",
#   architecture = c(8,6,2),
#   input_numbers = c(0,0,1,2)
# )


# load source data
data <- read.csv("C:/Users/kuzelkova/Desktop/data_all_hourly.csv") # path to data_all_daily.csv
setDT(data)
#data <- data[hour_start > "2024-01-01"] # filter by date

# select ids for cases
ids <- unique(data$ID)

set.seed(42)

ids <- sample(ids, size = 50, replace = FALSE) # randomly select ids




# basic epochs

epochs <- seq(1, 500, by = 100)

# advanced epochs
 
# epochs <- c(
#     1:50,
#     seq(55, 100, by = 5),
#     seq(200, 1000, by = 100),
#     seq(1000, 5000, by = 500)
#   )

# cycle for generating each case, set the parameters, unset will be default

for (id in ids) {
  for (ep in epochs) {
    tryCatch(
      generate_config_case(
        root_dir = "cases",
        id = id,
        epoch = ep,
        data_source = "C:/Users/kuzelkova/Desktop/data_all_hourly.csv",
        filter_input_by_id = TRUE,            # filters the data by ID before coping into inputs
        select_only_model_columns = FALSE,
        
        timestamp = "hour_start",
        activation = "LINEAR",
        weight_init = "RANDOM",
        architecture = c(10,20),
        input_numbers = c(0,0,0,1),
        max_error = 0
      ),
      error = function(e) message("Case ", id, ", epoch ", ep, " failed: ", e$message)
    )
  }
  cat(sprintf("Generating case_%s completed\n", id)); flush.console()
}
cat("Task completed\n"); flush.console()


