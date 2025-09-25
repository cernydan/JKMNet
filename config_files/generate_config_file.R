
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
      trainer       = "batch",   # online, batch
      architecture  = c(8, 6, 2),
      input_numbers = c(0, 0, 1, 2),
      activation    = "RELU",  # RELU, SIGMOID, LINEAR, TANH, GAUSSIAN, ...
      weight_init   = "RANDOM"   # RANDOM, LHS, LHS2
    ),
    training = list(
      max_iterations = 500L,
      max_error      = 0.002,
      learning_rate  = 0.001,
      shuffle        = TRUE,
      seed           = 42L,
      batch_size     = 30L,      # only for batch trainer
      train_fraction = 0.8,
      split_shuffle  = FALSE,    # false = chrono split, true = random split
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
      run_info = "data/outputs/run_info.csv"
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
    paste0("run_info = ", cfg$paths$run_info)
  )
  paste(lines, collapse = "\n")
}


# generate ini file
generate_config <- function(
  out_dir,
  config_name_pattern = "config_{id}.ini",
  include_id_in_paths = TRUE,
  
  # parameters
  id = NULL, 
  data_file = NULL, 
  id_col = NULL, 
  columns = NULL, 
  timestamp = NULL,
  trainer = NULL, 
  architecture = NULL, 
  input_numbers = NULL, 
  activation = NULL, 
  weight_init = NULL,
  max_iterations = NULL, 
  max_error = NULL, 
  learning_rate = NULL, 
  shuffle = NULL,
  seed = NULL, 
  batch_size = NULL, 
  train_fraction = NULL, 
  transform = NULL,
  transform_alpha = NULL, 
  exclude_last_col_from_transform = NULL, 
  remove_na_before_calib = NULL
) {
  

  # set default values for parameters
  if (!is.null(id))        cfg$data$id <- as.character(id)
  if (!is.null(data_file)) cfg$data$data_file <- data_file
  if (!is.null(id_col))    cfg$data$id_col <- id_col
  if (!is.null(columns))   cfg$data$columns <- columns
  if (!is.null(timestamp)) cfg$data$timestamp <- timestamp
  
  if (!is.null(trainer))       cfg$model$trainer <- trainer
  if (!is.null(architecture))  cfg$model$architecture <- architecture
  if (!is.null(input_numbers)) cfg$model$input_numbers <- input_numbers
  if (!is.null(activation))    cfg$model$activation <- activation
  if (!is.null(weight_init))   cfg$model$weight_init <- weight_init
  
  if (!is.null(max_iterations))  cfg$training$max_iterations <- as.integer(max_iterations)
  if (!is.null(max_error))       cfg$training$max_error <- as.numeric(max_error)
  if (!is.null(learning_rate))   cfg$training$learning_rate <- as.numeric(learning_rate)
  if (!is.null(shuffle))         cfg$training$shuffle <- isTRUE(shuffle)
  if (!is.null(seed))            cfg$training$seed <- as.integer(seed)
  if (!is.null(batch_size))      cfg$training$batch_size <- as.integer(batch_size)
  if (!is.null(train_fraction))  cfg$training$train_fraction <- as.numeric(train_fraction)
  if (!is.null(transform))       cfg$training$transform <- transform
  if (!is.null(transform_alpha)) cfg$training$transform_alpha <- as.numeric(transform_alpha)
  if (!is.null(exclude_last_col_from_transform))
    cfg$training$exclude_last_col_from_transform <- isTRUE(exclude_last_col_from_transform)
  if (!is.null(remove_na_before_calib))
    cfg$training$remove_na_before_calib <- isTRUE(remove_na_before_calib)
  
  # file output
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  cfg$paths$out_dir <- if (grepl("/$", out_dir)) out_dir else paste0(out_dir, "/")
  
  # ids in filenames
  idc <- as.character(cfg$data$id)
  suffix <- if (isTRUE(include_id_in_paths)) paste0("_", idc) else ""
  cfg$paths$calib_mat   <- file.path(out_dir, paste0("calib_mat",   suffix, ".csv"))
  cfg$paths$weights_csv <- file.path(out_dir, paste0("weights_final",     suffix, ".csv"))
  cfg$paths$weights_bin <- file.path(out_dir, paste0("weights_final",     suffix, ".bin"))
  cfg$paths$real_calib  <- file.path(out_dir, paste0("calib_real",  suffix, ".csv"))
  cfg$paths$pred_calib  <- file.path(out_dir, paste0("calib_pred",  suffix, ".csv"))
  cfg$paths$real_valid  <- file.path(out_dir, paste0("valid_real",  suffix, ".csv"))
  cfg$paths$pred_valid  <- file.path(out_dir, paste0("valid_pred",  suffix, ".csv"))
  cfg$paths$metrics_cal <- file.path(out_dir, paste0("calib_metrics", suffix, ".csv"))
  cfg$paths$metrics_val <- file.path(out_dir, paste0("valid_metrics", suffix, ".csv"))
  
  # config file name
  cfg_name <- gsub("\\{id\\}", idc, config_name_pattern)
  out_path <- file.path(out_dir, cfg_name)
  
  txt <- render_ini_text(cfg)
  writeLines(txt, out_path, useBytes = TRUE)
  invisible(out_path)
  
  print(paste("Config file written to", out_path))
}
  
 
# generate config file
generate_config(include_id_in_paths = FALSE, out_dir = "config/")

# for loop though individual ids

data <- read.csv("C://Users/kuzelkova/Desktop/C/JKMNet/data/inputs/data_all_daily.csv")

ids <- unique(data$ID)

for (i in seq_along(ids)) {
  generate_config(
    id = ids[i],
    out_dir = "config",
    include_id_in_paths = TRUE,
    activation = "RELU",
    weight_init = "LHS",
    architecture  = c(4, 10, 4)
  )
} 
