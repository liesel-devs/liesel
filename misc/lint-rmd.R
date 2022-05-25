#!/usr/bin/env Rscript

lint_file <- function(file) {
  input <- readLines(file)
  output <- character()

  in_chunk <- FALSE

  for (line in input) {
    in_closing_fence <- grepl("^```$", line)

    if (in_chunk && in_closing_fence) {
      in_chunk <- FALSE
    }

    if (in_chunk) {
      output <- c(output, line)
    } else {
      output <- c(output, "")
    }

    in_opening_fence <- grepl("^```\\{python.*\\}$", line)

    if (!in_chunk && in_opening_fence) {
      in_chunk <- TRUE
    }
  }

  system2("flake8", c("--ignore=E303,W391", "-"), input = output)

  invisible(NULL)
}

args <- commandArgs(trailingOnly = TRUE)
for (arg in args) lint_file(arg)
