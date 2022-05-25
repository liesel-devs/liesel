#!/usr/bin/env Rscript

format_chunk <- function(chunk) {
  chunk <- system2("pyupgrade", c("--py310-plus", "-"),
                   stdout = TRUE, input = chunk)

  chunk <- system2("isort", c("--profile black", "--quiet", "-"),
                   stdout = TRUE, input = chunk)

  chunk <- system2("black", c("--quiet", "-"),
                   stdout = TRUE, input = chunk)

  chunk
}

format_file <- function(file) {
  input <- readLines(file)
  output <- character()

  chunk <- character()
  in_chunk <- FALSE

  for (line in input) {
    in_closing_fence <- grepl("^```$", line)

    if (in_chunk && in_closing_fence) {
      chunk <- format_chunk(chunk)
      output <- c(output, chunk)

      chunk <- character()
      in_chunk <- FALSE
    }

    if (in_chunk) {
      chunk <- c(chunk, line)
    } else {
      output <- c(output, line)
    }

    in_opening_fence <- grepl("^```\\{python.*\\}$", line)

    if (!in_chunk && in_opening_fence) {
      in_chunk <- TRUE
    }
  }

  writeLines(output, file)

  invisible(NULL)
}

args <- commandArgs(trailingOnly = TRUE)
for (arg in args) format_file(arg)
