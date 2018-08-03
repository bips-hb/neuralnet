print.nn <-
  function (x, ...) 
  {
    matrix <- x$result.matrix
    cat("Call: ", deparse(x$call), "\n\n", sep = "")
    if (!is.null(matrix)) {
      if (ncol(matrix) > 1) {
        cat(ncol(matrix), " repetitions were calculated.\n\n", 
            sep = "")
        sorted.matrix <- matrix[, order(matrix["error", ])]
        if (any(rownames(sorted.matrix) == "aic")) {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], AIC = sorted.matrix["aic", ], BIC = sorted.matrix["bic", 
                                                                                                   ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                                                                          ], Steps = sorted.matrix["steps", ])))
        }
        else {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                     ], Steps = sorted.matrix["steps", ])))
        }
      }
      else {
        cat(ncol(matrix), " repetition was calculated.\n\n", 
            sep = "")
        if (any(rownames(matrix) == "aic")) {
          print(t(matrix(c(matrix["error", ], matrix["aic", 
                                                     ], matrix["bic", ], matrix["reached.threshold", 
                                                                                ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                                                         "AIC", "BIC", "Reached Threshold", "Steps"), 
                                                                                                                       c(1)))))
        }
        else {
          print(t(matrix(c(matrix["error", ], matrix["reached.threshold", 
                                                     ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                              "Reached Threshold", "Steps"), c(1)))))
        }
      }
    }
    cat("\n")
  }