# Convert named activation functions in R functions, including derivatives 
convert.activation.function <- function(fun) {
  if (fun == "tanh") {
    fct <- function(x) {
      tanh(x)
    }
    attr(fct, "type") <- "tanh"
    deriv.fct <- function(x) {
      1 - x^2
    }
  } else if (fun == "logistic") {
    fct <- function(x) {
      1/(1 + exp(-x))
    }
    attr(fct, "type") <- "logistic"
    deriv.fct <- function(x) {
      x * (1 - x)
    }
  } else if (fun == "relu" || fun == "ReLu") {
    fct <- function(x) {
      x * (x > 0)
    }
    attr(fct, "type") <- "relu"
    deriv.fct <- function(x) {
      1 * (x > 0)
    }
  } else {
    stop("Unknown function.", call. = FALSE)
  }
  list(fct = fct, deriv.fct = deriv.fct)
}

# Convert named error functions in R functions, including derivatives 
convert.error.function <- function(fun) {
  if (fun == "sse") {
    fct <- function(x, y) {
      1/2 * (y - x)^2
    }
    attr(fct, "type") <- "sse"
    deriv.fct <- function(x, y) {
      x - y
    }
  } else if (fun == "ce") {
    fct <- function(x, y) {
      -(y * log(x) + (1 - y) * log(1 - x))
    }
    attr(fct, "type") <- "ce"
    deriv.fct <- function(x, y) {
      (1 - y)/(1 - x) - y/x
    }
  } else {
    stop("Unknown function.", call. = FALSE)
  }
  list(fct = fct, deriv.fct = deriv.fct)
}