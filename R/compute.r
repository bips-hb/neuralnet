
#' Deprecated function
#' 
#' The function \code{compute} is deprecated. Please refer to the new function \code{\link{predict.nn}}.
#' 
#' @param x an object of class \code{nn}.
#' @param covariate a dataframe or matrix containing the variables that had
#' been used to train the neural network.
#' @param rep an integer indicating the neural network's repetition which
#' should be used.
#' @return \code{compute} returns a list containing the following components:
#' \item{neurons}{a list of the neurons' output for each layer of the neural
#' network.} \item{net.result}{a matrix containing the overall result of the
#' neural network.}
#' 
#' @export
compute <- function(x, covariate, rep = 1) {
  .Deprecated("predict", package = "neuralnet")
  pred <- predict.nn(x, newdata = covariate, rep = rep, all.units = TRUE)
  
  # Create old format by adding intercept
  for (i in 1:(length(pred) - 1)) {
    pred[[i]] <- cbind(1, pred[[i]])
  }
  
  list(neurons = pred[-length(pred)], net.result = pred[[length(pred)]])
}
