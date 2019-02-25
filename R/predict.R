
#' Neural network prediction
#' 
#' Prediction of artificial neural network of class \code{nn}, produced by \code{neuralnet()}. 
#' 
#' @param object Neural network of class \code{nn}.
#' @param newdata New data of class \code{data.frame} or \code{matrix}. 
#' @param rep Integer indicating the neural network's repetition which should be used.
#' @param all.units Return output for all units instead of final output only.
#' @param ... further arguments passed to or from other methods.
#'
#' @return Matrix of predictions. Each column represents one output unit. 
#' If \code{all.units=TRUE}, a list of matrices with output for each unit.
#'
#' @examples
#' library(neuralnet)
#' 
#' # Split data
#' train_idx <- sample(nrow(iris), 2/3 * nrow(iris))
#' iris_train <- iris[train_idx, ]
#' iris_test <- iris[-train_idx, ]
#' 
#' # Binary classification
#' nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris_train, linear.output = FALSE)
#' pred <- predict(nn, iris_test)
#' table(iris_test$Species == "setosa", pred[, 1] > 0.5)
#' 
#' # Multiclass classification
#' nn <- neuralnet((Species == "setosa") + (Species == "versicolor") + (Species == "virginica")
#'                  ~ Petal.Length + Petal.Width, iris_train, linear.output = FALSE)
#' pred <- predict(nn, iris_test)
#' table(iris_test$Species, apply(pred, 1, which.max))
#' 
#' @author Marvin N. Wright
#' @export
predict.nn <- function(object, newdata, rep = 1, all.units = FALSE, ...) {
  weights <- object$weights[[rep]]
  num_hidden_layers <- length(weights) - 1
  
  # Init prediction with data, subset if necessary
  if (ncol(newdata) == length(object$model.list$variables)) {
    pred <- as.matrix(newdata)
  } else {
    pred <- as.matrix(newdata[, object$model.list$variables])
  }
  
  
  # Init units if requested
  if (all.units) {
    units <- list(pred)
  }
  
  # Hidden layers
  if (num_hidden_layers > 0) {
    for (i in 1:num_hidden_layers) {
      pred <- object$act.fct(cbind(1, pred) %*% weights[[i]])
      
      # Save unit outputs if requested
      if (all.units) {
        units <- append(units, list(pred))
      }
    }
  }
  
  # Output layer: Only apply activation function if non-linear output
  pred <- cbind(1, pred) %*% weights[[num_hidden_layers + 1]]
  if (!object$linear.output) {
    pred <- object$output.act.fct(pred)
  }
  
  # Save unit outputs if requested
  if (all.units) {
    units <- append(units, list(pred))
  } 
  
  # Return result
  if (all.units) {
    units
  } else {
    pred
  }
}
  
