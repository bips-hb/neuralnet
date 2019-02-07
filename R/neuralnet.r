#' Training of neural networks
#' 
#' Train neural networks using backpropagation,
#' resilient backpropagation (RPROP) with (Riedmiller, 1994) or without weight
#' backtracking (Riedmiller and Braun, 1993) or the modified globally
#' convergent version (GRPROP) by Anastasiadis et al. (2005). The function
#' allows flexible settings through custom-choice of error and activation
#' function. Furthermore, the calculation of generalized weights (Intrator O.
#' and Intrator N., 1993) is implemented.
#' 
#' The globally convergent algorithm is based on the resilient backpropagation
#' without weight backtracking and additionally modifies one learning rate,
#' either the learningrate associated with the smallest absolute gradient (sag)
#' or the smallest learningrate (slr) itself. The learning rates in the grprop
#' algorithm are limited to the boundaries defined in learningrate.limit.
#' 
#' @aliases neuralnet print.nn
#' @param formula a symbolic description of the model to be fitted.
#' @param data a data frame containing the variables specified in
#' \code{formula}.
#' @param hidden a vector of integers specifying the number of hidden neurons
#' (vertices) in each layer.
#' @param threshold a numeric value specifying the threshold for the partial
#' derivatives of the error function as stopping criteria.
#' @param stepmax the maximum steps for the training of the neural network.
#' Reaching this maximum leads to a stop of the neural network's training
#' process.
#' @param rep the number of repetitions for the neural network's training.
#' @param startweights a vector containing starting values for the weights. 
#' Set to \code{NULL} for random initialization. 
#' @param learningrate.limit a vector or a list containing the lowest and
#' highest limit for the learning rate. Used only for RPROP and GRPROP.
#' @param learningrate.factor a vector or a list containing the multiplication
#' factors for the upper and lower learning rate. Used only for RPROP and
#' GRPROP.
#' @param learningrate a numeric value specifying the learning rate used by
#' traditional backpropagation. Used only for traditional backpropagation.
#' @param lifesign a string specifying how much the function will print during
#' the calculation of the neural network. 'none', 'minimal' or 'full'.
#' @param lifesign.step an integer specifying the stepsize to print the minimal
#' threshold in full lifesign mode.
#' @param algorithm a string containing the algorithm type to calculate the
#' neural network. The following types are possible: 'backprop', 'rprop+',
#' 'rprop-', 'sag', or 'slr'. 'backprop' refers to backpropagation, 'rprop+'
#' and 'rprop-' refer to the resilient backpropagation with and without weight
#' backtracking, while 'sag' and 'slr' induce the usage of the modified
#' globally convergent algorithm (grprop). See Details for more information.
#' @param err.fct a differentiable function that is used for the calculation of
#' the error. Alternatively, the strings 'sse' and 'ce' which stand for the sum
#' of squared errors and the cross-entropy can be used.
#' @param act.fct a differentiable function that is used for smoothing the
#' result of the cross product of the covariate or neurons and the weights.
#' Additionally the strings, 'logistic' and 'tanh' are possible for the
#' logistic function and tangent hyperbolicus.
#' @param linear.output logical. If act.fct should not be applied to the output
#' neurons set linear output to TRUE, otherwise to FALSE.
#' @param exclude a vector or a matrix specifying the weights, that are
#' excluded from the calculation. If given as a vector, the exact positions of
#' the weights must be known. A matrix with n-rows and 3 columns will exclude n
#' weights, where the first column stands for the layer, the second column for
#' the input neuron and the third column for the output neuron of the weight.
#' @param constant.weights a vector specifying the values of the weights that
#' are excluded from the training process and treated as fix.
#' @param likelihood logical. If the error function is equal to the negative
#' log-likelihood function, the information criteria AIC and BIC will be
#' calculated. Furthermore the usage of confidence.interval is meaningfull.
#' 
#' @return \code{neuralnet} returns an object of class \code{nn}.  An object of
#' class \code{nn} is a list containing at most the following components:
#' 
#' \item{ call }{ the matched call. } 
#' \item{ response }{ extracted from the \code{data argument}.  } 
#' \item{ covariate }{ the variables extracted from the \code{data argument}. } 
#' \item{ model.list }{ a list containing the covariates and the response variables extracted from the \code{formula argument}. } 
#' \item{ err.fct }{ the error function. } 
#' \item{ act.fct }{ the activation function. } 
#' \item{ data }{ the \code{data argument}.} 
#' \item{ net.result }{ a list containing the overall result of the neural network for every repetition.} 
#' \item{ weights }{ a list containing the fitted weights of the neural network for every repetition. } 
#' \item{ generalized.weights }{ a list containing the generalized weights of the neural network for every repetition. } 
#' \item{ result.matrix }{ a matrix containing the reached threshold, needed steps, error, AIC and BIC (if computed) and weights for every repetition. Each column represents one repetition. } 
#' \item{ startweights }{ a list containing the startweights of the neural network for every repetition. }
#' 
#' @author Stefan Fritsch, Frauke Guenther, Marvin N. Wright
#' 
#' @seealso \code{\link{plot.nn}} for plotting the neural network.
#' 
#' \code{\link{gwplot}} for plotting the generalized weights.
#' 
#' \code{\link{predict.nn}} for computation of a given neural network for given
#' covariate vectors (formerly \code{compute}).
#' 
#' \code{\link{confidence.interval}} for calculation of confidence intervals of
#' the weights.
#' 
#' \code{\link{prediction}} for a summary of the output of the neural network.
#' 
#' @references Riedmiller M. (1994) \emph{Rprop - Description and
#' Implementation Details.} Technical Report. University of Karlsruhe.
#' 
#' Riedmiller M. and Braun H. (1993) \emph{A direct adaptive method for faster
#' backpropagation learning: The RPROP algorithm.} Proceedings of the IEEE
#' International Conference on Neural Networks (ICNN), pages 586-591.  San
#' Francisco.
#' 
#' Anastasiadis A. et. al. (2005) \emph{New globally convergent training scheme
#' based on the resilient propagation algorithm.} Neurocomputing 64, pages
#' 253-270.
#' 
#' Intrator O. and Intrator N. (1993) \emph{Using Neural Nets for
#' Interpretation of Nonlinear Models.} Proceedings of the Statistical
#' Computing Section, 244-249 San Francisco: American Statistical Society
#' (eds).
#' @keywords neural
#' @examples
#' 
#' library(neuralnet)
#' 
#' # Binary classification
#' nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, linear.output = FALSE)
#' \dontrun{print(nn)}
#' \dontrun{plot(nn)}
#' 
#' # Multiclass classification
#' nn <- neuralnet(Species ~ Petal.Length + Petal.Width, iris, linear.output = FALSE)
#' \dontrun{print(nn)}
#' \dontrun{plot(nn)}
#' 
#' # Custom activation function
#' softplus <- function(x) log(1 + exp(x))
#' nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width, iris, 
#'                 linear.output = FALSE, hidden = c(3, 2), act.fct = softplus)
#' \dontrun{print(nn)}
#' \dontrun{plot(nn)}
#' 
#' @import stats 
#' @export
neuralnet <-
  function (formula, data, hidden = 1, threshold = 0.01, stepmax = 1e+05,
            rep = 1, startweights = NULL, learningrate.limit = NULL,
            learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
            lifesign = "none", lifesign.step = 1000, algorithm = "rprop+", 
            err.fct = "sse", act.fct = "logistic", linear.output = TRUE,
            exclude = NULL, constant.weights = NULL, likelihood = FALSE) {
    
  # Save call
  call <- match.call()
  
  # Check arguments
  if (is.null(data)) {
    stop("Missing 'data' argument.", call. = FALSE)
  }
  data <- as.data.frame(data)
  
  if (is.null(formula)) {
    stop("Missing 'formula' argument.", call. = FALSE)
  }
  formula <- stats::as.formula(formula)
  
  # Learning rate limit
  if (!is.null(learningrate.limit)) {
    if (length(learningrate.limit) != 2) {
      stop("Argument 'learningrate.factor' must consist of two components.", 
           call. = FALSE)
    }
    learningrate.limit <- as.list(learningrate.limit)
    names(learningrate.limit) <- c("min", "max")
    
    if (is.na(learningrate.limit$min) || is.na(learningrate.limit$max)) {
      stop("'learningrate.limit' must be a numeric vector", 
           call. = FALSE)
    }
  } else {
    learningrate.limit <- list(min = 1e-10, max = 0.1)
  }
  
  # Learning rate factor
  if (!is.null(learningrate.factor)) {
    if (length(learningrate.factor) != 2) {
      stop("Argument 'learningrate.factor' must consist of two components.", 
           call. = FALSE)
    }
    learningrate.factor <- as.list(learningrate.factor)
    names(learningrate.factor) <- c("minus", "plus")

    if (is.na(learningrate.factor$minus) || is.na(learningrate.factor$plus)) {
      stop("'learningrate.factor' must be a numeric vector", 
           call. = FALSE)
    } 
  } else {
    learningrate.factor <- list(minus = 0.5, plus = 1.2)
  }
  
  # Learning rate (backprop)
  if (algorithm == "backprop") {
    if (is.null(learningrate) || !is.numeric(learningrate)) {
      stop("Argument 'learningrate' must be a numeric value, if the backpropagation algorithm is used.", 
           call. = FALSE)
    }
  }
  
  # TODO: Rename?
  # Lifesign
  if (!(lifesign %in% c("none", "minimal", "full"))) {
    stop("Argument 'lifesign' must be one of 'none', 'minimal', 'full'.", call. = FALSE)
  }

  # Algorithm
  if (!(algorithm %in% c("rprop+", "rprop-", "slr", "sag", "backprop"))) {
    stop("Unknown algorithm.", call. = FALSE)
  }
  
  # Threshold
  if (is.na(threshold)) {
    stop("Argument 'threshold' must be a numeric value.", call. = FALSE)
  }
    
  # Hidden units
  if (any(is.na(hidden))) {
    stop("Argument 'hidden' must be an integer vector or a single integer.", 
         call. = FALSE)
  } 
  if (length(hidden) > 1 && any(hidden == 0)) {
    stop("Argument 'hidden' contains at least one 0.", call. = FALSE)
  }
    
  # Replications
  if (is.na(rep)) {
    stop("Argument 'rep' must be an integer", call. = FALSE)
  }
    
  # Max steps
  if (is.na(stepmax)) {
    stop("Argument 'stepmax' must be an integer", call. = FALSE)
  }
    
  # Activation function
  if (!(is.function(act.fct) || act.fct %in% c("logistic", "tanh"))) {
    stop("Unknown activation function.", call. = FALSE)
  }
  
  # Error function
  if (!(is.function(err.fct) || err.fct %in% c("sse", "ce"))) {
    stop("Unknown error function.", call. = FALSE)
  }
  
  # Formula interface
  model.list <- list(response = attr(terms(as.formula(call("~", formula[[2]]))), "term.labels"), 
                     variables = attr(terms(formula, data = data), "term.labels"))
  response <- as.matrix(model.frame(as.formula(call("~", formula[[2]])), data))
  covariate <- cbind(intercept = 1, as.matrix(data[, model.list$variables]))
  
  # Multiclass response
  if (is.character(response)) {
    class.names <- unique(response[, 1])
    response <- model.matrix( ~ response[,1]-1) == 1
    colnames(response) <- class.names
    model.list$response <- class.names
  }
  
  # Activation function
  if (is.function(act.fct)) {
    act.deriv.fct <- Deriv::Deriv(act.fct)
    attr(act.fct, "type") <- "function"
  } else {
    converted.fct <- convert.activation.function(act.fct)
    act.fct <- converted.fct$fct
    act.deriv.fct <- converted.fct$deriv.fct
  }
  
  # Error function
  if (is.function(err.fct)) {
    attr(err.fct, "type") <- "function"
    err.deriv.fct <- Deriv::Deriv(err.fct)
  } else {
    converted.fct <- convert.error.function(err.fct)
    err.fct <- converted.fct$fct
    err.deriv.fct <- converted.fct$deriv.fct
  }
  
  if (attr(err.fct, "type") == "ce" && !all(response %in% 0:1)) {
    stop("Error function 'ce' only implemented for binary response.", call. = FALSE)
  } 
  
  # Fit network for each replication
  list.result <- lapply(1:rep, function(i) {
    # Show progress
    if (lifesign != "none") {
      lifesign <- display(hidden, threshold, rep, i, lifesign)
    }
    
    # Fit network
    calculate.neuralnet(learningrate.limit = learningrate.limit, 
                        learningrate.factor = learningrate.factor, covariate = covariate, 
                        response = response, data = data, model.list = model.list, 
                        threshold = threshold, lifesign.step = lifesign.step, 
                        stepmax = stepmax, hidden = hidden, lifesign = lifesign, 
                        startweights = startweights, algorithm = algorithm, 
                        err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                        act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                        rep = i, linear.output = linear.output, exclude = exclude, 
                        constant.weights = constant.weights, likelihood = likelihood, 
                        learningrate.bp = learningrate)
  })
  matrix <- sapply(list.result, function(x) {x$output.vector})
  if (all(sapply(matrix, is.null))) {
    list.result <- NULL
    matrix <- NULL
    ncol.matrix <- 0
  } else {
    ncol.matrix <- ncol(matrix)
  }
  
  # Warning if some replications did not converge
  if (ncol.matrix < rep) {
    warning(sprintf("Algorithm did not converge in %s of %s repetition(s) within the stepmax.", 
                    (rep - ncol.matrix), rep), call. = FALSE)
  }
      
  # Return output
  generate.output(covariate, call, rep, threshold, matrix, 
      startweights, model.list, response, err.fct, act.fct, 
      data, list.result, linear.output, exclude)
}

# Display output of replication
display <- function (hidden, threshold, rep, i.rep, lifesign) {
  message("hidden: ", paste(hidden, collapse = ", "), "    thresh: ", 
          threshold, "    rep: ", strrep(" ", nchar(rep) - nchar(i.rep)), 
          i.rep, "/", rep, "    steps: ", appendLF = FALSE)
  utils::flush.console()
  
  if (lifesign == "full") {
    lifesign <- sum(nchar(hidden)) + 2 * length(hidden) - 
      2 + max(nchar(threshold)) + 2 * nchar(rep) + 41
  }
  return(lifesign)
}

# Generate output object
generate.output <- function(covariate, call, rep, threshold, matrix, startweights, 
                            model.list, response, err.fct, act.fct, data, list.result, 
                            linear.output, exclude) {
  
  nn <- list(call = call, response = response, covariate = covariate[, -1, drop = FALSE], 
             model.list = model.list, err.fct = err.fct, act.fct = act.fct, 
             linear.output = linear.output, data = data, exclude = exclude)
  
  if (!is.null(matrix)) {
    nn$net.result <- lapply(list.result, function(x) {x$net.result})
    nn$weights <- lapply(list.result, function(x) {x$weights})
    nn$generalized.weights <- lapply(list.result, function(x) {x$generalized.weights})
    nn$startweights <- lapply(list.result, function(x) {x$startweights})
    nn$result.matrix <- matrix
    rownames(nn$result.matrix) <- c(rownames(matrix)[rownames(matrix) != ""], 
                                    get_weight_names(nn$weights[[1]], model.list))
  }
  
  class(nn) <- c("nn")
  return(nn)
}

# Get names of all weights in network
get_weight_names <- function(weights, model.list) {
  # All hidden unit names
  if (length(weights) > 1) {
    hidden_units <- lapply(1:(length(weights) - 1), function(i) {
      paste0(i, "layhid", 1:ncol(weights[[i]]))
    })
  } else {
    hidden_units <- list()
  }
  
  # All unit names including input and output
  units <- c(list(model.list$variables),
             hidden_units, 
             list(model.list$response))
  
  # Combine each layer with the next, add intercept
  weight_names <- do.call(c, lapply(1:(length(units) - 1), function(i) {
    as.vector(outer(c("Intercept", units[[i]]), units[[i + 1]], paste, sep = ".to."))
  }))
 return(weight_names) 
}
