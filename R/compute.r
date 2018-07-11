#' Computation of a given neural network for given covariate vectors
#' 
#' \code{compute}, a method for objects of class \code{nn}, typically produced
#' by \code{neuralnet}.  Computes the outputs of all neurons for specific
#' arbitrary covariate vectors given a trained neural network. Please make sure
#' that the order of the covariates is the same in the new matrix or dataframe
#' as in the original neural network.
#' 
#' 
#' @param x an object of class \code{nn}.
#' @param covariate a dataframe or matrix containing the variables that had
#' been used to train the neural network.
#' @param rep an integer indicating the neural network's repetition which
#' should be used.
#' @return \code{compute} returns a list containing the following components:
#' 
#' \item{neurons}{a list of the neurons' output for each layer of the neural
#' network.} \item{net.result}{a matrix containing the overall result of the
#' neural network.}
#' @author Stefan Fritsch, Frauke Guenther \email{guenther@@leibniz-bips.de}
#' @keywords neural
#' @examples
#' 
#' Var1 <- runif(50, 0, 100) 
#' sqrt.data <- data.frame(Var1, Sqrt=sqrt(Var1))
#' print(net.sqrt <- neuralnet(Sqrt~Var1,  sqrt.data, hidden=10, 
#'                   threshold=0.01))
#' compute(net.sqrt, (1:10)^2)$net.result
#' 
#' @export compute
compute <-
function (x, covariate, rep = 1) 
{
    nn <- x
    linear.output <- nn$linear.output
    weights <- nn$weights[[rep]]
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    weights <- unlist(weights)
    if (any(is.na(weights))) 
        weights[is.na(weights)] <- 0
    weights <- relist(weights, nrow.weights, ncol.weights)
    length.weights <- length(weights)
    covariate <- as.matrix(cbind(1, covariate))
    act.fct <- nn$act.fct
    neurons <- list(covariate)
    if (length.weights > 1) 
        for (i in 1:(length.weights - 1)) {
            temp <- neurons[[i]] %*% weights[[i]]
            act.temp <- act.fct(temp)
            neurons[[i + 1]] <- cbind(1, act.temp)
        }
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    if (linear.output) 
        net.result <- temp
    else net.result <- act.fct(temp)
    list(neurons = neurons, net.result = net.result)
}
