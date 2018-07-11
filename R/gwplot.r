#' Plot method for generalized weights
#' 
#' \code{gwplot}, a method for objects of class \code{nn}, typically produced
#' by \code{neuralnet}.  Plots the generalized weights (Intrator and Intrator,
#' 1993) for one specific covariate and one response variable.
#' 
#' 
#' @param x an object of class \code{nn}
#' @param rep an integer indicating the repetition to plot. If rep="best", the
#' repetition with the smallest error will be plotted. If not stated all
#' repetitions will be plotted.
#' @param max maximum of the y axis. In default, max is set to the highest
#' y-value.
#' @param min minimum of the y axis. In default, min is set to the smallest
#' y-value.
#' @param file a character string naming the plot to write to. If not stated,
#' the plot will not be saved.
#' @param selected.covariate either a string of the covariate's name or an
#' integer of the ordered covariates, indicating the reference covariate in the
#' generalized weights plot. Defaulting to the first covariate.
#' @param selected.response either a string of the response variable's name or
#' an integer of the ordered response variables, indicating the reference
#' response in the generalized weights plot. Defaulting to the first response
#' variable.
#' @param highlight a logical value, indicating whether to highlight (red
#' color) the best repetition (smallest error). Only reasonable if rep=NULL.
#' Default is FALSE
#' @param type a character indicating the type of plotting; actually any of the
#' types as in \code{\link{plot.default}}.
#' @param col a color of the generalized weights.
#' @param \dots Arguments to be passed to methods, such as graphical parameters
#' (see \code{\link{par}}).
#' @author Stefan Fritsch, Frauke Guenther \email{guenther@@leibniz-bips.de}
#' @seealso \code{\link{neuralnet}}
#' @references Intrator O. and Intrator N. (1993) \emph{Using Neural Nets for
#' Interpretation of Nonlinear Models.} Proceedings of the Statistical
#' Computing Section, 244-249 San Francisco: American Statistical Society
#' (eds.)
#' @keywords neural
#' @examples
#' 
#' data(infert, package="datasets")
#' print(net.infert <- neuralnet(case~parity+induced+spontaneous, infert, 
#' 		                err.fct="ce", linear.output=FALSE, likelihood=TRUE))
#' gwplot(net.infert, selected.covariate="parity")
#' gwplot(net.infert, selected.covariate="induced")
#' gwplot(net.infert, selected.covariate="spontaneous")
#' 
#' @export gwplot
gwplot <-
function (x, rep = NULL, max = NULL, min = NULL, file = NULL, 
    selected.covariate = 1, selected.response = 1, highlight = FALSE, 
    type = "p", col = "black", ...) 
{
    net <- x
    if (is.null(net$generalized.weights)) 
        stop("generalized weights were not calculated", call. = F)
    if (!is.null(file)) {
        if (!is.character(file)) 
            stop("'file' must be a string")
        if (file.exists(file)) 
            stop(sprintf("%s already exists", sQuote(file)))
    }
    if (!is.numeric(selected.covariate)) 
        for (i in 1:length(net$model.list$variables)) if (net$model.list$variables[i] == 
            selected.covariate) 
            selected.covariate = i
    if (!is.numeric(selected.covariate) || selected.covariate < 
        1 || selected.covariate > ncol(net$covariate)) 
        stop("'selected.covariate' does not exist")
    if (!is.numeric(selected.response)) 
        for (i in 1:length(net$model.list$response)) if (net$model.list$response[i] == 
            selected.response) 
            selected.response = i
    if (!is.numeric(selected.response) || selected.response < 
        1 || selected.response > ncol(net$response)) 
        stop("'selected.response' does not exist")
    if (!is.null(rep)) {
        if (rep == "best") 
            rep <- as.integer(which.min(net$result.matrix["error", 
                ]))
        if (length(net$generalized.weights) < rep) 
            stop("'rep' does not exist")
    }
    covariate <- as.vector(net$covariate[, selected.covariate])
    variablename <- net$model.list$variables[selected.covariate]
    column <- (selected.response - 1) * ncol(net$covariate) + 
        selected.covariate
    if (is.null(rep)) {
        matrix <- as.matrix(sapply(net$generalized.weights, function(x) rbind(x[, 
            column])))
        item.to.print <- min(which.min(net$result.matrix["error", 
            ]))
    }
    else {
        highlight = F
        matrix <- as.matrix(net$generalized.weights[[rep]][, 
            column])
        item.to.print <- 1
    }
    if (is.null(max)) 
        max <- max(matrix)
    if (is.null(min)) 
        min <- min(matrix)
    ylim <- c(min, max)
    if (!highlight || item.to.print != 1 || ncol(matrix) == 1) 
        graphics::plot(x = covariate, y = matrix[, 1], ylim = ylim, xlab = variablename, 
            ylab = "GW", type = type, col = col, ...)
    else graphics::plot(x = covariate, y = matrix[, 2], ylim = ylim, xlab = variablename, 
        ylab = "GW", type = type, col = col, ...)
    if (ncol(matrix) >= 2) {
        for (i in 2:ncol(matrix)) if (!highlight || (i != item.to.print)) 
          graphics::lines(x = covariate, y = matrix[, i], type = type, 
                col = col, ...)
    }
    if (highlight) {
      graphics::lines(x = covariate, y = matrix[, item.to.print], type = type, 
            col = "red", ...)
      graphics::legend("topright", paste("Minimal Error: ", round(net$result.matrix["error", 
            item.to.print], 3), sep = ""), col = "red", ...)
    }
    graphics::title(paste("Response: ", net$model.list$response[selected.response], 
        sep = ""))
    if (!is.null(file)) {
        weight.plot <- grDevices::recordPlot()
        save(weight.plot, file = file)
    }
}
