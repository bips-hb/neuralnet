#' Summarizes the output of the neural network, the data and the fitted values
#' of glm objects (if available)
#' 
#' \code{prediction}, a method for objects of class \code{nn}, typically
#' produced by \code{neuralnet}.  In a first step, the dataframe will be
#' amended by a mean response, the mean of all responses corresponding to the
#' same covariate-vector.  The calculated data.error is the error function
#' between the original response and the new mean response.  In a second step,
#' all duplicate rows will be erased to get a quick overview of the data.  To
#' obtain an overview of the results of the neural network and the glm objects,
#' the covariate matrix will be bound to the output of the neural network and
#' the fitted values of the glm object(if available) and will be reduced by all
#' duplicate rows.
#' 
#' 
#' @param x neural network
#' @param list.glm an optional list of glm objects
#' @return a list of the summaries of the repetitions of the neural networks,
#' the data and the glm objects (if available).
#' @author Stefan Fritsch, Frauke Guenther \email{guenther@@leibniz-bips.de}
#' @seealso \code{\link{neuralnet}}
#' @keywords neural
#' @examples
#' 
#' Var1 <- rpois(100,0.5)
#' Var2 <- rbinom(100,2,0.6)
#' Var3 <- rbinom(100,1,0.5)
#' SUM <- as.integer(abs(Var1+Var2+Var3+(rnorm(100))))
#' sum.data <- data.frame(Var1+Var2+Var3, SUM)
#' print(net.sum <- neuralnet( SUM~Var1+Var2+Var3,  sum.data, hidden=1, 
#'                  act.fct="tanh"))
#' main <- glm(SUM~Var1+Var2+Var3, sum.data, family=poisson())
#' full <- glm(SUM~Var1*Var2*Var3, sum.data, family=poisson())
#' prediction(net.sum, list.glm=list(main=main, full=full))
#' 
#' @export prediction
prediction <-
function (x, list.glm = NULL) 
{
    nn <- x
    data.result <- calculate.data.result(response = nn$response, 
        model.list = nn$model.list, covariate = nn$covariate)
    predictions <- calculate.predictions(covariate = nn$covariate, 
        data.result = data.result, list.glm = list.glm, matrix = nn$result.matrix, 
        list.net.result = nn$net.result, model.list = nn$model.list)
    if (type(nn$err.fct) == "ce" && all(data.result >= 0) && 
        all(data.result <= 1)) 
        data.error <- sum(nn$err.fct(data.result, nn$response), 
            na.rm = T)
    else data.error <- sum(nn$err.fct(data.result, nn$response))
    cat("Data Error:\t", data.error, ";\n", sep = "")
    predictions
}
calculate.predictions <-
function (covariate, data.result, list.glm, matrix, list.net.result, 
    model.list) 
{
    not.duplicated <- !duplicated(covariate)
    nrow.notdupl <- sum(not.duplicated)
    covariate.mod <- matrix(covariate[not.duplicated, ], nrow = nrow.notdupl)
    predictions <- list(data = cbind(covariate.mod, matrix(data.result[not.duplicated, 
        ], nrow = nrow.notdupl)))
    if (!is.null(matrix)) {
        for (i in length(list.net.result):1) {
            pred.temp <- cbind(covariate.mod, matrix(list.net.result[[i]][not.duplicated, 
                ], nrow = nrow.notdupl))
            predictions <- eval(parse(text = paste("c(list(rep", 
                i, "=pred.temp), predictions)", sep = "")))
        }
    }
    if (!is.null(list.glm)) {
        for (i in 1:length(list.glm)) {
            pred.temp <- cbind(covariate.mod, matrix(list.glm[[i]]$fitted.values[not.duplicated], 
                nrow = nrow.notdupl))
            text <- paste("c(predictions, list(glm.", names(list.glm[i]), 
                "=pred.temp))", sep = "")
            predictions <- eval(parse(text = text))
        }
    }
    for (i in 1:length(predictions)) {
        colnames(predictions[[i]]) <- c(model.list$variables, 
            model.list$response)
        if (nrow(covariate) > 1) 
            for (j in (1:ncol(covariate))) predictions[[i]] <- predictions[[i]][order(predictions[[i]][, 
                j]), ]
        rownames(predictions[[i]]) <- 1:nrow(predictions[[i]])
    }
    predictions
}
calculate.data.result <-
function (response, covariate, model.list) 
{
    duplicated <- duplicated(covariate)
    if (!any(duplicated)) {
        return(response)
    }
    which.duplicated <- seq_along(duplicated)[duplicated]
    which.not.duplicated <- seq_along(duplicated)[!duplicated]
    ncol.response <- ncol(response)
    if (ncol(covariate) == 1) {
        for (each in which.not.duplicated) {
            out <- NULL
            if (length(which.duplicated) > 0) {
                out <- covariate[which.duplicated, ] == covariate[each, 
                  ]
                if (any(out)) {
                  rows <- c(each, which.duplicated[out])
                  response[rows, ] = matrix(colMeans(matrix(response[rows, 
                    ], ncol = ncol.response)), ncol = ncol.response, 
                    nrow = length(rows), byrow = T)
                  which.duplicated <- which.duplicated[-out]
                }
            }
        }
    }
    else {
        tcovariate <- t(covariate)
        for (each in which.not.duplicated) {
            out <- NULL
            if (length(which.duplicated) > 0) {
                out <- apply(tcovariate[, which.duplicated] == 
                  covariate[each, ], 2, FUN = all)
                if (any(out)) {
                  rows <- c(each, which.duplicated[out])
                  response[rows, ] = matrix(colMeans(matrix(response[rows, 
                    ], ncol = ncol.response)), ncol = ncol.response, 
                    nrow = length(rows), byrow = T)
                  which.duplicated <- which.duplicated[-out]
                }
            }
        }
    }
    response
}
