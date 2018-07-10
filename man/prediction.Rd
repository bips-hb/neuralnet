\name{prediction}
\alias{prediction}
\title{ Summarizes the output of the neural network, the data and the fitted values of glm objects (if available) }
         
\description{
 \code{prediction}, a method for objects of class \code{nn}, typically produced by \code{neuralnet}. 
 In a first step, the dataframe will be amended by a mean response, the mean of all responses corresponding to the same covariate-vector. 
 The calculated data.error is the error function between the original response and the new mean response.
 In a second step, all duplicate rows will be erased to get a quick overview of the data. 
 To obtain an overview of the results of the neural network and the glm objects, the covariate matrix will be bound to the output of the neural network and the fitted values of the glm object(if available) and will be reduced by all duplicate rows.

}
\usage{
prediction(x, list.glm = NULL)
}
\arguments{
  \item{x}{ neural network }
  \item{list.glm}{ an optional list of glm objects }
}

\value{
  a list of the summaries of the repetitions of the neural networks, the data and the glm objects (if available).
}
\author{ Stefan Fritsch, Frauke Guenther \email{guenther@leibniz-bips.de} }
\seealso{\code{\link{neuralnet}}}
\examples{
Var1 <- rpois(100,0.5)
Var2 <- rbinom(100,2,0.6)
Var3 <- rbinom(100,1,0.5)
SUM <- as.integer(abs(Var1+Var2+Var3+(rnorm(100))))
sum.data <- data.frame(Var1+Var2+Var3, SUM)
print(net.sum <- neuralnet( SUM~Var1+Var2+Var3,  sum.data, hidden=1, 
                 act.fct="tanh"))
main <- glm(SUM~Var1+Var2+Var3, sum.data, family=poisson())
full <- glm(SUM~Var1*Var2*Var3, sum.data, family=poisson())
prediction(net.sum, list.glm=list(main=main, full=full))
}
\keyword{ neural }
