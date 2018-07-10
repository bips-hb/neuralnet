\name{compute}
\alias{compute}
\title{ Computation of a given neural network for given covariate vectors}
\description{
 \code{compute}, a method for objects of class \code{nn}, typically
  produced by \code{neuralnet}. 
Computes the outputs of all neurons for specific arbitrary covariate vectors given a trained neural network. Please make sure that the order of the covariates is the same in the new matrix or dataframe as in the original neural network.
}
\usage{
compute(x, covariate, rep = 1)
}
\arguments{
  \item{x}{ an object of class \code{nn}. }
  \item{covariate}{ a dataframe or matrix containing the variables that had been used to train the neural network. }
  \item{rep}{ an integer indicating the neural network's repetition which should be used. }
}

\value{
 \code{compute} returns a list containing the following components:

  \item{neurons}{a list of the neurons' output for each layer of the neural network.}
  \item{net.result}{a matrix containing the overall result of the neural network.}
}

\author{ Stefan Fritsch, Frauke Guenther \email{guenther@leibniz-bips.de} }
\examples{
Var1 <- runif(50, 0, 100) 
sqrt.data <- data.frame(Var1, Sqrt=sqrt(Var1))
print(net.sqrt <- neuralnet(Sqrt~Var1,  sqrt.data, hidden=10, 
                  threshold=0.01))
compute(net.sqrt, (1:10)^2)$net.result
}
\keyword{ neural }

