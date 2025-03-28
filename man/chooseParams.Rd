% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/main.R
\name{chooseParams}
\alias{chooseParams}
\title{Select the parameters in COAP models}
\usage{
chooseParams(
  X_count,
  Adj_sp,
  H,
  Z = matrix(1, nrow(X_count), 1),
  offset = rep(0, nrow(X_count)),
  q_max = 15,
  r_max = 24,
  threshold = c(0.1, 0.01),
  verbose = TRUE,
  ...
)
}
\arguments{
\item{X_count}{a count matrix, the observed count matrix with shape n-by-p.}

\item{Adj_sp}{a sparse matrix, the weighted adjacency matrix;}

\item{H}{a n-by-d matrix, the covariate matrix with low-rank regression coefficient matrix;}

\item{Z}{an optional matrix, the fixed-dimensional covariate matrix with control variables; default as a full-one column vector if there is no additional covariates.}

\item{offset}{an optional vector, the offset for each unit; default as full-zero vector.}

\item{q_max}{an optional string, specify the upper bound for the number of factors; default as 15.}

\item{r_max}{an optional integer, specify the upper bound for the rank of the regression coefficient matrix; default as 24.}

\item{threshold}{an optional 2-dimensional positive vector, specify the the thresholds that filters the singular values of beta and B, respectively.}

\item{verbose}{a logical value, whether output the information in iteration.}

\item{..., }{other arguments passed to the function \code{\link{SpaCOAP}}.}
}
\value{
return a named vector with names `hr` and `hq`, the estimated rank and number of factors.
}
\description{
Select the number of factors and the rank of coefficient matrix in the covariate-augmented overdispersed Poisson factor model
}
\details{
The threshold is to filter the singular values with  low signal, to assist the identification of underlying model structure.
}
\examples{
width <- 20; height <- 15; p <- 300
d <- 20; k <- 3; q <- 6; r <- 3
datlist <- gendata_spacoap(width=width, height=height, p=p, d=d, k=k, q=q, rank0=r)
set.seed(1)
para_vec <- chooseParams(X_count=datlist$X, Adj_sp=datlist$Adj_sp,
 H= datlist$H, Z = datlist$Z, r_max=6)
print(para_vec)
}
\references{
None
}
\seealso{
\code{\link{SpaCOAP}}
}
