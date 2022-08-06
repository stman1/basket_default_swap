library(MASS)
library(ks)


data(iris)
Fhat <- kcde(iris[, 1])
predict(Fhat, x=as.matrix(iris[, 1]))

plot(Fhat)