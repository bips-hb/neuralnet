library(neuralnet)
context("neuralnet")

nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, linear.output = FALSE)
nn2 <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, hidden = c(3, 2), linear.output = FALSE)

if (!is.null(nn$weights)) {
  pred <- predict(nn, iris[, c("Petal.Length", "Petal.Width")])
}
if (!is.null(nn2$weights)) {
  pred2 <- predict(nn2, iris[, c("Petal.Length", "Petal.Width")])
}

test_that("Fitting returns nn object with correct size", {
  skip_if(is.null(nn$weights))
  expect_is(nn, "nn")
  expect_length(nn, 14)
})

test_that("Prediction returns numeric with correct size", {
  skip_if(is.null(nn$weights))
  expect_is(pred, "matrix")
  expect_equal(dim(pred), c(nrow(iris), 1))
})

test_that("predict() function returns list of correct size for unit prediction", {
  skip_if(is.null(nn$weights))
  pred_all <- predict(nn, iris[, c("Petal.Length", "Petal.Width")], all.units = TRUE)
  expect_equal(length(pred_all), 3)

  skip_if(is.null(nn2$weights))
  pred_all2 <- predict(nn2, iris[, c("Petal.Length", "Petal.Width")], all.units = TRUE)
  expect_equal(length(pred_all2), 4)
})

test_that("predict() works if more variables in data", {
  skip_if(is.null(nn$weights))
  pred_all <- predict(nn, iris)
  expect_equal(dim(pred_all), c(nrow(iris), 1))
})

test_that("Custom activation function works", {
  expect_silent(neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, 
                          linear.output = FALSE, act.fct = function(x) log(1 + exp(x))))
  expect_silent(neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, 
                          linear.output = FALSE, act.fct = function(x) {log(1 + exp(x))}))
  
})

test_that("Same result with custom activation function", {
  set.seed(10)
  nn_custom <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, 
                         linear.output = FALSE, act.fct = function(x) 1/(1 + exp(-x)))
  
  set.seed(10)
  nn_custom2 <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, 
                          linear.output = FALSE, act.fct = function(x) {1/(1 + exp(-x))})
  
  set.seed(10)
  nn_default <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, 
                          linear.output = FALSE, act.fct = "logistic")
  
  expect_equal(nn_custom$net.result, nn_custom2$net.result)
  expect_equal(nn_custom$net.result, nn_default$net.result)
  
  expect_equal(nn_custom$result.matrix, nn_custom2$result.matrix)
  expect_equal(nn_custom$result.matrix, nn_default$result.matrix)
})

test_that("Error if 'ce' error function used in non-binary outcome", {
  expect_error(neuralnet(Sepal.Length ~ Petal.Length + Petal.Width, 
                         iris, linear.output = TRUE, err.fct = "ce"), 
               "Error function 'ce' only implemented for binary response\\.")
})

test_that("Dots in formula work", {
  set.seed(42)
  nn_dot <- neuralnet(Species ~ ., iris, linear.output = FALSE)
  
  set.seed(42)
  nn_nodot <- neuralnet(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, iris, linear.output = FALSE)
  
  expect_equal(nn_dot$weights[[1]][[1]], nn_nodot$weights[[1]][[1]])
  expect_equal(nn_dot$weights[[1]][[2]], nn_nodot$weights[[1]][[2]])
  expect_equal(nn_dot$net.result[[1]], nn_nodot$net.result[[1]])
})

test_that("Works with ReLu activation", {
  expect_silent(neuralnet(Species ~ ., iris, linear.output = FALSE, act.fct = "relu"))
  expect_silent(neuralnet(Species ~ ., iris, linear.output = FALSE, act.fct = "ReLu"))
})