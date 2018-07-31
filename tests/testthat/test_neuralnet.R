library(neuralnet)
context("neuralnet")

nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, linear.output = FALSE)
nn2 <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, hidden = c(3, 2), linear.output = FALSE)
pred <- predict(nn, iris[, c("Petal.Length", "Petal.Width")])
pred2 <- predict(nn2, iris[, c("Petal.Length", "Petal.Width")])

test_that("Fitting returns nn object with correct size", {
  expect_is(nn, "nn")
  expect_length(nn, 13)
})

test_that("Prediction returns numeric with correct size", {
  expect_is(pred, "matrix")
  expect_equal(dim(pred), c(nrow(iris), 1))
})

test_that("Prediction is about right", {
  set.seed(100)
  expect_true(mean(abs(pred[, 1] - (iris$Species == "setosa"))) <= .1)
})

test_that("predict() function returns list of correct size for unit prediction", {
  pred_all <- predict(nn, iris[, c("Petal.Length", "Petal.Width")], all.units = TRUE)
  expect_equal(length(pred_all), 3)

  pred_all2 <- predict(nn2, iris[, c("Petal.Length", "Petal.Width")], all.units = TRUE)
  expect_equal(length(pred_all2), 4)
})

test_that("Multiclass returns correct dimensions", {
  nn_multi <- neuralnet((Species == "setosa") + (Species == "versicolor") ~ Petal.Length + Petal.Width, iris, 
                        hidden = c(2, 3), linear.output = FALSE)
  pred_multi <- predict(nn_multi, iris[, c("Petal.Length", "Petal.Width")])
  
  expect_length(nn_multi$weights[[1]], 3) # 2 hidden + 1 output layer
  expect_equal(dim(nn_multi$weights[[1]][[1]]), c(3, 2)) # First hidden layer: 3 in (1 bias) x 2 units
  expect_equal(dim(nn_multi$weights[[1]][[2]]), c(3, 3)) # Second hidden layer: 3 in (1 bias) x 3 units
  expect_equal(dim(nn_multi$weights[[1]][[3]]), c(4, 2)) # Output layer: 4 in (1 bias) x 2 units
  
  expect_equal(dim(pred_multi), c(nrow(iris), 2)) # Predict 2 outputs
})

test_that("predict() works if more variables in data", {
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


