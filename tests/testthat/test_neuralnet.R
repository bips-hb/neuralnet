library(neuralnet)
context("neuralnet")

nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris)
pred <- compute(nn, iris[, c("Petal.Length", "Petal.Width")])

test_that("Fitting returns nn object with correct size", {
  expect_is(nn, "nn")
  expect_length(nn, 13)
})

test_that("Prediction returns list with correct size", {
  expect_is(pred, "list")
  expect_length(pred, 2)
  expect_length(pred$net.result, nrow(iris))
})

test_that("Prediction is about right", {
  expect_true(all(abs(pred$net.result[, 1] - (iris$Species == "setosa")) <= .1))
})

