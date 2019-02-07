library(neuralnet)
context("neuralnet_multiclass")

test_that("Manual multiclass returns correct dimensions", {
  nn_multi <- neuralnet((Species == "setosa") + (Species == "versicolor") ~ Petal.Length + Petal.Width, iris, 
                        hidden = c(2, 3), linear.output = FALSE)

  # Skip if not converged
  skip_if(is.null(nn_multi$weights))
  
  pred_multi <- predict(nn_multi, iris[, c("Petal.Length", "Petal.Width")])
  
  expect_length(nn_multi$weights[[1]], 3) # 2 hidden + 1 output layer
  expect_equal(dim(nn_multi$weights[[1]][[1]]), c(3, 2)) # First hidden layer: 3 in (1 bias) x 2 units
  expect_equal(dim(nn_multi$weights[[1]][[2]]), c(3, 3)) # Second hidden layer: 3 in (1 bias) x 3 units
  expect_equal(dim(nn_multi$weights[[1]][[3]]), c(4, 2)) # Output layer: 4 in (1 bias) x 2 units
  
  expect_equal(dim(pred_multi), c(nrow(iris), 2)) # Predict 2 outputs 
})

test_that("Auto multiclass returns correct dimensions", {
  nn_multi <- neuralnet(Species ~ Petal.Length + Petal.Width, iris, hidden = c(2, 3), linear.output = FALSE)
  
  # Skip if not converged
  skip_if(is.null(nn_multi$weights))
  
  pred_multi <- predict(nn_multi, iris[, c("Petal.Length", "Petal.Width")])
    
  expect_length(nn_multi$weights[[1]], 3) # 2 hidden + 1 output layer
  expect_equal(dim(nn_multi$weights[[1]][[1]]), c(3, 2)) # First hidden layer: 3 in (1 bias) x 2 units
  expect_equal(dim(nn_multi$weights[[1]][[2]]), c(3, 3)) # Second hidden layer: 3 in (1 bias) x 3 units
  expect_equal(dim(nn_multi$weights[[1]][[3]]), c(4, 3)) # Output layer: 4 in (1 bias) x 3 units
    
  expect_equal(dim(pred_multi), c(nrow(iris), 3)) # Predict 3 outputs
})


test_that("Same results with manual multiclass", {
  set.seed(42)
  nn_manual <- neuralnet((Species == "setosa") + (Species == "versicolor") + (Species == "virginica") ~ Petal.Length + Petal.Width, iris, linear.output = FALSE)
  
  set.seed(42)
  nn_auto <- neuralnet(Species ~ Petal.Length + Petal.Width, iris, linear.output = FALSE)
  
  expect_equal(nn_manual$weights[[1]][[1]], nn_auto$weights[[1]][[1]])
  expect_equal(nn_manual$weights[[1]][[2]], nn_auto$weights[[1]][[2]])
  expect_equal(nn_manual$net.result[[1]], nn_auto$net.result[[1]])
})

