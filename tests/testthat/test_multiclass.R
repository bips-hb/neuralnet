library(neuralnet)
context("neuralnet_multiclass")

# Scale iris data
dat <- iris
dat[, -5] <- scale(dat[, -5])

test_that("Manual multiclass returns correct dimensions", {
  nn_multi <- neuralnet((Species == "setosa") + (Species == "versicolor") ~ Petal.Length + Petal.Width, dat, 
                        hidden = c(2, 3), linear.output = FALSE)

  # Skip if not converged
  skip_if(is.null(nn_multi$weights))
  
  pred_multi <- predict(nn_multi, dat[, c("Petal.Length", "Petal.Width")])
  
  expect_length(nn_multi$weights[[1]], 3) # 2 hidden + 1 output layer
  expect_equal(dim(nn_multi$weights[[1]][[1]]), c(3, 2)) # First hidden layer: 3 in (1 bias) x 2 units
  expect_equal(dim(nn_multi$weights[[1]][[2]]), c(3, 3)) # Second hidden layer: 3 in (1 bias) x 3 units
  expect_equal(dim(nn_multi$weights[[1]][[3]]), c(4, 2)) # Output layer: 4 in (1 bias) x 2 units
  
  expect_equal(dim(pred_multi), c(nrow(dat), 2)) # Predict 2 outputs 
})

test_that("Auto multiclass returns correct dimensions", {
  nn_multi <- neuralnet(Species ~ Petal.Length + Petal.Width, dat, hidden = c(2, 3), linear.output = FALSE)
  
  # Skip if not converged
  skip_if(is.null(nn_multi$weights))
  
  pred_multi <- predict(nn_multi, dat[, c("Petal.Length", "Petal.Width")])
    
  expect_length(nn_multi$weights[[1]], 3) # 2 hidden + 1 output layer
  expect_equal(dim(nn_multi$weights[[1]][[1]]), c(3, 2)) # First hidden layer: 3 in (1 bias) x 2 units
  expect_equal(dim(nn_multi$weights[[1]][[2]]), c(3, 3)) # Second hidden layer: 3 in (1 bias) x 3 units
  expect_equal(dim(nn_multi$weights[[1]][[3]]), c(4, 3)) # Output layer: 4 in (1 bias) x 3 units
    
  expect_equal(dim(pred_multi), c(nrow(dat), 3)) # Predict 3 outputs
})


test_that("Same results with manual multiclass", {
  set.seed(42)
  nn_manual <- neuralnet((Species == "setosa") + (Species == "versicolor") + (Species == "virginica") ~ Petal.Length + Petal.Width, dat, linear.output = FALSE)
  
  set.seed(42)
  nn_auto <- neuralnet(Species ~ Petal.Length + Petal.Width, dat, linear.output = FALSE)
  
  expect_equal(nn_manual$weights[[1]][[1]], nn_auto$weights[[1]][[1]])
  expect_equal(nn_manual$weights[[1]][[2]], nn_auto$weights[[1]][[2]])
  expect_equal(nn_manual$net.result[[1]], nn_auto$net.result[[1]])
})

test_that("Response vector has correct factor levels", {
  nn <- neuralnet(Species ~ Petal.Length + Petal.Width, dat, linear.output = FALSE)
  expect_equal(unname(apply(nn$response, 1, which)), as.numeric(dat$Species))
  
  dat_char <- dat
  dat_char$Species <- as.character(dat_char$Species)
  nn <- neuralnet(Species ~ Petal.Length + Petal.Width, dat_char, linear.output = FALSE)
  expect_equal(colnames(nn$response)[apply(nn$response, 1, which)], dat_char$Species)
  
  dat_reordered <- rbind(dat[101:150, ], dat[51:100, ], dat[1:50, ])
  nn <- neuralnet(Species ~ Petal.Length + Petal.Width, dat_reordered, linear.output = FALSE)
  expect_equal(unname(apply(nn$response, 1, which)), as.numeric(dat_reordered$Species))
  
  dat_char_reordered <- rbind(dat_char[101:150, ], dat_char[51:100, ], dat_char[1:50, ])
  nn <- neuralnet(Species ~ Petal.Length + Petal.Width, dat_char_reordered, linear.output = FALSE)
  expect_equal(colnames(nn$response)[apply(nn$response, 1, which)], dat_char_reordered$Species)
})

