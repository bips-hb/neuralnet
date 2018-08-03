calculate.neuralnet <-
  function (data, model.list, hidden, stepmax, rep, threshold, 
            learningrate.limit, learningrate.factor, lifesign, covariate, 
            response, lifesign.step, startweights, algorithm, act.fct, 
            act.deriv.fct, err.fct, err.deriv.fct, linear.output, likelihood, 
            exclude, constant.weights, learningrate.bp) 
  {
    time.start.local <- Sys.time()
    result <- generate.startweights(model.list, hidden, startweights, 
                                    rep, exclude, constant.weights)
    weights <- result$weights
    exclude <- result$exclude
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    result <- rprop(weights = weights, threshold = threshold, 
                    response = response, covariate = covariate, learningrate.limit = learningrate.limit, 
                    learningrate.factor = learningrate.factor, stepmax = stepmax, 
                    lifesign = lifesign, lifesign.step = lifesign.step, act.fct = act.fct, 
                    act.deriv.fct = act.deriv.fct, err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                    algorithm = algorithm, linear.output = linear.output, 
                    exclude = exclude, learningrate.bp = learningrate.bp)
    startweights <- weights
    weights <- result$weights
    step <- result$step
    reached.threshold <- result$reached.threshold
    net.result <- result$net.result
    error <- sum(err.fct(net.result, response))
    if (is.na(error) & attr(err.fct, "type") == "ce") 
      if (all(net.result <= 1, net.result >= 0)) 
        error <- sum(err.fct(net.result, response), na.rm = T)
    if (!is.null(constant.weights) && any(constant.weights != 
                                          0)) 
      exclude <- exclude[-which(constant.weights != 0)]
    if (length(exclude) == 0) 
      exclude <- NULL
    aic <- NULL
    bic <- NULL
    if (likelihood) {
      synapse.count <- length(unlist(weights)) - length(exclude)
      aic <- 2 * error + (2 * synapse.count)
      bic <- 2 * error + log(nrow(response)) * synapse.count
    }
    if (is.na(error)) 
      warning("'err.fct' does not fit 'data' or 'act.fct'", 
              call. = F)
    if (lifesign != "none") {
      if (reached.threshold <= threshold) {
        message(rep(" ", (max(nchar(stepmax), nchar("stepmax")) - 
                            nchar(step))), step, appendLF = FALSE)
        message("\terror: ", round(error, 5), rep(" ", 6 - (nchar(round(error, 
                                                                        5)) - nchar(round(error, 0)))), appendLF = FALSE)
        if (!is.null(aic)) {
          message("\taic: ", round(aic, 5), rep(" ", 6 - (nchar(round(aic, 
                                                                      5)) - nchar(round(aic, 0)))), appendLF = FALSE)
        }
        if (!is.null(bic)) {
          message("\tbic: ", round(bic, 5), rep(" ", 6 - (nchar(round(bic, 
                                                                      5)) - nchar(round(bic, 0)))), appendLF = FALSE)
        }
        time <- difftime(Sys.time(), time.start.local)
        message("\ttime: ", round(time, 2), " ", attr(time, "units"))
      }
    }
    if (reached.threshold > threshold) 
      return(result = list(output.vector = NULL, weights = NULL))
    output.vector <- c(error = error, reached.threshold = reached.threshold, 
                       steps = step)
    if (!is.null(aic)) {
      output.vector <- c(output.vector, aic = aic)
    }
    if (!is.null(bic)) {
      output.vector <- c(output.vector, bic = bic)
    }
    for (w in 1:length(weights)) output.vector <- c(output.vector, 
                                                    as.vector(weights[[w]]))
    generalized.weights <- calculate.generalized.weights(weights, 
                                                         neuron.deriv = result$neuron.deriv, net.result = net.result)
    startweights <- unlist(startweights)
    weights <- unlist(weights)
    if (!is.null(exclude)) {
      startweights[exclude] <- NA
      weights[exclude] <- NA
    }
    startweights <- relist(startweights, nrow.weights, ncol.weights)
    weights <- relist(weights, nrow.weights, ncol.weights)
    return(list(generalized.weights = generalized.weights, weights = weights, 
                startweights = startweights, net.result = result$net.result, 
                output.vector = output.vector))
  }
generate.startweights <-
  function (model.list, hidden, startweights, rep, exclude, constant.weights) 
  {
    input.count <- length(model.list$variables)
    output.count <- length(model.list$response)
    if (!(length(hidden) == 1 && hidden == 0)) {
      length.weights <- length(hidden) + 1
      nrow.weights <- array(0, dim = c(length.weights))
      ncol.weights <- array(0, dim = c(length.weights))
      nrow.weights[1] <- (input.count + 1)
      ncol.weights[1] <- hidden[1]
      if (length(hidden) > 1) 
        for (i in 2:length(hidden)) {
          nrow.weights[i] <- hidden[i - 1] + 1
          ncol.weights[i] <- hidden[i]
        }
      nrow.weights[length.weights] <- hidden[length.weights - 
                                               1] + 1
      ncol.weights[length.weights] <- output.count
    }
    else {
      length.weights <- 1
      nrow.weights <- array((input.count + 1), dim = c(1))
      ncol.weights <- array(output.count, dim = c(1))
    }
    length <- sum(ncol.weights * nrow.weights)
    vector <- rep(0, length)
    if (!is.null(exclude)) {
      if (is.matrix(exclude)) {
        exclude <- matrix(as.integer(exclude), ncol = ncol(exclude), 
                          nrow = nrow(exclude))
        if (nrow(exclude) >= length || ncol(exclude) != 3) 
          stop("'exclude' has wrong dimensions", call. = FALSE)
        if (any(exclude < 1)) 
          stop("'exclude' contains at least one invalid weight", 
               call. = FALSE)
        temp <- relist(vector, nrow.weights, ncol.weights)
        for (i in 1:nrow(exclude)) {
          if (exclude[i, 1] > length.weights || exclude[i, 
                                                        2] > nrow.weights[exclude[i, 1]] || exclude[i, 
                                                                                                    3] > ncol.weights[exclude[i, 1]]) 
            stop("'exclude' contains at least one invalid weight", 
                 call. = FALSE)
          temp[[exclude[i, 1]]][exclude[i, 2], exclude[i, 
                                                       3]] <- 1
        }
        exclude <- which(unlist(temp) == 1)
      }
      else if (is.vector(exclude)) {
        exclude <- as.integer(exclude)
        if (max(exclude) > length || min(exclude) < 1) {
          stop("'exclude' contains at least one invalid weight", 
               call. = FALSE)
        }
      }
      else {
        stop("'exclude' must be a vector or matrix", call. = FALSE)
      }
      if (length(exclude) >= length) 
        stop("all weights are exluded", call. = FALSE)
    }
    length <- length - length(exclude)
    if (!is.null(exclude)) {
      if (is.null(startweights) || length(startweights) < (length * 
                                                           rep)) 
        vector[-exclude] <- stats::rnorm(length)
      else vector[-exclude] <- startweights[((rep - 1) * length + 
                                               1):(length * rep)]
    }
    else {
      if (is.null(startweights) || length(startweights) < (length * 
                                                           rep)) 
        vector <- stats::rnorm(length)
      else vector <- startweights[((rep - 1) * length + 1):(length * 
                                                              rep)]
    }
    if (!is.null(exclude) && !is.null(constant.weights)) {
      if (length(exclude) < length(constant.weights)) 
        stop("constant.weights contains more weights than exclude", 
             call. = FALSE)
      else vector[exclude[1:length(constant.weights)]] <- constant.weights
    }
    weights <- relist(vector, nrow.weights, ncol.weights)
    return(list(weights = weights, exclude = exclude))
  }
rprop <-
  function (weights, response, covariate, threshold, learningrate.limit, 
            learningrate.factor, stepmax, lifesign, lifesign.step, act.fct, 
            act.deriv.fct, err.fct, err.deriv.fct, algorithm, linear.output, 
            exclude, learningrate.bp) 
  {
    step <- 1
    nchar.stepmax <- max(nchar(stepmax), 7)
    length.weights <- length(weights)
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    length.unlist <- length(unlist(weights)) - length(exclude)
    learningrate <- as.vector(matrix(0.1, nrow = 1, ncol = length.unlist))
    gradients.old <- as.vector(matrix(0, nrow = 1, ncol = length.unlist))
    if (is.null(exclude)) 
      exclude <- length(unlist(weights)) + 1
    if (attr(act.fct, "type") == "tanh" || attr(act.fct, "type") == "logistic") 
      special <- TRUE
    else special <- FALSE
    if (linear.output) {
      output.act.fct <- function(x) {
        x
      }
      output.act.deriv.fct <- function(x) {
        matrix(1, nrow(x), ncol(x))
      }
    }
    else {
      if (attr(err.fct, "type") == "ce" && attr(act.fct, "type") == "logistic") {
        err.deriv.fct <- function(x, y) {
          x * (1 - y) - y * (1 - x)
        }
        linear.output <- TRUE
      }
      output.act.fct <- act.fct
      output.act.deriv.fct <- act.deriv.fct
    }
    result <- compute.net(weights, length.weights, covariate = covariate, 
                          act.fct = act.fct, act.deriv.fct = act.deriv.fct, output.act.fct = output.act.fct, 
                          output.act.deriv.fct = output.act.deriv.fct, special)
    err.deriv <- err.deriv.fct(result$net.result, response)
    gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
                                     neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
                                     err.deriv = err.deriv, exclude = exclude, linear.output = linear.output)
    reached.threshold <- max(abs(gradients))
    min.reached.threshold <- reached.threshold
    while (step < stepmax && reached.threshold > threshold) {
      if (!is.character(lifesign) && step%%lifesign.step == 
          0) {
        text <- paste("%", nchar.stepmax, "s", sep = "")
        message(sprintf(eval(expression(text)), step), "\tmin thresh: ", 
                min.reached.threshold, "\n", rep(" ", lifesign), appendLF = FALSE)
        utils::flush.console()
      }
      if (algorithm == "rprop+") 
        result <- plus(gradients, gradients.old, weights, 
                       nrow.weights, ncol.weights, learningrate, learningrate.factor, 
                       learningrate.limit, exclude)
      else if (algorithm == "backprop") 
        result <- backprop(gradients, weights, length.weights, 
                           nrow.weights, ncol.weights, learningrate.bp, 
                           exclude)
      else result <- minus(gradients, gradients.old, weights, 
                           length.weights, nrow.weights, ncol.weights, learningrate, 
                           learningrate.factor, learningrate.limit, algorithm, 
                           exclude)
      gradients.old <- result$gradients.old
      weights <- result$weights
      learningrate <- result$learningrate
      result <- compute.net(weights, length.weights, covariate = covariate, 
                            act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                            output.act.fct = output.act.fct, output.act.deriv.fct = output.act.deriv.fct, 
                            special)
      err.deriv <- err.deriv.fct(result$net.result, response)
      gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
                                       neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
                                       err.deriv = err.deriv, exclude = exclude, linear.output = linear.output)
      reached.threshold <- max(abs(gradients))
      if (reached.threshold < min.reached.threshold) {
        min.reached.threshold <- reached.threshold
      }
      step <- step + 1
    }
    if (lifesign != "none" && reached.threshold > threshold) {
      message("stepmax\tmin thresh: ", min.reached.threshold)
    }
    return(list(weights = weights, step = as.integer(step), reached.threshold = reached.threshold, 
                net.result = result$net.result, neuron.deriv = result$neuron.deriv))
  }
compute.net <-
  function (weights, length.weights, covariate, act.fct, act.deriv.fct, 
            output.act.fct, output.act.deriv.fct, special) 
  {
    neuron.deriv <- NULL
    neurons <- list(covariate)
    if (length.weights > 1) 
      for (i in 1:(length.weights - 1)) {
        temp <- neurons[[i]] %*% weights[[i]]
        act.temp <- act.fct(temp)
        if (special) 
          neuron.deriv[[i]] <- act.deriv.fct(act.temp)
        else neuron.deriv[[i]] <- act.deriv.fct(temp)
        neurons[[i + 1]] <- cbind(1, act.temp)
      }
    if (!is.list(neuron.deriv)) 
      neuron.deriv <- list(neuron.deriv)
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    net.result <- output.act.fct(temp)
    if (special) 
      neuron.deriv[[length.weights]] <- output.act.deriv.fct(net.result)
    else neuron.deriv[[length.weights]] <- output.act.deriv.fct(temp)
    if (any(is.na(neuron.deriv))) 
      stop("neuron derivatives contain a NA; varify that the derivative function does not divide by 0", 
           call. = FALSE)
    list(neurons = neurons, neuron.deriv = neuron.deriv, net.result = net.result)
  }
calculate.gradients <-
  function (weights, length.weights, neurons, neuron.deriv, err.deriv, 
            exclude, linear.output) 
  {
    if (any(is.na(err.deriv))) 
      stop("the error derivative contains a NA; varify that the derivative function does not divide by 0 (e.g. cross entropy)", 
           call. = FALSE)
    if (!linear.output) 
      delta <- neuron.deriv[[length.weights]] * err.deriv
    else delta <- err.deriv
    gradients <- crossprod(neurons[[length.weights]], delta)
    if (length.weights > 1) 
      for (w in (length.weights - 1):1) {
        delta <- neuron.deriv[[w]] * tcrossprod(delta, weights[[w + 1]][-1,, drop = FALSE])
        gradients <- c(crossprod(neurons[[w]], delta), gradients)
      }
    gradients[-exclude]
  }
plus <-
  function (gradients, gradients.old, weights, nrow.weights, ncol.weights, 
            learningrate, learningrate.factor, learningrate.limit, exclude) 
  {
    weights <- unlist(weights)
    sign.gradient <- sign(gradients)
    temp <- gradients.old * sign.gradient
    positive <- temp > 0
    negative <- temp < 0
    not.negative <- !negative
    if (any(positive)) {
      learningrate[positive] <- pmin.int(learningrate[positive] * 
                                           learningrate.factor$plus, learningrate.limit$max)
    }
    if (any(negative)) {
      weights[-exclude][negative] <- weights[-exclude][negative] + 
        gradients.old[negative] * learningrate[negative]
      learningrate[negative] <- pmax.int(learningrate[negative] * 
                                           learningrate.factor$minus, learningrate.limit$min)
      gradients.old[negative] <- 0
      if (any(not.negative)) {
        weights[-exclude][not.negative] <- weights[-exclude][not.negative] - 
          sign.gradient[not.negative] * learningrate[not.negative]
        gradients.old[not.negative] <- sign.gradient[not.negative]
      }
    }
    else {
      weights[-exclude] <- weights[-exclude] - sign.gradient * 
        learningrate
      gradients.old <- sign.gradient
    }
    list(gradients.old = gradients.old, weights = relist(weights, 
                                                         nrow.weights, ncol.weights), learningrate = learningrate)
  }
backprop <-
  function (gradients, weights, length.weights, nrow.weights, ncol.weights, 
            learningrate.bp, exclude) 
  {
    weights <- unlist(weights)
    if (!is.null(exclude)) 
      weights[-exclude] <- weights[-exclude] - gradients * 
        learningrate.bp
    else weights <- weights - gradients * learningrate.bp
    list(gradients.old = gradients, weights = relist(weights, 
                                                     nrow.weights, ncol.weights), learningrate = learningrate.bp)
  }
minus <-
  function (gradients, gradients.old, weights, length.weights, 
            nrow.weights, ncol.weights, learningrate, learningrate.factor, 
            learningrate.limit, algorithm, exclude) 
  {
    weights <- unlist(weights)
    temp <- gradients.old * gradients
    positive <- temp > 0
    negative <- temp < 0
    if (any(positive)) 
      learningrate[positive] <- pmin.int(learningrate[positive] * 
                                           learningrate.factor$plus, learningrate.limit$max)
    if (any(negative)) 
      learningrate[negative] <- pmax.int(learningrate[negative] * 
                                           learningrate.factor$minus, learningrate.limit$min)
    if (algorithm != "rprop-") {
      delta <- 10^-6
      notzero <- gradients != 0
      gradients.notzero <- gradients[notzero]
      if (algorithm == "slr") {
        min <- which.min(learningrate[notzero])
      }
      else if (algorithm == "sag") {
        min <- which.min(abs(gradients.notzero))
      }
      if (length(min) != 0) {
        temp <- learningrate[notzero] * gradients.notzero
        sum <- sum(temp[-min]) + delta
        learningrate[notzero][min] <- min(max(-sum/gradients.notzero[min], 
                                              learningrate.limit$min), learningrate.limit$max)
      }
    }
    weights[-exclude] <- weights[-exclude] - sign(gradients) * 
      learningrate
    list(gradients.old = gradients, weights = relist(weights, 
                                                     nrow.weights, ncol.weights), learningrate = learningrate)
  }
calculate.generalized.weights <-
  function (weights, neuron.deriv, net.result) 
  {
    for (w in 1:length(weights)) {
      weights[[w]] <- weights[[w]][-1,, drop = FALSE]
    }
    generalized.weights <- NULL
    for (k in 1:ncol(net.result)) {
      for (w in length(weights):1) {
        if (w == length(weights)) {
          temp <- neuron.deriv[[length(weights)]][, k] * 
            1/(net.result[, k] * (1 - (net.result[, k])))
          delta <- tcrossprod(temp, weights[[w]][, k])
        }
        else {
          delta <- tcrossprod(delta * neuron.deriv[[w]], 
                              weights[[w]])
        }
      }
      generalized.weights <- cbind(generalized.weights, delta)
    }
    return(generalized.weights)
  }

relist <-
  function (x, nrow, ncol) 
  {
    list.x <- NULL
    for (w in 1:length(nrow)) {
      length <- nrow[w] * ncol[w]
      list.x[[w]] <- matrix(x[1:length], nrow = nrow[w], ncol = ncol[w])
      x <- x[-(1:length)]
    }
    list.x
  }