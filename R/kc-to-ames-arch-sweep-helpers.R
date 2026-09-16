# Shared by the sweep driver and every daemon -- see R/kc-to-ames-arch-sweep.R
# for why these variants are being tested.

variants <- list(
  node_norm      = list(norm_mode = "node",  concat = TRUE),
  no_concat      = list(norm_mode = "graph", concat = FALSE),
  node_no_concat = list(norm_mode = "node",  concat = FALSE)
)

fit_sage_variant <- function(train_id, val_id, norm_mode, concat) {
  ids <- c(train_id, val_id)
  scaler <- make_scaler(ids)
  g <- make_sub_graph(geom_of("kc")[ids])

  x_t <- torch_tensor(scaler(ids, "kc"), dtype = torch_float32())
  y_t <- torch_tensor(kc$y[ids], dtype = torch_float32())$view(c(-1, 1))

  train_idx <- seq_along(train_id)
  val_idx <- seq_along(val_id) + length(train_id)

  model <- model_sage(
    in_features = length(feature_cols),
    hidden_dims = sage_hidden,
    out_features = 1,
    norm = function(dim) layer_layer_norm(dim, mode = norm_mode),
    concat = concat
  )
  optimizer <- optim_adam(model$parameters, lr = lr)

  best_val <- Inf
  best_state <- NULL
  no_improve <- 0L

  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    out <- model(x_t, g$adj)
    loss <- nnf_l1_loss(out[train_idx, ], y_t[train_idx, ])
    loss$backward()
    optimizer$step()

    with_no_grad({
      model$eval()
      v <- nnf_l1_loss(model(x_t, g$adj)[val_idx, ], y_t[val_idx, ])$item()
    })

    if (v < best_val) {
      best_val <- v
      best_state <- lapply(model$state_dict(), \(t) t$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
    }
    if (no_improve >= patience) break
  }

  model$load_state_dict(best_state)
  model$eval()

  function(ids2, region) {
    g2 <- make_sub_graph(geom_of(region)[ids2])
    x2 <- torch_tensor(scaler(ids2, region), dtype = torch_float32())
    with_no_grad({
      as.numeric(model(x2, g2$adj)$squeeze())
    })
  }
}

sweep_task <- function(spec) {
  v <- variants[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant(final_train, final_val, v$norm_mode, v$concat)
  cbind(
    variant = spec$variant,
    seed = spec$seed,
    score(ames$y, predictor(seq_len(nrow(ames)), "ames"))
  )
}
