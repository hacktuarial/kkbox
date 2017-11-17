import("stringr")
import("stats")
import("mbest")
fit <- function(ff, df_train) {
  if (str_detect(ff, fixed("|"))) {
    m <- mhglm(ff, family=binomial, data=df_train,
               model=FALSE, y=FALSE, group=FALSE,
               control=list(parallel=TRUE))
  } else {
    # see also MatrixModels::glm4
    m <- stats::glm(as.formula(ff),
            family=binomial, data=df_train)
  }
  return(m)
}

eval <- function(model, df_test) {
  preds <- predict(model, df_test)
  p <- ROCR::prediction(preds, df_test[["target"]])
  return(ROCR::performance(p, "auc")@y.values[[1]])
}