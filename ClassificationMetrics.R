library(glmnet)
accuracy_score <- function(y, yhat) {
  # Accuracy of yhat
  sum(y == yhat) / length(y)
}

zero_one_loss <- function(y, yhat) {
  # 1 - the accuracy score is called the zero one loss
  1 - accuracy_score(y, yhat)
}

# Area Under Curve Measure of Fit
roc_auc_score <- glmnet::auc

confusion_matrix <- function(y, yhat) {
  a <- matrix(0, nrow = 2, ncol = 2)
  a[1,1] <- sum((yhat == 1) & (y == 1))
  a[2,1] <- sum((yhat == 0) & (y == 1))
  a[1,2] <- sum((yhat == 1) & (y == 0))
  a[2,2] <- sum((yhat == 0) & (y == 0))
  return(a)
}

precision_score <- function(y, yhat) {
  a <- confusion_matrix(y, yhat)
  a[1,1] / (a[1,1] + a[1,2])
}

recall_score <- function(y, yhat) {
  a <- confusion_matrix(y, yhat)
  a[1,1] / (a[1,1] + a[2,1])
}

f1_score <- function(y, yhat) {
  precision <- precision_score(y, yhat)
  recall <- recall_score(y, yhat)
  2 * (precision * recall) / (precision + recall)
}

all_scores <- function(y, prob, yhat = NULL) {
  if(is.null(yhat)) {
    yhat <- as.numeric(prob > 0.5)
  }
  base_acc <- max(table(y)) / length(y)
  set.seed(1)
  base_roc <- roc_auc_score(y, rnorm(length(y)))
  base_f1  <- f1_score(y, rep(1, length(y)))
  results<-matrix(c(accuracy_score(y,yhat),
                    roc_auc_score(y,prob),
                    f1_score(y,yhat),
                    base_acc,
                    base_roc,
                    base_f1),ncol=2)
  colnames(results)<-c('Model','Baseline')
  rownames(results)<-c('Accuracy Score',
                       'ROC Score',
                       'F1 Score')
  return(results)
}