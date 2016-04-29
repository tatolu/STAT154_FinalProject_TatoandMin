set.seed(0)
setwd('~/Berkeley/senior/SPRING/STAT154_MachineLearning/FINAL')
source("./ClassificationMetrics.R")

#####################
### DATA
#####################
load(file='TrainTest.Rdata')
train.X<-X
train.Y<-y
test.X<-Xtest
test.X.numeric<-apply(test.X,2,as.numeric)
n<-nrow(train.X)
train2<-sample(n,floor(0.7*n))
#randomly select a portion of the training data to be yet another training set
dim(train.X)


# train.pc<-prcomp(train.X,center=T,scale.=T,retx=T)
# train.pc.x<-train.pc$x[,1:651] 
# I used PCA for dimension reduction to quicken computing speed
# However, PCA fails to reduce the dimension drastically

#####################
### RANDOM FOREST
#####################
library(xgboost)
rf.model<-xgboost(data = train.X[train2,], label = train.Y[train2], 
                   nrounds = 100, verbose = 2, 
                   max.depth = 50, num_parallel_tree = 20, 
                   subsample = 1., colsample_bytree = .04, 
                   objective = "multi:softmax", num_class = 10)
rf.yhat<-predict(rf.model, train.X[-train2,])
rf.scores<-all_scores(train.Y[-train2],rf.yhat)

#####################
### LDA 
#####################
library(MASS)
lda.model<-lda(x=train.X[train2,],grouping=train.Y[train2])
lda.yhat<-predict(lda.model,newdata=train.X[-train2,])$class
lda.yhat<-as.numeric(lda.yhat)
lda.scores<-all_scores(train.Y[-train2],lda.yhat)


#################
## Logistic regression
#################
logit.data<-cbind(train.Y,train.X)
logit.data<-data.frame(as.matrix(logit.data))
logit.model<-glm(train.Y~.,data=logit.data[train2,],
                 family='binomial')
logit.yhat<-predict(logit.model,newdata=logit.data[-train2,-1])
logit.scores<-all_scores(train.Y[-train2],logit.yhat)

#################
## Ridge logistic regression
#################
library(glmnet)
ridge.logit.model<-cv.glmnet(train.X[train2,],y=train.Y[train2],
                           family='binomial',alpha=0)
ridge.logit.yhat<-predict(ridge.logit.model,newx=train.X[-train2,],
                        s='lambda.1se')
ridge.logit.scores<-all_scores(train.Y[-train2],ridge.logit.yhat)


# Out of the tested methods, logistic regression and random forest perform the best.

scores.summary<-cbind(lda.scores,logit.scores,ridge.logit.scores,rf.scores)
scores.summary<-scores.summary[,c(1,3,5,7,8)]
colnames(scores.summary)<-c('LDA','Logistic','L2 Logistic','Random Forest','Baseline')
write.csv(scores.summary,file='ModelPerformanceTable.csv',row.names=TRUE)

#################
## Output
#################
id<-1:nrow(test.X)
#Baseline
baseline.output<-rep(1,nrow(test.X))
baseline.output<-cbind(id,baseline.output)
colnames(baseline.output)[2]<-'y'
#Logistic Regression
logit.test<-data.frame(as.matrix(cbind(train.Y,test.X)))
logit.output<-predict(logit.model,newdata=logit.test[,-1])
logit.output<-logit.output>0.5
logit.output<-cbind(id,logit.output)
colnames(logit.output)[2]<-'y'
#Random Forest
rf.output<-predict(rf.model,test.X.numeric)
rf.output<-cbind(id,rf.output)
colnames(rf.output)[2]<-'y'

#Ridge Logistic Regression
ridge.logit.output<-predict(ridge.logit.model,newx=test.X,
                            s='lambda.1se')
ridge.logit.output<-ridge.logit.output>0.5
ridge.logit.output<-cbind(id,ridge.logit.output)
colnames(ridge.logit.output)[2]<-'y'
#LDA
lda.output<-predict(lda.model,newdata=test.X)$class
lda.output<-cbind(id,lda.output)
colnames(lda.output)[2]<-'y'

write.csv(baseline.output,file='BaselineOutput.csv',row.names=FALSE)
write.csv(logit.output,file='LogitOutput.csv',row.names=FALSE)
write.csv(rf.output,file='RFOutput.csv',row.names=FALSE)
write.csv(ridge.logit.output,file='RidgeLogitOutput.csv',row.names=FALSE)
write.csv(lda.output,file='LDAOutput.csv',row.names=FALSE)


#################
## ROC Curves
#################
# install.packages('ROCR')
library(ROCR)

#Logistic Regression
p <- predict(logit.model, newdata = logit.data[-train2,], type = 'response')
pr <- prediction(p, train.Y[-train2])
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
title(main = "Logistic Regression ROC")

#Random Forest
p_rf <- predict(rf.model, newdata = train.X[-train2,])
pr_rf <- prediction(p_rf, train.Y[-train2])
prf_rf <- performance(pr_rf, measure = "tpr", x.measure = "fpr")
plot(prf_rf)
title(main = "Random Forest ROC")