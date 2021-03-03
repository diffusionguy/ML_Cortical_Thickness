# Import all the libraries that might be used
 
 library(caret)
 library(pROC)
 
# Clear the variables

rm(list=ls())
rm(list=lsf.str())

# Set the path
setwd("I:\\Boxers_Study\\Ctx_Analysis_04182020\\ML\\Radiology_CTX+Volume")
x_train<-as.matrix(read.csv("training.csv",na.strings=NaN,header=F))
x_test<-as.matrix(read.csv("testing.csv",na.strings=NaN,header=F))
y_train<-as.factor(unlist(read.csv("Y_training.csv",na.strings=NaN,header=F)))
y_test<-as.factor(unlist(read.csv("Y_testing.csv",na.strings=NaN,header=F)))

x_train_std<-scale(x_train,center=T, scale =T)
x_test_std<-scale(x_test,center=T, scale =T)

#*******************************************************************************#
#*******************************************************************************#
#******************* Train random forest model with all features ***************#
#*******************************************************************************#
#*******************************************************************************# 

x_train_new<-x_train_std
x_test_new<-x_test_std

y_train_new<-y_train
y_test_new<-y_test

# Using Caret -- Random Forest
set.seed(123)
rf_model<-train(as.data.frame(x_train_new),as.factor(y_train_new),method="RRF",
                trControl=trainControl(method="cv",number=10,seeds=NULL,allowParallel = F),
                prox=TRUE)
#preProc = c("center", "scale"),
results_rf<-data.frame(pred=predict(rf_model,x_test_new),obs=y_test_new)
confusionMatrix(table(results_rf))

plot.roc(as.numeric(y_test),as.numeric(results_rf[,1]),ci=T,main="ROC for Random Forest model")

saveRDS(rf_model, "./rf_model.rds")

super_model <- readRDS("./rf_model.rds")

results_rf1<-data.frame(pred=predict(super_model,x_test_new),obs=y_test_new)
confusionMatrix(table(results_rf1))
#*******************************************************************************#
#*******************************************************************************#
#******************* Train linear svm with all features ***************#
#*******************************************************************************#
#*******************************************************************************# 

x_train_new<-x_train_std
x_test_new<-x_test_std

y_train_new<-y_train
y_test_new<-y_test

# Using Caret -- Linear SVM
set.seed(123)
linearsvm_model<-train(as.data.frame(x_train_new),as.factor(y_train_new),method="svmLinear",
                trControl=trainControl(method="cv",number=10,seeds=NULL,allowParallel = F),
                prox=TRUE)
#preProc = c("center", "scale"),
results_linsvm<-data.frame(pred=predict(linearsvm_model,x_test_new),obs=y_test_new)
confusionMatrix(table(results_linsvm))

plot.roc(as.numeric(y_test),as.numeric(results_linsvm[,1]),ci=T,main="ROC for linear SVM model")

saveRDS(linearsvm_model, "./linsvm_model.rds")

super_model <- readRDS("./linsvm_model.rds")

results_linsvm1<-data.frame(pred=predict(super_model,x_test_new),obs=y_test_new)
confusionMatrix(table(results_linsvm1))


#*******************************************************************************#
#*******************************************************************************#
#******************* Train Nonlinear svm + RBFN with all features ***************#
#*******************************************************************************#
#*******************************************************************************# 

x_train_new<-x_train_std
x_test_new<-x_test_std

y_train_new<-y_train
y_test_new<-y_test

# Using Caret -- Nonlinear SVM + RBFN 
set.seed(123)
nlinearsvm_model<-train(as.data.frame(x_train_new),as.factor(y_train_new),method="svmRadial",
                       trControl=trainControl(method="cv",number=10,seeds=NULL,allowParallel = F),
                       prox=TRUE)
#preProc = c("center", "scale"),
results_nlinsvm<-data.frame(pred=predict(nlinearsvm_model,x_test_new),obs=y_test_new)
confusionMatrix(table(results_nlinsvm))

plot.roc(as.numeric(y_test),as.numeric(results_nlinsvm[,1]),ci=T,main="ROC for nonlinear SVM model")

saveRDS(nlinearsvm_model, "./nlinsvm_model.rds")

super_model <- readRDS("./nlinsvm_model.rds")

results_nlinsvm1<-data.frame(pred=predict(super_model,x_test_new),obs=y_test_new)
confusionMatrix(table(results_nlinsvm1))



