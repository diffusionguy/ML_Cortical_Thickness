# Import all the libraries that might be used

library(caret)
library(pROC)

# Clear the variables

rm(list=ls())
rm(list=lsf.str())

# Set the working Directory
setwd("I:\\Boxers_Study\\Ctx_Analysis_04182020\\ML\\Radiology_CTX+Volume")

# Get the real data performance metrics
super_model_RF <- readRDS("./rf_model.rds")
super_model_linsvm <- readRDS("./linsvm_model.rds")
super_model_nlinsvm <- readRDS("./nlinsvm_model.rds")

rf_sensitivity<-0
rf_specificity<-0
rf_accuracy<-0
rf_roc<-0

linsvm_sensitivity<-0
linsvm_specificity<-0
linsvm_accuracy<-0
linsvm_roc<-0

nlinsvm_sensitivity<-0
nlinsvm_specificity<-0
nlinsvm_accuracy<-0
nlinsvm_roc<-0

for (i in 1:5000){
  training_name<-paste("Randomization\\training_",toString(i),".csv",sep="")
  testing_name<-paste("Randomization\\testing_",toString(i),".csv",sep="")
  x_test<-as.matrix(read.csv(training_name,na.strings=NaN,header=F))
  #y_test<-as.factor(unlist(read.csv(testing_name,na.strings=NaN,header=F)))
  y_test<-(unlist(read.csv(testing_name,na.strings=NaN,header=F)))
  x_test_std<-scale(x_test,center=T, scale =T)
  x_test_new<-x_test_std
  y_test[y_test==0] <- 2
  y_test<-as.factor(y_test)
  y_test_new<-y_test
  
  results_rf1<-data.frame(pred=predict(super_model_RF,x_test_new),obs=y_test_new)
  m<-confusionMatrix(table(results_rf1))
  rf_sensitivity[i]<-m$byClass[1]
  rf_specificity[i]<-m$byClass[2]
  rf_accuracy[i]<-m$overall[1]
  t<-plot.roc(as.numeric(y_test),as.numeric(results_rf1[,1]))
  rf_roc[i]<-t$auc[1]
  
  results_linsvm1<-data.frame(pred=predict(super_model_linsvm,x_test_new),obs=y_test_new)
  m<-confusionMatrix(table(results_linsvm1))
  linsvm_sensitivity[i]<-m$byClass[1]
  linsvm_specificity[i]<-m$byClass[2]
  linsvm_accuracy[i]<-m$overall[1]
  t<-plot.roc(as.numeric(y_test),as.numeric(results_linsvm1[,1]))
  linsvm_roc[i]<-t$auc[1]
  
  results_nlinsvm1<-data.frame(pred=predict(super_model_nlinsvm,x_test_new),obs=y_test_new)
  m<-confusionMatrix(table(results_nlinsvm1))
  nlinsvm_sensitivity[i]<-m$byClass[1]
  nlinsvm_specificity[i]<-m$byClass[2]
  nlinsvm_accuracy[i]<-m$overall[1]
  t<-plot.roc(as.numeric(y_test),as.numeric(results_nlinsvm1[,1]))
  nlinsvm_roc[i]<-t$auc[1]
  print(i)
}

rf_accuracy_95<-quantile(rf_accuracy,0.95)
rf_sensitivity_95<-quantile(rf_sensitivity,0.95)
rf_specificity_95<-quantile(rf_specificity,0.95)
rf_roc_95<-quantile(rf_roc,0.95)
print(rf_accuracy_95)
print(rf_sensitivity_95)
print(rf_specificity_95)
print(rf_roc_95)

linsvm_accuracy_95<-quantile(linsvm_accuracy,0.95)
linsvm_sensitivity_95<-quantile(linsvm_sensitivity,0.95)
linsvm_specificity_95<-quantile(linsvm_specificity,0.95)
linsvm_roc_95<-quantile(linsvm_roc,0.95)
print(linsvm_accuracy_95)
print(linsvm_sensitivity_95)
print(linsvm_specificity_95)
print(linsvm_roc_95)

nlinsvm_accuracy_95<-quantile(nlinsvm_accuracy,0.95)
nlinsvm_sensitivity_95<-quantile(nlinsvm_sensitivity,0.95)
nlinsvm_specificity_95<-quantile(nlinsvm_specificity,0.95)
nlinsvm_roc_95<-quantile(nlinsvm_roc,0.95)
print(nlinsvm_accuracy_95)
print(nlinsvm_sensitivity_95)
print(nlinsvm_specificity_95)
print(nlinsvm_roc_95)
