# Developer: Virendra R Mishra, Imaging Research, Cleveland Clinic, Las Vegas
# Import all the libraries that might be used
library(glmnet)
library(caret)
library(pROC)

# Clear the variables

rm(list=ls())
rm(list=lsf.str())

# Set path
setwd("I:\\Boxers_Study\\Ctx_Analysis_04182020\\ML\\CTX+Volume")
x_train<-as.matrix(read.csv("training.csv",na.strings=NaN,header=F))
x_test<-as.matrix(read.csv("testing.csv",na.strings=NaN,header=F))
y_train<-as.factor(unlist(read.csv("Y_training.csv",na.strings=NaN,header=F)))
y_test<-as.factor(unlist(read.csv("Y_testing.csv",na.strings=NaN,header=F)))

x_train_std<-scale(x_train,center=T, scale =T)
x_test_std<-scale(x_test,center=T, scale =T)

#*******************************************************************************#
#******************* Train lasso glment with all features **********************#
#*******************************************************************************#
#*******************************************************************************# 
set.seed(123)
glmmod<-glmnet(x_train_std,as.factor(y_train),family="binomial",
               alpha=1,type.logistic="modified.Newton",standardize=F)
cvfit<-cv.glmnet(x_train_std,as.factor(y_train),family="binomial",
                 nfolds=10,alpha=1,type.logistic="modified.Newton",
                 parallel=F,type.measure = "class",standardize=F)

idx_coef<-which(glmmod$lambda==cvfit$lambda.1se)
coeff<-coef(glmmod)[,idx_coef]

yhat<-predict(cvfit, x_test_std, s = "lambda.1se", type = "class")

confusionMatrix(yhat,y_test)
yhat_lasso<-yhat

# Get the indices of the selected features

temp_x<-(which(coeff!=0)) # This has the indices of the top N non-zero features with the intercept
temp_idx<-temp_x[-1]-1 # This has the indices of the top N non-zero features with the intercept
lasso_idx<-temp_idx

# Get the indices of the selected features

temp_x<-(which(coeff!=0)) # This has the indices of the top N non-zero features with the intercept
temp_idx<-temp_x[-1]-1 # This has the indices of the top N non-zero features with the intercept
lasso_idx<-temp_idx

