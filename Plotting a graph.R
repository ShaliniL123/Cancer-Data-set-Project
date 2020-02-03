install.packages("tidyverse")
install.packages("corrplot")
install.packages("corrr")
install.packages("kernlab")
install.packages("DT")

library(tidyverse)
library(corrplot)
library(corrr)
library(kernlab)
library(DT)
library(ggplot2)
library(dplyr)
library(corrplot)


bc=read.csv("C:/Users/shalini lingampally/Desktop/R program/Material/All Practice xlx sheets/(10) Regularization/wisc_bc_data-KNN.csv")
dim(bc)
head(bc)
summary(bc)
str(bc)
bc1=bc[-1]

corrplot(cor(bc[-c(1,2)]),order='hclust')

library (e1071)
svmfit =svm(bc$diagnosis  ~., data=bc1)
summary(svmfit)

head(bc1)
#linear
cost_range = c(seq(1,3,by=0.01))
tune.out = tune(svm, diagnosis~., data = bc1 , kernel = "linear",
                 ranges = list(cost=cost_range))
summary(tune.out)

bestmod =tune.out$best.model
summary (bestmod)

#trainset
ypredd = predict(bestmod,trainset)
table(ypredd,trainset$diagnosis)
mean(ypredd==trainset$diagnosis)

#testset
ypred = predict(bestmod,testset)
table(ypred,testset$diagnosis)
mean(ypred==testset$diagnosis)

set.seed(1)
train=sample(2,nrow(bc1),replace = TRUE,prob = c(0.75,0.25))
trainset = bc[train==1,]
testset = bc[train==2,]

cost_range = c(seq(1,3,by=0.01))
tune.out = tune(svm, diagnosis~., data = trainset , kernel = "linear",
                ranges = list(cost=cost_range))
summary(tune.out)

bestmod =tune.out$best.model
summary (bestmod)

#trainset
ypredd = predict(bestmod,trainset)
table(ypredd,trainset$diagnosis)
mean(ypredd==trainset$diagnosis)

#testset
ypred = predict(bestmod,testset)
table(ypred,testset$diagnosis)
mean(ypred==testset$diagnosis)

bc2=bc[-c(1,2,3,4,5,6,7,9,10,11,12,14,15,16,17,18,19,20,21,22,25,26,29,32)]
head(bc2)

svmfit =svm(bc$diagnosis  ~., data=bc2)
summary(svmfit)

plot(svmfit , bc2 ,radius_worst~concave.points_worst,slice = list(texture_worst = 26.5 ,smoothness_worst = 0.13,radius_se = 0.44 ,
                                                                  compactness_mean =0.11 ,compactness_worst = 0.27 ,symmetry_worst = 0.29))


#plot(svmfit , bc2 ,radius_worst~concave.points_worst,slice = list(texture_worst = 30 ,smoothness_worst = 5,radius_se = 6,
#                                                                  compactness_mean =1 ,compactness_worst = 1 ,symmetry_worst = 1))


svmfit$index  # support vectors


#plot( bc$symmetry_worst,bc$radius_se)
#'concave points_mean', 'radius_se', 'texture_worst','smoothness_worst', 'compactness_worst','symmetry_worst'
# 8 variables


a=mean(bc$concave.points_worst[bc$diagnosis=='M'])
b=mean(bc$concave.points_worst[bc$diagnosis=='B'])
mean(c(a,b))

c=mean(bc$compactness_worst[bc$diagnosis=='M'])
d=mean(bc$compactness_worst[bc$diagnosis=='B'])
mean(c(c,d))

c=mean(bc$radius_se[bc$diagnosis=='M'])
d=mean(bc$radius_se[bc$diagnosis=='B'])
mean(c(c,d))

c=mean(bc$texture_worst[bc$diagnosis=='M'])
d=mean(bc$texture_worst[bc$diagnosis=='B'])
mean(c(c,d))

c=mean(bc$smoothness_worst[bc$diagnosis=='M'])
d=mean(bc$smoothness_worst[bc$diagnosis=='B'])
mean(c(c,d))

c=mean(bc$symmetry_worst[bc$diagnosis=='M'])
d=mean(bc$symmetry_worst[bc$diagnosis=='B'])
mean(c(c,d))

c=mean(bc$compactness_mean[bc$diagnosis=='M'])
d=mean(bc$compactness_mean[bc$diagnosis=='B'])
mean(c(c,d))

c=mean(bc$compactness_worst[bc$diagnosis=='M'])
d=mean(bc$compactness_worst[bc$diagnosis=='B'])
mean(c(c,d))
