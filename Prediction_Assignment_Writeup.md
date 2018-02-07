# Prediction Assignment Writeup
Goutham Manghnani
Feb 02, 2018  

## 1. Environment Preparation
R Libraries are uploaded to complete the analysis

```r

install.packages("https://cran.r-project.org/bin/windows/contrib/3.3/RGtk2_2.20.31.zip", repos=NULL)
install.packages("e1071")

rm(list=ls())                # To ree up memory for the
library(knitr)
library(caret)
library(rpart)
library(e1071)
library(rattle)
library(randomForest)
library(corrplot)
library(rpart.plot)
```
## 2. Loading of Data

```r
set.seed(12463)
# set the URL for the download
TrainingURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestingURL  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
NewTrain <- read.csv(url(TrainingURL))
NewTest  <- read.csv(url(TestingURL))

# create a partition with the training dataset 
inTrain  <- createDataPartition(NewTrain$classe, p=0.7, list=FALSE)
Training <- NewTrain[inTrain, ]
Testing  <- NewTrain[-inTrain, ]
dim(TrainSet)
```
```r
13737   160
```
```r
dim(TestSet)
```
```r
5885  160
```

## 3. Data Cleaning

## Remove variables with Nearly Zero Variance


```r
NZV <- nearZeroVar(Training)
Training <- Training[, -NZV]
Testing  <- Testing[, -NZV]
dim(TrainSet)
```
```r
13737   104
```


## Remove variables that are mostly NA

```r
AllNA    <- sapply(Training, function(x) mean(is.na(x))) > 0.95
Training <- Training[, AllNA==FALSE]
Testing  <- Testing[, AllNA==FALSE]
dim(TrainSet)
```
```r
13737    59
```
```r
dim(TestSet)
```
```r
5885   54
```
## remove identification only variables (columns 1 to 5)

```r
Training <- Training[, -(1:5)]
Testing  <- Testing[, -(1:5)]

correlationMatrix <- cor(Training[, -54])
corrplot(correlationMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

The variable that are highly correlated are displayed in dark colors in the graph above. To make an evem more better analysis, a PCA (Principal Components Analysis) is performed as pre-processing step to the datasets. As the correlations are quite few, this step will not be applied for this assignment.

## Prediction Model Building

# Decision Tree - Model fit

```r

set.seed(12345)
modFitDecisionTree <- rpart(classe ~ ., data=Training, method="class")
fancyRpartPlot(modFitDecisionTree)
```


# prediction on Test dataset

```r
predictDecisionTree <- predict(modFitDecisionTree, newdata=Testing, type="class")
confMatDecisionTree <- confusionMatrix(predictDecisionTree, Testing$classe)
confMatDecisionTree
```

```r
##Confusion Matrix and Statistics
##
##          Reference
##Prediction    A    B    C    D    E
##         A 1672    5    0    0    0
##         B    1 1131    3    0    0
##         C    0    3 1022    2    0
##         D    0    0    1  961    0
##         E    1    0    0    1 1082
##
##Overall Statistics
##                                          
##               Accuracy : 0.9971          
##                 95% CI : (0.9954, 0.9983)
##    No Information Rate : 0.2845          
##    P-Value [Acc > NIR] : < 2.2e-16       
##                                          
##                  Kappa : 0.9963          
## Mcnemar's Test P-Value : NA              
##
##Statistics by Class:
##
##                     Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9988   0.9930   0.9961   0.9969   1.0000
##Specificity            0.9988   0.9992   0.9990   0.9998   0.9996
##Pos Pred Value         0.9970   0.9965   0.9951   0.9990   0.9982
##Neg Pred Value         0.9995   0.9983   0.9992   0.9994   1.0000
##Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
##Detection Rate         0.2841   0.1922   0.1737   0.1633   0.1839
##Detection Prevalence   0.2850   0.1929   0.1745   0.1635   0.1842
##Balanced Accuracy      0.9988   0.9961   0.9975   0.9983   0.9998
```
# plot matrix results

```r
plot(confMatDecisionTree$table, col = confMatDecisionTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecisionTree$overall['Accuracy'], 4)))
```

# model fit

```r
set.seed(12345)
controlRandomForest <- trainControl(method="cv", number=3, verboseIter=FALSE)
modleFitRandForest <- train(classe ~ ., data=Training, method="rf",
                          trControl=controlRandomForest)
modleFitRandForest$finalModel
```

```R
##randomForest(x = x, y = y, mtry = param$mtry) 
##               Type of random forest: classification
##                     Number of trees: 500
##No. of variables tried at each split: 27
##
##        OOB estimate of  error rate: 0.24%
##Confusion matrix:
##     A    B    C    D    E class.error
##A 3906    0    0    0    0 0.000000000
##B    5 2650    2    1    0 0.003009782
##C    0    6 2389    1    0 0.002921536
##D    0    0   11 2240    1 0.005328597
##E    0    0    0    6 2519 0.002376238
```

# prediction on Test dataset

```r
predictRandomForest <- predict(modleFitRandForest, newdata=Testing)
confMatRandomForest <- confusionMatrix(predictRandomForest, Testing$classe)
confMatRandomForest
```
```r
##Confusion Matrix and Statistics
##
##          Reference
##Prediction    A    B    C    D    E
##         A 1452  226   23  106   99
##         B   33  641   71   33   89
##         C   10   56  809  150   64
##         D  137  184   92  643  128
##         E   42   32   31   32  702
##
##Overall Statistics
##                                        
##               Accuracy : 0.7217        
##                 95% CI : (0.71, 0.7331)
##    No Information Rate : 0.2845        
##    P-Value [Acc > NIR] : < 2.2e-16     
##                                        
##                  Kappa : 0.6468        
## Mcnemar's Test P-Value : < 2.2e-16     
##
##Statistics by Class:
##
##                     Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.8674   0.5628   0.7885   0.6670   0.6488
##Specificity            0.8922   0.9524   0.9424   0.8901   0.9715
##Pos Pred Value         0.7618   0.7393   0.7429   0.5431   0.8367
##Neg Pred Value         0.9442   0.9008   0.9548   0.9317   0.9247
##Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
##Detection Rate         0.2467   0.1089   0.1375   0.1093   0.1193
##Detection Prevalence   0.3239   0.1473   0.1850   0.2012   0.1426
##Balanced Accuracy      0.8798   0.7576   0.8654   0.7785   0.8101
```

# plot matrix results

```r
plot(confMatRandomForest$table, col = confMatRandomForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandomForest$overall['Accuracy'], 4)))

```

# model fit

```r
set.seed(12345)
controlGBoostingM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBoostingM  <- train(classe ~ ., data=Training, method = "gbm",
                    trControl = controlGBoostingM, verbose = FALSE)
modFitGBoostingM$finalModel

```
```r
A gradient boosted model with multinomial loss function.
150 iterations were performed.
```


# prediction on Test dataset

```r
predictGBoostingM <- predict(modFitGBoostingM, newdata=Testing)
confMatGBoostingM <- confusionMatrix(predictGBoostingM, Testing$classe)
confMatGBoostingM

```
```r
##Confusion Matrix and Statistics
##
##          Reference
##Prediction    A    B    C    D    E
##         A 1668    8    0    0    0
##         B    6 1115   16    5    0
##         C    0   14 1005    9    2
##         D    0    2    4  949   11
##         E    0    0    1    1 1069
##
##Overall Statistics
##                                          
##               Accuracy : 0.9866          
##                 95% CI : (0.9833, 0.9894)
##    No Information Rate : 0.2845          
##    P-Value [Acc > NIR] : < 2.2e-16       
##                                          
##                  Kappa : 0.983           
## Mcnemar's Test P-Value : NA              
##
##Statistics by Class:
##
##                     Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9964   0.9789   0.9795   0.9844   0.9880
##Specificity            0.9981   0.9943   0.9949   0.9965   0.9996
##Pos Pred Value         0.9952   0.9764   0.9757   0.9824   0.9981
##Neg Pred Value         0.9986   0.9949   0.9957   0.9970   0.9973
##Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
##Detection Rate         0.2834   0.1895   0.1708   0.1613   0.1816
##Detection Prevalence   0.2848   0.1941   0.1750   0.1641   0.1820
##Balanced Accuracy      0.9973   0.9866   0.9872   0.9905   0.9938
```

## Applying the Selected Model to the Test Data

The accuracy of the 3 regression modeling methods above are:

    Decision Tree : 0.7217
    Random Forest : 0.9971
    GBM : 0.9866

# Plotting matrix results
```r
plot(confMatGBoostingM$table, col = confMatGBoostingM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBoostingM$overall['Accuracy'], 4)))

predictTEST <- predict(modleFitRandForest, newdata=NewTest)
predictTEST
```
```r
##[1] B A B A A E D B A A B C B A E E A B B B
##Levels: A B C D E
```
