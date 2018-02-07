# Prediction Assignment Writeup
Goutham Manghnani
Feb 02, 2018  

## Pre-processing Data
Several columns of the raw data set have string contaning nothing, so we delete those columns first, and we also delete the first 7 columns: X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window. These features are obviously not related to predict the outcome.


```r

install.packages("https://cran.r-project.org/bin/windows/contrib/3.3/RGtk2_2.20.31.zip", repos=NULL)
install.packages("e1071")

rm(list=ls())                # free up memory for the download of the data sets
setwd("~/Cursos/Data Science/08 Practical Machine Learning/Projeto")
library(knitr)
library(caret)
library(rpart)
library(e1071)
library(rattle)
library(randomForest)
library(corrplot)
library(rpart.plot)
```

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
> dim(TrainSet)
```
```r
13737   160
```

## Remove variables with Nearly Zero Variance


```r
NZV <- nearZeroVar(Training)
Training <- Training[, -NZV]
Testing  <- Testing[, -NZV]

```

## Remove variables that are mostly NA

```r
AllNA    <- sapply(Training, function(x) mean(is.na(x))) > 0.95
Training <- Training[, AllNA==FALSE]
Testing  <- Testing[, AllNA==FALSE]
```


## remove identification only variables (columns 1 to 5)

```r
Training <- Training[, -(1:5)]
Testing  <- Testing[, -(1:5)]

correlationMatrix <- cor(Training[, -54])
corrplot(correlationMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

# model fit

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

# prediction on Test dataset

```r
predictRandomForest <- predict(modleFitRandForest, newdata=Testing)
confMatRandomForest <- confusionMatrix(predictRandomForest, Testing$classe)
confMatRandomForest
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

# prediction on Test dataset

```r
predictGBoostingM <- predict(modFitGBoostingM, newdata=Testing)
confMatGBoostingM <- confusionMatrix(predictGBoostingM, Testing$classe)
confMatGBoostingM

```

# plot matrix results
```r
plot(confMatGBoostingM$table, col = confMatGBoostingM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBoostingM$overall['Accuracy'], 4)))

predictTEST <- predict(modleFitRandForest, newdata=NewTest)
predictTEST

```
