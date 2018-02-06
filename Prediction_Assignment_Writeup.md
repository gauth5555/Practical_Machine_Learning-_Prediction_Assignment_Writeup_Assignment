# Prediction Assignment Writeup
Goutham Manghnani
Feb 02, 2018  

## Pre-processing Data
Several columns of the raw data set have string contaning nothing, so we delete those columns first, and we also delete the first 7 columns: X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window. These features are obviously not related to predict the outcome.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12463)

training <- read.csv("pml-training.csv", stringsAsFactors=FALSE)
training$classe <- as.factor(training$classe)
training <- training[,-nearZeroVar(training)]
training <- training[,-c(1,2,3,4,5,6,7)]
```


There are many NA values in the data set, so we use KnnImpute method to impute those values. Besides, we try to standardize each features and use PCA to reduce features.


```r
inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
training <- training[inTrain,]
testing <- training[-inTrain,]

preObj <- preProcess(training[,-length(training)],method=c("center", "scale", "knnImpute", "pca"), thresh=0.9)
clean_data <- predict(preObj,training[,-length(training)])
```

## Prediction

After getting the clean data set from the above processing, we use Knn method to build model. We use testing data to evaluate the performance of our model. The accuracy is 0.9748. 


```r
modelFit <- train(training$classe ~.,data=clean_data, method="knn")
test <- predict(preObj, testing[,-length(testing)])
confusionMatrix(testing$classe, predict(modelFit,test))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1019    3    8    2    0
##          B   14  709   13    1    0
##          C    6   10  607    7    2
##          D    2    1   16  581    1
##          E    0    5    2    0  677
## 
## Overall Statistics
##                                        
##                Accuracy : 0.975        
##                  95% CI : (0.969, 0.98)
##     No Information Rate : 0.282        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.968        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.979    0.974    0.940    0.983    0.996
## Specificity             0.995    0.991    0.992    0.994    0.998
## Pos Pred Value          0.987    0.962    0.960    0.967    0.990
## Neg Pred Value          0.992    0.994    0.987    0.997    0.999
## Prevalence              0.282    0.198    0.175    0.160    0.184
## Detection Rate          0.276    0.192    0.165    0.158    0.184
## Detection Prevalence    0.280    0.200    0.171    0.163    0.186
## Balanced Accuracy       0.987    0.982    0.966    0.988    0.997
```


Finally, we load the testing data file and predict the reult as the following:

```r
testing <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)
testing <- testing[,names(testing) %in% names(training)]

test <- predict(preObj, testing)
predict_result <- predict(modelFit, test)
```
