# RVML_Project.Rmd


## Practical Machine Learning 
## Date : 4/23/2017 


## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Data:
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

## Goal:
The goal of the project is to predict the manner in which the exercise was done. This is the "classe" variable in the training set. Other variables may be used to predict with. A report describing how the model was built including how cross validation was used, the expected out of sample error and why the choices made were done. Also utilize the prediction model to predict 20 different test cases. 

## Loading data:

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.3.2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.3.3
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.3.3
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.3.3
```

```r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
dim(training) ; dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```
## Data Cleaning: 
Let's do some cleaning before we split the data.
Let's delete rows with missing values and columns that may be contextual and not provide any prediction

```r
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
dim(training); dim(testing)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

Now partition the dataset into 2 pieces for training and validating

```r
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
myTraining <- training[inTrain, ]
myvalidating <- training[-inTrain, ]
dim(myTraining); dim(myvalidating)
```

```
## [1] 13737    53
```

```
## [1] 5885   53
```
## Prediction Algorithms:
We will use classification trees, Neural Net and Random forests to predict the outcome with cross validation.

### Classification Trees with k-fold cross validation
Let's use the default 10 fold , 10 times cross validation here. 


```r
set.seed(123)
foldcontrol <- trainControl(method = "cv", number = 10, repeats = 10)
fitrpart <- train(classe ~ ., data = myTraining, method = "rpart", 
                   trControl = foldcontrol)
print(fitrpart, digits = 4)
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12363, 12363, 12362, 12363, 12364, ... 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa  
##   0.03560  0.5067    0.36159
##   0.06076  0.4145    0.20666
##   0.11443  0.3304    0.07043
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0356.
```


```r
fancyRpartPlot(fitrpart$finalModel)
```

![](RVMLProject_files/figure-html/unnamed-chunk-5-1.png)<!-- -->
####Predict outcomes using the "myvalidating " set

```r
predict_rpart <- predict(fitrpart, myvalidating)

## Show output
confusionMatrix(myvalidating$classe, predict_rpart)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1519   28  123    0    4
##          B  495  356  288    0    0
##          C  462   31  533    0    0
##          D  464  128  372    0    0
##          E  153  140  293    0  496
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4935          
##                  95% CI : (0.4806, 0.5063)
##     No Information Rate : 0.5256          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3376          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4911  0.52123  0.33126       NA  0.99200
## Specificity            0.9445  0.84948  0.88471   0.8362  0.89118
## Pos Pred Value         0.9074  0.31255  0.51949       NA  0.45841
## Neg Pred Value         0.6262  0.93110  0.77856       NA  0.99917
## Prevalence             0.5256  0.11606  0.27341   0.0000  0.08496
## Detection Rate         0.2581  0.06049  0.09057   0.0000  0.08428
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.7178  0.68536  0.60798       NA  0.94159
```

```r
confusionMatrix(myvalidating$classe, predict_rpart)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.4934579      0.3376374      0.4806055      0.5063169      0.5255735 
## AccuracyPValue  McnemarPValue 
##      0.9999996            NaN
```

```r
confusionMatrix(myvalidating$classe, predict_rpart)$overall[1]
```

```
##  Accuracy 
## 0.4934579
```
#### Observation on k-fold Cross validation 
The confustion Matrix shows an accuracy rate with K-fold cross validation (10 fold, 10 times) is only 0.49. So the out-of-sample error rate is 0.51. Using classification tree does not predict the outcome classe very well. 

### Neural Networks:
Now let's see what Neural Networks can do

```r
fit_nn <- train(classe ~ ., data = myTraining, method = "nnet", 
                   trControl = foldcontrol,  returnResamp = "all", verboseIter = FALSE)
```

```
## Loading required package: nnet
```
```r
print(fit_nn, digits = 4)
```

```
## Neural Network 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12364, 12363, 12363, 12363, 12364, ... 
## Resampling results across tuning parameters:
## 
##   size  decay  Accuracy  Kappa 
##   1     0e+00  0.3386    0.1259
##   1     1e-04  0.3326    0.1189
##   1     1e-01  0.3341    0.1282
##   3     0e+00  0.3901    0.2291
##   3     1e-04  0.3783    0.2066
##   3     1e-01  0.3949    0.2285
##   5     0e+00  0.4165    0.2587
##   5     1e-04  0.3826    0.2119
##   5     1e-01  0.3931    0.2212
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were size = 5 and decay = 0.
```
####Predict outcomes using the "myvalidating " set
Now let's look at the prediction on the validation set..


```r
predict_nn_validating <- predict(fit_nn, myvalidating)
conf_nn_validating <- confusionMatrix(myvalidating$classe, predict_nn_validating)
conf_nn_validating
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 704  68 420 439  43
##          B 221 388 104 297 129
##          C 262  89 354 251  70
##          D  44  37 210 617  56
##          E 174 324 106 395  83
## 
## Overall Statistics
##                                           
##                Accuracy : 0.3647          
##                  95% CI : (0.3523, 0.3771)
##     No Information Rate : 0.3397          
##     P-Value [Acc > NIR] : 3.023e-05       
##                                           
##                   Kappa : 0.2052          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5011  0.42826  0.29648   0.3087  0.21785
## Specificity            0.7835  0.84917  0.85675   0.9107  0.81850
## Pos Pred Value         0.4205  0.34065  0.34503   0.6400  0.07671
## Neg Pred Value         0.8335  0.89086  0.82712   0.7192  0.93796
## Prevalence             0.2387  0.15395  0.20289   0.3397  0.06474
## Detection Rate         0.1196  0.06593  0.06015   0.1048  0.01410
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.6423  0.63871  0.57661   0.6097  0.51817
```
####Observation on Neural Networks: 
Neural Networks gives an accuracy of .36 and the out-of-sample error is about .64. Looking at balanced accuracy for the various classes it ranges from .51 - .64; 

### Random Forest
Now lets see what Random forest can do.

```r
fit_rf <- train(classe ~ ., data = myTraining, method = "rf", 
                   trControl = foldcontrol)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
print(fit_rf, digits = 4)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12364, 12364, 12365, 12364, 12361, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9924    0.9904
##   27    0.9924    0.9903
##   52    0.9878    0.9845
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
####Predict outcomes using the "myvalidating " set
Now let's look at the prediction on the validation set..


```r
predict_rf_validating <- predict(fit_rf, myvalidating)
conf_rf_validating <- confusionMatrix(myvalidating$classe, predict_rf_validating)
conf_rf_validating
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B   10 1129    0    0    0
##          C    0    8 1018    0    0
##          D    0    0   15  949    0
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2858          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9941   0.9912   0.9855   0.9979   1.0000
## Specificity            0.9995   0.9979   0.9984   0.9970   0.9996
## Pos Pred Value         0.9988   0.9912   0.9922   0.9844   0.9982
## Neg Pred Value         0.9976   0.9979   0.9969   0.9996   1.0000
## Prevalence             0.2858   0.1935   0.1755   0.1616   0.1835
## Detection Rate         0.2841   0.1918   0.1730   0.1613   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9968   0.9946   0.9919   0.9974   0.9998
```
####Observation on Random Forest: 
The Random forest computationally was very expensive and was not efficient. For this dataset, random forest method is way better than classification tree method. The accuracy rate is 0.9942. So out-of-sample error is only about 0.01.


## Prediction on the Testing Data:

Having identified Random forest is better of the two - we can now run it on the test data. 

Let now predict the outcome on the testing set..


```r
predict_rf_testing <-predict(fit_rf, testing)
predict_rf_testing
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


