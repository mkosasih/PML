#**Predictions on Weight Lifting Exercises**
*by Marlina Kosasih*

###**1. Executive Summary**
Using devices such as Jawbone Up and Fitbit, datasets were collected from six participants to perform weight lifting exercises. The participants were asked to simulate the exact exercise and to simulate 4 common mistakes. The results were grouped into 5 classes: doing exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

This analysis is performed to predict the manner in which the participant did the exercise.  The data was collected and filtered to provide the most relevant information for the predictions.  Once it was cleaned, the data was separated into 60% train sets and 40% test sets for Cross Validations.  Decision tree method was first evaluated with only less than 60% accuracy, more than 40% out-of-sample error.  Random Forest method generated a better prediction model with 99.2%, less than 1% out-of-sample error

###**2. Inputing Data**
The analysis was prepared and performed in R version 3.1.2 or later.  Few required packages are listed below:

```r
require(caret); require(rpart.plot); library(RCurl)
```

2 Datasets are downloaded to current working directory.
These data sets are the training data for fitting a model and the testing data to test the model as well as to submit the result.

```r
WD <- getwd()

if(!file.exists(paste(WD,"/training.csv",sep=""))) {       
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "training.csv", method="curl") }
if(!file.exists(paste(WD,"/testing.csv",sep=""))) {
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","testing.csv",method="curl") }

training <- read.csv("training.csv", na.strings=c("NA","")); testing <- read.csv("testing.csv", na.strings=c("NA",""))

dim(training)
```

```
## [1] 19622   160
```

###**3. Pre-Processing Data**
The training dataset consist of 160 number of variables.  For fitting a prediction model, data must only consist of relevant predictors. The following are the steps to filter and clean the data.

**3.1. Removing irrelevant variables**

The first 7 columns in the data are only to inform the participants names/ID, time of exercises, etc.  These lists are not useful to predict the quality of participants activities.


```r
str(training[,1:7])
```

```
## 'data.frame':	19622 obs. of  7 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
```

```r
training <- training[,-c(1:7)]
```

**3.2. Removing Zero or Near Zero Variance**

The variables that have one unique value (zero variance) or somewhat unique values (near zero variance) will not be relevant for the predictions.  Therefore these variables will be removed.

```r
nzv <- nearZeroVar(training)
Ftraining <- training[, -nzv]
```

**3.4. Removing predictors with NA more than 60%**

To further retain relevant information for predictions, the data with many NA (more than 60%) will be omitted from the dataset.

```r
colRemove=0 ; j = 1
for(i in 1:ncol(Ftraining)) {
  if( sum( is.na( Ftraining[, i] ) ) /nrow(Ftraining) >= .6) {
    colRemove[j] <- i
    j = j+1  }
  } 

training <- Ftraining[,-colRemove]
dim(training)
```

```
## [1] 19622    53
```

Numbers of predictors are now reduced to 53

**3.4. Matching predictors in Testing dataset**

Once the training dataset was filtered and cleaned, the testing data set would only collect the same predictors as the training dataset.
Adding the last column 'problem_id' for the result submissions.


```r
colset <- colnames(training)
testing <- cbind(testing[,colnames(testing)%in% colset],testing$problem_id)
```

###**4. Cross Validation**
To avoid over fitting the model, cross validation is performed by splitting the data into 2 sets of data: train and test.
Since the dataset is considered medium sample size, the split between the train and test dataset was 60% and 40%.


```r
set.seed(123)
inTrain<- createDataPartition(y=training$classe, p=0.6, list=FALSE)
train <- training[inTrain,]
test <- training[-inTrain,]
```

###**5. Prediction Model Selection**
Since the data have many predictors, Decision Tree or Random Forests will be good methods to use; and they are both easy to interpret

**5.1. Model 1: Decision Tree**

A Decision Tree method was modeled based on the test dataset.

```r
modFit1 <- train(classe~., method="rpart", data=train)
print(modFit1$finalModel)
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 10776 7434 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.95 956    7 A (0.99 0.0073 0 0 0) *
##      5) pitch_forearm>=-33.95 9820 7427 A (0.24 0.23 0.21 0.2 0.12)  
##       10) yaw_belt>=169.5 478   47 A (0.9 0.046 0 0.042 0.01) *
##       11) yaw_belt< 169.5 9342 7092 B (0.21 0.24 0.22 0.2 0.12)  
##         22) magnet_dumbbell_z< -93.5 1104  447 A (0.6 0.28 0.043 0.058 0.023) *
##         23) magnet_dumbbell_z>=-93.5 8238 6231 C (0.16 0.24 0.24 0.22 0.14)  
##           46) pitch_belt< -42.95 491   75 B (0.014 0.85 0.084 0.026 0.029) *
##           47) pitch_belt>=-42.95 7747 5781 C (0.17 0.2 0.25 0.24 0.15)  
##             94) magnet_dumbbell_y< 290.5 3437 2046 C (0.19 0.12 0.4 0.16 0.12) *
##             95) magnet_dumbbell_y>=290.5 4310 3032 D (0.15 0.26 0.13 0.3 0.16) *
##    3) roll_belt>=130.5 1000    6 E (0.006 0 0 0 0.99) *
```

Checking the prediction model, the test dataset is used to evaluate the Out of Sample error and accuracy.


```r
prediction1 <- predict(modFit1, newdata=test)
confusionMatrix(prediction1,test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1356  239   42   68   20
##          B    5  264   34    8    6
##          C  441  263  907  397  269
##          D  422  752  385  813  510
##          E    8    0    0    0  637
## 
## Overall Statistics
##                                          
##                Accuracy : 0.5069         
##                  95% CI : (0.4958, 0.518)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3865         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6075  0.17391   0.6630   0.6322  0.44175
## Specificity            0.9343  0.99162   0.7885   0.6846  0.99875
## Pos Pred Value         0.7861  0.83281   0.3983   0.2821  0.98760
## Neg Pred Value         0.8569  0.83344   0.9172   0.9047  0.88821
## Prevalence             0.2845  0.19347   0.1744   0.1639  0.18379
## Detection Rate         0.1728  0.03365   0.1156   0.1036  0.08119
## Detection Prevalence   0.2199  0.04040   0.2902   0.3673  0.08221
## Balanced Accuracy      0.7709  0.58277   0.7258   0.6584  0.72025
```

**The accuracy of this model is very low: less than 60% and Expected out of sample error is more than 40%**

**5.2. Model 2: Random Forest**

Next, to find a better accuracy, the Random Forests prediction method was being evaluated.  
Bootstrap re-sampling in Random Forests can be very time consuming. To increase the speed of computations, the number of re-sampling iterations (k-folds) was set to 3. If the accuracy is low, k-fold can be increased.


```r
modFit2 <- train(classe~., data=train, method="rf", prox=TRUE, trControl = trainControl(method = "cv", number = 3))
print(modFit2$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.85%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3340    5    1    0    2 0.002389486
## B   16 2256    6    1    0 0.010092146
## C    0   15 2029   10    0 0.012171373
## D    0    2   27 1898    3 0.016580311
## E    0    1    4    7 2153 0.005542725
```

Checking the prediction model, the test dataset is used to evaluate the Out of Sample error and accuracy.

```r
prediction2 <- predict(modFit2, newdata=test)
confusionMatrix(prediction2,test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228   10    1    0    0
##          B    4 1505    8    0    0
##          C    0    3 1346   17    5
##          D    0    0   13 1268    5
##          E    0    0    0    1 1432
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9915          
##                  95% CI : (0.9892, 0.9934)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9892          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9914   0.9839   0.9860   0.9931
## Specificity            0.9980   0.9981   0.9961   0.9973   0.9998
## Pos Pred Value         0.9951   0.9921   0.9818   0.9860   0.9993
## Neg Pred Value         0.9993   0.9979   0.9966   0.9973   0.9984
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1918   0.1716   0.1616   0.1825
## Detection Prevalence   0.2854   0.1933   0.1747   0.1639   0.1826
## Balanced Accuracy      0.9981   0.9948   0.9900   0.9916   0.9965
```
**The accuracy of this model is very high: 99% and Expected out of sample error is 1%**

###**6. Conclusions**
The best model to predict participants exercise quality was the Random Forests model.  With only 3-fold bootstrap re-sampling, the model will predict with 99.2% accuracy and 1% out-of-sample error.
