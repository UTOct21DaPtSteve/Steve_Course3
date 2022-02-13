# File: Course3Task2_Steve_02132022.R
# Project name: Predict which product brand customers prefer
#

################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("randomForest")
install.packages("klaR")
install.packages("doParallel")  # for Win parallel processing (see below) 
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)             # for Win
library(randomForest)

#####################
# Parallel processing
#####################

#--- for Win ---#

detectCores()          # detect number of cores
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

##############
# Import data 
##############

##-- Load Train/Existing data (Dataset 1  CompleteResponses.csv) --##

pwd <- getwd()
responseCompOOB <- read.csv(paste(pwd, "/CompleteResponses.csv", sep = ""), stringsAsFactors = FALSE)

################
# Evaluate data
################

##--- Dataset 1 ---##

str(responseCompOOB)
#'data.frame':	9898 obs. of  7 variables:
#  $ salary : num  119807 106880 78021 63690 50874 ...
#$ age    : int  45 63 23 51 20 56 24 62 29 41 ...
#$ elevel : int  0 1 0 3 3 3 4 3 4 1 ...
#$ car    : int  14 11 15 6 14 14 8 3 17 5 ...
#$ zipcode: int  4 6 2 5 4 3 5 0 0 4 ...
#$ credit : num  442038 45007 48795 40889 352951 ...
#$ brand  : int  0 1 0 1 0 1 1 1 0 1 ...

# view first/last obs/rows
head(responseCompOOB)
#     salary age elevel car zipcode    credit brand
#1 119806.54  45      0  14       4 442037.71     0
#2 106880.48  63      1  11       6  45007.18     1
#3  78020.75  23      0  15       2  48795.32     0
#4  63689.94  51      3   6       5  40888.88     1
#5  50873.62  20      3  14       4 352951.50     0
#6 130812.74  56      3  14       3 135943.02     1
 
tail(responseCompOOB)

#        salary age elevel car zipcode   credit brand
#9893  28751.26  60      2  10       0      0.0     1
#9894  87580.91  75      1  18       8 282511.9     1
#9895 129181.38  75      2   7       4 384871.4     1
#9896  97828.09  66      2  15       0 399446.7     1
#9897  20000.00  24      1  14       1 223204.6     1
#9898  96430.16  34      1   2       7 224029.8     0

# check for missing values 
anyNA(responseCompOOB)
#[1] FALSE

# check for duplicates
anyDuplicated((responseCompOOB))
#[1] 0

##--- Load Predict/New data (Dataset 2  SurveyIncomplete.csv) [dv = NA, 0, or Blank] ---##

surveyIncompOOB <- read.csv(paste(pwd, "/SurveyIncomplete.csv", sep = ""), stringsAsFactors = FALSE)

str(surveyIncompOOB)
#'data.frame':	5000 obs. of  7 variables:
#  $ salary : num  150000 82524 115647 141443 149211 ...
#$ age    : int  76 51 34 22 56 26 64 50 26 46 ...
#$ elevel : int  1 1 0 3 0 4 3 3 2 3 ...
#$ car    : int  3 8 10 18 5 12 1 9 3 18 ...
#$ zipcode: int  3 3 2 2 3 1 2 0 4 6 ...
#$ credit : num  377980 141658 360980 282736 215667 ...
#$ brand  : int  1 0 1 1 1 1 1 1 1 0 ...

# view first/last obs/rows
head(surveyIncompOOB)
#     salary age elevel car zipcode   credit brand
#1 150000.00  76      1   3       3 377980.1     1
#2  82523.84  51      1   8       3 141657.6     0
#3 115646.64  34      0  10       2 360980.4     1
#4 141443.39  22      3  18       2 282736.3     1
#5 149211.27  56      0   5       3 215667.3     1
#6  46202.25  26      4  12       1 150419.4     1

tail(surveyIncompOOB)
#        salary age elevel car zipcode    credit brand
#4995  29945.49  75      2   9       1 170179.21     0
#4996  83891.56  52      2  14       5  28685.23     0
#4997 125979.29  71      0  12       7 276614.83     0
#4998  74064.71  24      2   2       2 202279.58     0
#4999 106485.57  46      3  16       0 381242.09     0
#5000  50333.58  70      1   5       5 224871.17     0

# check for missing values 
anyNA(surveyIncompOOB) 
#[1] FALSE

# do we expect the dv to have any blank/na values?
## The NA check above did not find missing values. But if there are any missing values, do the following:

## Replacing them with mean/mode.
## Replacing them with a constant say -1.
## Using classifier models to predict them.

# can we evaluate if any NA in just the dv?

## Yes,  R provides various packages for missing value imputation like kNN
##  https://cran.r-project.org/doc/contrib/de_Jonge+van_der_Loo-Introduction_to_data_cleaning_with_R.pdf

# can we evaluate if any NA in all vars excluding the dv?

## Use KNN Imputation or if necessary, remove the missing data rows.

# check for duplicates
anyDuplicated(surveyIncompOOB)
#[1] 0

#############
# Preprocess
#############

##--- Dataset 1 ---##

# remove ID and obvious features
responseCompOOB$feature_to_remove <- NULL   # remove ID, if applicable
str(responseCompOOB) # confirm removed features
#'data.frame':	9898 obs. of  7 variables:
#  $ salary : num  119807 106880 78021 63690 50874 ...
#$ age    : int  45 63 23 51 20 56 24 62 29 41 ...
#$ elevel : int  0 1 0 3 3 3 4 3 4 1 ...
#$ car    : int  14 11 15 6 14 14 8 3 17 5 ...
#$ zipcode: int  4 6 2 5 4 3 5 0 0 4 ...
#$ credit : num  442038 45007 48795 40889 352951 ...
#$ brand  : int  0 1 0 1 0 1 1 1 0 1 ...

  
# rename a column
names(responseCompOOB) <- c("Salary","Age","EducationLevel", "CarType", "Zipcode", "CreditLimit", "Brand") 

# confirm column names
str(responseCompOOB)
#'data.frame':	9898 obs. of  7 variables:
#  $ Salary        : num  119807 106880 78021 63690 50874 ...
#$ Age           : int  45 63 23 51 20 56 24 62 29 41 ...
#$ EducationLevel: int  0 1 0 3 3 3 4 3 4 1 ...
#$ CarType       : int  14 11 15 6 14 14 8 3 17 5 ...
#$ Zipcode       : int  4 6 2 5 4 3 5 0 0 4 ...
#$ CreditLimit   : num  442038 45007 48795 40889 352951 ...
#$ Brand         : int  0 1 0 1 0 1 1 1 0 1 ...

# change data types
## responseCompOOB$ColumnName <- as.typeofdata(responseCompOOB$ColumnName)
# handle missing values (if applicable) 
## na.omit(responseCompOOB$ColumnName)
## na.exclude(responseCompOOB$ColumnName)        
## responseCompOOB$ColumnName[is.na(responseCompOOB$ColumnName)] <- mean(responseCompOOB$ColumnName,na.rm = TRUE)


##--- Dataset 2 ---##

# If there is a dataset with unseen data to make predictions on, then preprocess 
# here to make sure that it is preprossed the same as the dataset that had
# the best results - e.g., oob or a tuned ds from feature selection/engineering.

surveyIncompOOB$feature_to_remove <- NULL   # remove ID, if applicable

# rename a column
names(surveyIncompOOB) <- c("Salary","Age","EducationLevel", "CarType", "Zipcode", "CreditLimit", "Brand") 

str(surveyIncompOOB) # confirm removed features and column name changes
#'data.frame':	5000 obs. of  7 variables:
#$ Salary        : num  150000 82524 115647 141443 149211 ...
#$ Age           : int  76 51 34 22 56 26 64 50 26 46 ...
#$ EducationLevel: int  1 1 0 3 0 4 3 3 2 3 ...
#$ CarType       : int  3 8 10 18 5 12 1 9 3 18 ...
#$ Zipcode       : int  3 3 2 2 3 1 2 0 4 6 ...
#$ CreditLimit   : num  377980 141658 360980 282736 215667 ...
#$ Brand         : int  1 0 1 1 1 1 1 1 1 0 ...


#####################
# EDA/Visualizations
#####################

# statistics
summary(responseCompOOB)
#    Salary            Age        EducationLevel     CarType         Zipcode       CreditLimit         Brand       
#Min.   : 20000   Min.   :20.00   Min.   :0.000   Min.   : 1.00   Min.   :0.000   Min.   :     0   Min.   :0.0000  
#1st Qu.: 52082   1st Qu.:35.00   1st Qu.:1.000   1st Qu.: 6.00   1st Qu.:2.000   1st Qu.:120807   1st Qu.:0.0000  
#Median : 84950   Median :50.00   Median :2.000   Median :11.00   Median :4.000   Median :250607   Median :1.0000  
#Mean   : 84871   Mean   :49.78   Mean   :1.983   Mean   :10.52   Mean   :4.041   Mean   :249176   Mean   :0.6217  
#3rd Qu.:117162   3rd Qu.:65.00   3rd Qu.:3.000   3rd Qu.:15.75   3rd Qu.:6.000   3rd Qu.:374640   3rd Qu.:1.0000  
#Max.   :150000   Max.   :80.00   Max.   :4.000   Max.   :20.00   Max.   :8.000   Max.   :500000   Max.   :1.0000 

# plots
hist(responseCompOOB$Brand)
# See the image file Brand_Hist.png

plot(responseCompOOB$Salary, responseCompOOB$Brand)
# See the image file Plot_Salary_Brand.png

plot(responseCompOOB$Age, responseCompOOB$Brand)
# See the image file Plot_Age_Brand.png

plot(responseCompOOB$EducationLevel, responseCompOOB$Brand)
# See the image file Plot_EducationLevel_Brand.png

plot(responseCompOOB$CarType, responseCompOOB$Brand)
# See the image file Plot_CarType_Brand.png

plot(responseCompOOB$Zipcode, responseCompOOB$Brand)
# See the image file Plot_Zipcode_Brand.png

plot(responseCompOOB$Credit, responseCompOOB$Brand)
# See the image file Plot_Credit_Brand.png

qqnorm(responseCompOOB$Brand) # Be familiar with this plot, but don't spend a lot of time on it
# See the image file qqnorm_Brand.png

################
# Sampling
################

# create 10% sample 
set.seed(1) # set random seed
responseCompOOB10p <- responseCompOOB[sample(1:nrow(responseCompOOB), round(nrow(responseCompOOB)*.1),replace=FALSE),]
nrow(responseCompOOB10p)
# 990
head(responseCompOOB10p) # ensure randomness
#        Salary Age EducationLevel CarType Zipcode CreditLimit Brand
#1017  64071.24  20              1       1       0    10309.12     0
#8004 149567.98  41              4       4       2   271248.82     1
#4775  62627.06  68              2       2       2   338219.55     0
#9725  50516.92  56              0      10       6    51021.03     1
#8462  21887.73  80              1      20       4   121553.78     1
#4050 127288.14  29              3       5       2   133670.69     1

# 1k sample
set.seed(1)       # set random seed
responseCompOOB1k <- responseCompOOB[sample(1:nrow(responseCompOOB), 1000, replace=FALSE),]
nrow(responseCompOOB1k) # ensure number of obs
# 1000
head(responseCompOOB1k) # ensure randomness
#        Salary Age EducationLevel CarType Zipcode CreditLimit Brand
#1017  64071.24  20              1       1       0    10309.12     0
#8004 149567.98  41              4       4       2   271248.82     1
#4775  62627.06  68              2       2       2   338219.55     0
#9725  50516.92  56              0      10       6    51021.03     1
#8462  21887.73  80              1      20       4   121553.78     1
#4050 127288.14  29              3       5       2   133670.69     1

#######################
# Feature selection
#######################

#######################
# Correlation analysis
#######################

# for regression problems, the below rules apply.
# 1) compare each IV to the DV, if cor > 0.95, remove
# 2) compare each pair of IVs, if cor > 0.90, remove the
#    IV that has the lowest cor to the DV. (see code
#    below for setting a threshold to automaticall select
#    IVs that are highly correlated)

# for classification problems, the below rule applies.
# 1) compare each pair of IVs, if cor > 0.90, remove one
#    of the IVs. (see code below to do this programmatically)

# calculate correlation matrix for all vars
corrAll <- cor(responseCompOOB1k[,1:7])
# view the correlation matrix
corrAll
#                     Salary          Age EducationLevel    CarType      Zipcode CreditLimit       Brand
#Salary          1.000000000 -0.006894664    -0.01824619 0.02246635  0.001246535 -0.05285718  0.20638127
#Age            -0.006894664  1.000000000    -0.01636817 0.03767539  0.052630591  0.01536689  0.09607296
#EducationLevel -0.018246187 -0.016368167     1.00000000 0.02179891  0.052784659 -0.04430464 -0.01226768
#CarType         0.022466345  0.037675391     0.02179891 1.00000000  0.023603702  0.03257828  0.05145255
#Zipcode         0.001246535  0.052630591     0.05278466 0.02360370  1.000000000  0.06009704 -0.01275105
#CreditLimit    -0.052857183  0.015366889    -0.04430464 0.03257828  0.060097036  1.00000000 -0.03897453
#Brand           0.206381274  0.096072962    -0.01226768 0.05145255 -0.012751053 -0.03897453  1.00000000

# plot correlation matrix
corrplot(corrAll, method = "circle")
# See the image file Corrplot_correlation_matrix_circle.png

corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
# See the image file Corrplot_correlation_matrix_hclust.png

# find IVs that are highly corrected (ideally >0.90)
corrIV <- cor(responseCompOOB1k[,1:6])
# view the correlation matrix of IVs
corrIV
#                     Salary          Age EducationLevel    CarType     Zipcode CreditLimit
#Salary          1.000000000 -0.006894664    -0.01824619 0.02246635 0.001246535 -0.05285718
#Age            -0.006894664  1.000000000    -0.01636817 0.03767539 0.052630591  0.01536689
#EducationLevel -0.018246187 -0.016368167     1.00000000 0.02179891 0.052784659 -0.04430464
#CarType         0.022466345  0.037675391     0.02179891 1.00000000 0.023603702  0.03257828
#Zipcode         0.001246535  0.052630591     0.05278466 0.02360370 1.000000000  0.06009704
#CreditLimit    -0.052857183  0.015366889    -0.04430464 0.03257828 0.060097036  1.00000000

# create object with indexes of highly corr features 
corrIVhigh <- findCorrelation(corrIV, cutoff=0.8)  # used 0.8 for illustration purposes  
# print indexes of highly correlated attributes
corrIVhigh
#integer(0)
# get var name of high corr IV
colnames(responseCompOOB1k[corrIVhigh])  # Cannot find one since all of them are way lower than 0.90 
#character(0)

str(responseCompOOB1k)
#'data.frame':	1000 obs. of  7 variables:
# $ Salary        : num  64071 149568 62627 50517 21888 ...
# $ Age           : int  20 41 68 56 80 29 41 56 36 67 ...
# $ EducationLevel: int  1 4 2 0 1 3 3 2 1 4 ...
# $ CarType       : int  1 4 2 10 20 5 19 18 11 7 ...
# $ Zipcode       : int  0 2 2 6 4 2 0 3 2 1 ...
# $ CreditLimit   : num  10309 271249 338220 51021 121554 ...
# $ Brand         : int  0 1 0 1 1 1 1 1 0 1 ...

# remove highly correlated features, if applicable
responseCompCOR1k <- responseCompOOB1k[-corrIVhigh] 
str(responseCompCOR1k)
#'data.frame':	1000 obs. of  0 variables

############
# caret RFE 
############

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees

## 1. ---- lm ---- ##

# define refControl using a linear model selection function (regression only)
LMcontrol <- rfeControl(functions=lmFuncs, method="cv", number=10)
# run the RFE algorithm
set.seed(7)
# Set up the cluster runtime enviroment.
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
rfeLM <- rfe(responseCompOOB1k[,1:6], responseCompOOB1k[,7], sizes=c(1:6), rfeControl=LMcontrol)
rfeLM
#Recursive feature selection
#
#Outer resampling method: Cross-Validated (10 fold) 
#
#Resampling performance over subset size:
#  
#  Variables   RMSE Rsquared    MAE  RMSESD RsquaredSD   MAESD Selected
#          1 0.4888 0.011637 0.4746 0.01482   0.011164 0.01266         
#          2 0.4884 0.006132 0.4736 0.01524   0.009789 0.01325         
#          3 0.4874 0.007949 0.4718 0.01440   0.010210 0.01247         
#          4 0.4860 0.015320 0.4695 0.01433   0.017567 0.01248         
#          5 0.4758 0.048436 0.4497 0.01544   0.028524 0.01568        *
#          6 0.4759 0.047810 0.4495 0.01524   0.027241 0.01579         
#
#The top 5 variables (out of 5):
#  CarType, EducationLevel, Zipcode, Age, Salary

# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

# plot the results
plot(rfeLM, type=c("g", "o"))
# See the plot in the image "rfeLM_plot.png"

# show predictors used
predictors(rfeLM) 
#[1] "CarType"        "EducationLevel" "Zipcode"        "Age"            "Salary"

# show var importance
varImp(rfeLM)
#                    Overall
#CarType        3.831903e-03
#EducationLevel 3.318832e-03
#Zipcode        3.243169e-03
#Age            2.655066e-03
#Salary         2.648084e-06

## 2. ---- rf ---- ##

# define the control using a random forest selection function (regression or classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, repeats=1)
# run the RFE algorithm
set.seed(7)
# Set up the cluster runtime enviroment.
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
rfeRF <- rfe(responseCompOOB1k[,1:6], responseCompOOB1k[,7], sizes=c(1:6), rfeControl=RFcontrol)
rfeRF
#Recursive feature selection
#
#Outer resampling method: Cross-Validated (5 fold) 
#
#Resampling performance over subset size:
#  
#  Variables   RMSE Rsquared    MAE  RMSESD RsquaredSD   MAESD Selected
#          1 0.4800   0.1313 0.3536 0.01320    0.03259 0.01895         
#          2 0.2534   0.7264 0.1385 0.03827    0.08379 0.02330        *
#          3 0.3116   0.6467 0.2555 0.02089    0.09101 0.01650         
#          4 0.3504   0.5592 0.3073 0.01957    0.09731 0.02073         
#          5 0.3828   0.4590 0.3465 0.01545    0.07935 0.01818         
#          6 0.3237   0.6044 0.2684 0.01995    0.07093 0.01946         
#
#The top 2 variables (out of 2):
#   Salary, Age

# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

# plot the results
plot(rfeRF, type=c("g", "o"))
# See the plot as the image file "rfeRF_Plot.png"

# show predictors used
predictors(rfeRF)
#[1] "Salary" "Age" 

# Note results.  
varImp(rfeRF)
#Overall
#Salary 79.96405
#Age    50.97920

##--- create ds with features using varImp from top model ---##

# create ds with predictors from varImp   
responseCompRFE1k <- responseCompOOB1k[,predictors(rfeRF)]
str(responseCompRFE1k)
#'data.frame':	1000 obs. of  2 variables:
# $ Salary: num  64071 149568 62627 50517 21888 ...
# $ Age   : int  20 41 68 56 80 29 41 56 36 67 ...

# add dv
responseCompRFE1k$Brand <- responseCompOOB1k$Brand
# confirm new ds
str(responseCompRFE1k)
#'data.frame':	1000 obs. of  3 variables:
# $ Salary: num  64071 149568 62627 50517 21888 ...
# $ Age   : int  20 41 68 56 80 29 41 56 36 67 ...
# $ Brand : int  0 1 0 1 1 1 1 1 0 1 ...


## 3. ---- treebagFuncs ---- ##

# define the control using a treebagFuncs selection function (regression or classification)
TBcontrol <- rfeControl(functions=treebagFuncs, method="cv", number=5, repeats=1)
# run the RFE algorithm
set.seed(7)
# Set up the cluster runtime enviroment.
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
tbeRF <- rfe(responseCompOOB1k[,1:6], responseCompOOB1k[,7], sizes=c(1:6), rfeControl=TBcontrol )
tbeRF
#Recursive feature selection
#
#Outer resampling method: Cross-Validated (5 fold) 
#
#Resampling performance over subset size:
#  
#  Variables   RMSE Rsquared    MAE  RMSESD RsquaredSD   MAESD Selected
#          1 0.4762   0.0482 0.4487 0.02227    0.09273 0.04797         
#          2 0.2524   0.7285 0.1332 0.03857    0.08411 0.02634         
#          3 0.2496   0.7358 0.1373 0.03440    0.07534 0.02273        *
#          4 0.2549   0.7279 0.1449 0.03214    0.07535 0.01919         
#          5 0.2528   0.7285 0.1455 0.04250    0.09462 0.02868         
#          6 0.2502   0.7340 0.1392 0.03646    0.07982 0.02502         
#
#The top 3 variables (out of 3):
#  Age, Salary, CreditLimit

# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

# plot the results
plot(nbeRF, type=c("g", "o"))
# See the plot as the image file "rfeRF_Plot.png"

# show predictors used
predictors(tbeRF)
#[1] "Age"         "Salary"      "CreditLimit" 

# Note results.  
varImp(tbeRF)
#             Overall
#Age         3.584634
#Salary      3.466982
#CreditLimit 1.220826

##--- create ds with features using varImp from top model ---##

# create ds with predictors from varImp   
responseCompRFE1k <- responseCompOOB1k[,predictors(tbeRF)]
str(responseCompRFE1k)
#'data.frame':	1000 obs. of  3 variables:
# $ Age        : int  20 41 68 56 80 29 41 56 36 67 ...
# $ Salary     : num  64071 149568 62627 50517 21888 ...
# $ CreditLimit: num  10309 271249 338220 51021 121554 ...

# add dv
responseCompRFE1k$Brand <- responseCompOOB1k$Brand
# confirm new ds
str(responseCompRFE1k)
#'data.frame':	1000 obs. of  4 variables:
# $ Age        : int  20 41 68 56 80 29 41 56 36 67 ...
# $ Salary     : num  64071 149568 62627 50517 21888 ...
# $ CreditLimit: num  10309 271249 338220 51021 121554 ...
# $ Brand      : int  0 1 0 1 1 1 1 1 0 1 ...



##############################
# Feature engineering
##############################

# code goes here, when applicable




##################
# Train/test sets
##################

# responseCompOOB1k
set.seed(123) 
inTraining <- createDataPartition(responseCompOOB1k$Brand, p=0.75, list=FALSE)
oobTrain <- responseCompOOB1k[inTraining,]   
oobTest <- responseCompOOB1k[-inTraining,]   
# verify number of obs 
nrow(oobTrain) 
# 750
nrow(oobTest)  
# 250

# responseCompCOR1k
set.seed(123) 
cor1kinTraining <- createDataPartition(responseCompCOR1k$Brand, p=0.75, list=FALSE)
cor1kTrain <- responseCompCOR1k[cor1kinTraining,]   
cor1kTest <- responseCompCOR1k[-cor1kinTraining,]   
# verify number of obs 
nrow(cor1kTrain) 
# 750
nrow(cor1kTest)  
# 250



# responseCompRFE1k
set.seed(123) 
rfe1kinTraining <- createDataPartition(responseCompRFE1k$Brand, p=0.75, list=FALSE)
rfe1kTrain <- responseCompRFE1k[rfe1kinTraining,]   
rfe1kTest <- responseCompRFE1k[-rfe1kinTraining,]   
# verify number of obs 
nrow(rfe1kTrain) 
# 750
nrow(rfe1kTest)  
# 250

################
# Train control
################

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1) 


###############
# Train models
###############

?modelLookup()
modelLookup('rf')

## ------- RF ------- ##

# default
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
set.seed(123)
oobRFfit <- train(Brand~., data=oobTrain, method="rf", importance=T, trControl=fitControl)
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

# manual grid
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit <- train(Brand~.,data=oobTrain,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

oobRFfit
#Random Forest 
#
#750 samples
#  6 predictor
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 600, 600, 600, 600, 600 
#Resampling results across tuning parameters:
#  
#  mtry  RMSE       Rsquared   MAE      
#  4     0.2688974  0.7030043  0.1828996
#  5     0.2565340  0.7233237  0.1566168
#  6     0.2537390  0.7250990  0.1402478
#
#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 6.

plot(oobRFfit)
# See the image as file "oobRFfit_plot.png"

varImp(oobRFfit)
#rf variable importance
#
#               Overall
#Salary         100.000
#Age             73.742
#Zipcode          4.339
#CreditLimit      2.574
#EducationLevel   1.019
#CarType          0.000

## ------- LM ------- ##

# LM train/fit
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
set.seed(123)
oobLMfit <- train(Brand~., data=oobTrain, method="lm", trControl=fitControl)

oobLMfit
#
#Linear Regression 
#
#750 samples
#6 predictor
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 600, 600, 600, 600, 600 
#Resampling results:
#  
#  RMSE       Rsquared    MAE     
#0.4742802  0.04405186  0.445075
#
#Tuning parameter 'intercept' was held constant at a value of TRUE

# eval variable importance
varImp(oobLMfit)
#lm variable importance
#
#               Overall
#Salary         100.000
#Age             43.974
#CarType          9.270
#CreditLimit      7.099
#Zipcode          4.415
#EducationLevel   0.000

# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)


##  -----Bagged Tree-----------------##


# default
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
set.seed(123)
oobTBfit <- train(Brand~., data=oobTrain, method="treebag", importance=T, trControl=fitControl)
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

oobTBfit
#Bagged CART 
#
#750 samples
#6 predictor
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 600, 600, 600, 600, 600 
#Resampling results:
#  
#  RMSE       Rsquared   MAE      
#  0.2464781  0.7444012  0.1433189



varImp(oobTBfit)
#treebag variable importance
#
#               Overall
#Salary         100.000
#Age             90.546
#CreditLimit     17.204
#CarType          6.977
#Zipcode          6.479
#EducationLevel   0.000

##################
# Model selection
##################

#-- responseCompOOB1k --# 

oobFitComp1k <- resamples(list(rf=oobRFfit, lm=oobLMfit, tb=oobTBfit))
# output summary metrics for tuned models 
summary(oobFitComp1k)
#Call:
#summary.resamples(object = oobFitComp1k)
#
#Models: rf, lm, tb 
#Number of resamples: 5 
#
#MAE 
#        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf 0.1114031 0.1378313 0.1448427 0.1402478 0.1530427 0.1541191    0
#lm 0.4365543 0.4432114 0.4433849 0.4450750 0.4498049 0.4524196    0
#tb 0.1229124 0.1383330 0.1453603 0.1433189 0.1510884 0.1589006    0
#
#RMSE 
#       Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf 0.1922389 0.2385066 0.2668726 0.2537390 0.2827131 0.2883637    0
#lm 0.4633160 0.4712680 0.4744568 0.4742802 0.4797396 0.4826207    0
#tb 0.2119504 0.2150328 0.2522703 0.2464781 0.2709628 0.2821740    0
#
#Rsquared 
#         Min.    1st Qu.     Median       Mean    3rd Qu.       Max. NA's
#rf 0.64587109 0.65810140 0.70825325 0.72509896 0.76115009 0.85211897    0
#lm 0.01848253 0.02256262 0.02949616 0.04405186 0.07056845 0.07914952    0
#tb 0.67274064 0.67604343 0.74376146 0.74440120 0.81200442 0.81745603    0

#-- responseCompCOR1k --#



#-- responseCompRFE1k --#


##--- Save/load top performing model ---##

# save top performing model after validation
saveRDS(oobRFfit, "oobRFfit.rds")  

# load and name model
rfFit1 <- readRDS("oobRFfit.rds")

############################
# Predict testSet/validation
############################

# -----predict with RF----
rfPred1 <- predict(oobRFfit, oobTest)
# performace measurment
postResample(rfPred1, oobTest$Brand)
#     RMSE  Rsquared       MAE 
#0.2470670 0.7499075 0.1295972 

# plot predicted verses actual
plot(rfPred1, oobTest$Brand)
# See the plot image as "Plot_Brand_Prediction_vs_actual.png"

# -----predict with TreeBag----
tbPred1 <- predict(oobTBfit, oobTest)
# performace measurment
postResample(tbPred1, oobTest$Brand)
#     RMSE  Rsquared       MAE 
#0.2504875 0.7424870 0.1326436 

# plot predicted verses actual
plot(tbPred1, oobTest$Brand)
# See the plot image as "TreeBag_predicted_vs_actual.png"

###############################
# Predict new data (Dataset 2)
###############################

# predict for new dataset with no values for DV

rfPredOOB <- predict(oobRFfit, surveyIncompOOB)

head(rfPredOOB)
#         1          2          3          4          5          6 
#1.00000000 0.02996667 0.96980000 1.00000000 0.99800000 0.99576667 

# -------------Predict for new dataset with Treebag

# predict for new dataset with no values for DV

tbPredOOB <- predict(oobTBfit, surveyIncompOOB)

head(tbPredOOB)
#[1] 0.98693861 0.06886471 0.98428977 0.98693861 0.98693861 0.95170647


