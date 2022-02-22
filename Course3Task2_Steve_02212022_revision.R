# File: Course3Task2_Steve_02202022_revision.R
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
install.packages("tidyverse")
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)             # for Win
library(randomForest)
library(dplyr)
library(tidyverse)

#####################
# Parallel processing
#####################

#--- for Win ---#

detectCores()          # detect number of cores
cl <- makeCluster(4)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
#stopCluster(cl)

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

####################################################################
#  In order to support classification problem type data analysis.
#  Change the brand variable to be factor.
####################################################################

responseCompOOB$brand <- as.factor(responseCompOOB$brand)

# Verify

str(responseCompOOB)
#'data.frame':	9898 obs. of  7 variables:
#  $ salary : num  119807 106880 78021 63690 50874 ...
#$ age    : int  45 63 23 51 20 56 24 62 29 41 ...
#$ elevel : int  0 1 0 3 3 3 4 3 4 1 ...
#$ car    : int  14 11 15 6 14 14 8 3 17 5 ...
#$ zipcode: int  4 6 2 5 4 3 5 0 0 4 ...
#$ credit : num  442038 45007 48795 40889 352951 ...
#$ brand  : Factor w/ 2 levels "0","1": 1 2 1 2 1 2 2 2 1 2 ...

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

####################################################################
#  In order to support classification problem type data analysis.
#  Change the brand variable to be factor.
####################################################################

surveyIncompOOB$brand <- as.factor(surveyIncompOOB$brand)

# Verify

str(surveyIncompOOB)

#'data.frame':	5000 obs. of  7 variables:
#  $ salary : num  150000 82524 115647 141443 149211 ...
#$ age    : int  76 51 34 22 56 26 64 50 26 46 ...
#$ elevel : int  1 1 0 3 0 4 3 3 2 3 ...
#$ car    : int  3 8 10 18 5 12 1 9 3 18 ...
#$ zipcode: int  3 3 2 2 3 1 2 0 4 6 ...
#$ credit : num  377980 141658 360980 282736 215667 ...
#$ brand  : Factor w/ 2 levels "0","1": 2 1 2 2 2 2 2 2 2 1 ...

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
#$ brand  : Factor w/ 2 levels "0","1": 1 2 1 2 1 2 2 2 1 2 ...

  
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
#$ Brand         : Factor w/ 2 levels "0","1": 1 2 1 2 1 2 2 2 1 2 ...

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
#  $ Salary        : num  150000 82524 115647 141443 149211 ...
#$ Age           : int  76 51 34 22 56 26 64 50 26 46 ...
#$ EducationLevel: int  1 1 0 3 0 4 3 3 2 3 ...
#$ CarType       : int  3 8 10 18 5 12 1 9 3 18 ...
#$ Zipcode       : int  3 3 2 2 3 1 2 0 4 6 ...
#$ CreditLimit   : num  377980 141658 360980 282736 215667 ...
#$ Brand         : Factor w/ 2 levels "0","1": 2 1 2 2 2 2 2 2 2 1 ...


#####################
# EDA/Visualizations
#####################

# statistics
summary(responseCompOOB)
#    Salary            Age        EducationLevel     CarType         Zipcode       CreditLimit     Brand   
#Min.   : 20000   Min.   :20.00   Min.   :0.000   Min.   : 1.00   Min.   :0.000   Min.   :     0   0:3744  
#1st Qu.: 52082   1st Qu.:35.00   1st Qu.:1.000   1st Qu.: 6.00   1st Qu.:2.000   1st Qu.:120807   1:6154  
#Median : 84950   Median :50.00   Median :2.000   Median :11.00   Median :4.000   Median :250607           
#Mean   : 84871   Mean   :49.78   Mean   :1.983   Mean   :10.52   Mean   :4.041   Mean   :249176           
#3rd Qu.:117162   3rd Qu.:65.00   3rd Qu.:3.000   3rd Qu.:15.75   3rd Qu.:6.000   3rd Qu.:374640           
#Max.   :150000   Max.   :80.00   Max.   :4.000   Max.   :20.00   Max.   :8.000   Max.   :500000            

# plots
hist(as.numeric(responseCompOOB$Brand))
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

qqnorm(as.numeric(responseCompOOB$Brand)) # Be familiar with this plot, but don't spend a lot of time on it
# See the image file qqnorm_Brand.png

ggplot(data=responseCompOOB, mapping = aes(x = Age, y=Brand)) + geom_jitter(aes(color = Brand))
# See the image file "Age_against_Brand_ggplot.png".

ggplot(data=responseCompOOB, mapping = aes(x = Salary, y=Brand)) + geom_jitter(aes(color = Brand))
# See the image file "Salary_against_Brand_ggplot.png".

ggplot(data=responseCompOOB, mapping = aes(x = CreditLimit, y=Brand)) + geom_jitter(aes(color = Brand))
# See the image file "CreditLimit_against_Brand_ggplot.png".


ggplot(responseCompOOB)  + 
  geom_bar(aes(x=Brand, y=Age),stat="identity", fill="cyan",colour="blue")+
  geom_line(aes(x=Brand, y=Salary), stat="identity",color="red")+
  labs(title= "Age vs Salary for Brand Selection",
       x="Brand",y="Age and Salary Ranges")
# See the image file "Age_vs_Salary_column_line_Brand_Selection_plot.png"


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
#3135 100439.04  50              3       5       7   224869.23     0
#5333  49192.60  72              0      18       4   429056.74     0
#2556  98942.75  20              1       4       7    47516.34     1
#7814 131903.40  39              4      14       3   190780.28     1
#5301  56805.22  67              4      16       6   244856.88     0
#6051  87647.46  25              3      12       4   298536.07     0

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


responseCompOOB1kInt <- responseCompOOB1k

responseCompOOB1kInt[] <- lapply(responseCompOOB1kInt[,1:7], as.integer)
corrAll <- cor(responseCompOOB1kInt)

#corrAll <- cor(responseCompOOB1k[,1:7])

# view the correlation matrix
corrAll
#                     Salary          Age EducationLevel    CarType      Zipcode CreditLimit       Brand
#Salary          1.000000000 -0.006894648    -0.01824629 0.02246659  0.001246374 -0.05285770  0.20638142
#Age            -0.006894648  1.000000000    -0.01636817 0.03767539  0.052630591  0.01536690  0.09607296
#EducationLevel -0.018246290 -0.016368167     1.00000000 0.02179891  0.052784659 -0.04430470 -0.01226768
#CarType         0.022466593  0.037675391     0.02179891 1.00000000  0.023603702  0.03257821  0.05145255
#Zipcode         0.001246374  0.052630591     0.05278466 0.02360370  1.000000000  0.06009696 -0.01275105
#CreditLimit    -0.052857702  0.015366895    -0.04430470 0.03257821  0.060096961  1.00000000 -0.03897447
#Brand           0.206381422  0.096072962    -0.01226768 0.05145255 -0.012751053 -0.03897447  1.00000000

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
#  $ Salary        : num  64071 149568 62627 50517 21888 ...
#$ Age           : int  20 41 68 56 80 29 41 56 36 67 ...
#$ EducationLevel: int  1 4 2 0 1 3 3 2 1 4 ...
#$ CarType       : int  1 4 2 10 20 5 19 18 11 7 ...
#$ Zipcode       : int  0 2 2 6 4 2 0 3 2 1 ...
#$ CreditLimit   : num  10309 271249 338220 51021 121554 ...
#$ Brand         : Factor w/ 2 levels "0","1": 1 2 1 2 2 2 2 2 1 2 ...

# remove highly correlated features, if applicable
responseCompCOR1k <- responseCompOOB1k[-corrIVhigh] 
str(responseCompCOR1k)
#'data.frame':	1000 obs. of  0 variables

############
# caret RFE 
############

# rfFuncs - random forests
# treebagFuncs - bagged trees


## 1. ---- rf ---- ##

# define the control using a random forest selection function (classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, repeats=1)
# run the RFE algorithm
set.seed(7)

rfeRF <- rfe(responseCompOOB1k[,1:6], responseCompOOB1k[,7], sizes=c(1:6), rfeControl=RFcontrol)
rfeRF
#Recursive feature selection
#
#Outer resampling method: Cross-Validated (5 fold) 
#
#Resampling performance over subset size:
#  
#  Variables Accuracy  Kappa AccuracySD KappaSD Selected
#          1    0.640 0.2350    0.05743 0.12350         
#          2    0.909 0.8085    0.02060 0.04186        *
#          3    0.887 0.7585    0.02502 0.05402         
#          4    0.905 0.7987    0.02399 0.05159         
#          5    0.895 0.7781    0.01519 0.03143         
#          6    0.885 0.7559    0.02005 0.04259         
#
#The top 2 variables (out of 2):
#  Salary, Age


# plot the results
plot(rfeRF, type=c("g", "o"))
# See the plot as the image file "rfeRF_Plot.png"

# show predictors used
predictors(rfeRF)
#[1] "Salary" "Age" 

# Note results.  
varImp(rfeRF)
#       Overall
#Age    45.30855

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
#  $ Salary: num  64071 149568 62627 50517 21888 ...
#$ Age   : int  20 41 68 56 80 29 41 56 36 67 ...
#$ Brand : Factor w/ 2 levels "0","1": 1 2 1 2 2 2 2 2 1 2 ...


## 2. ---- treebagFuncs ---- ##

# define the control using a treebagFuncs selection function (classification)
TBcontrol <- rfeControl(functions=treebagFuncs, method="cv", number=5, repeats=1)
# run the RFE algorithm
set.seed(7)

tbeRF <- rfe(responseCompOOB1k[,1:6], responseCompOOB1k[,7], sizes=c(1:6), rfeControl=TBcontrol )
tbeRF
#Recursive feature selection
#
#Outer resampling method: Cross-Validated (5 fold) 
#
#Resampling performance over subset size:
# 
#  Variables Accuracy  Kappa AccuracySD KappaSD Selected
#1   0.6501 0.2557    0.04312 0.10138         
#2   0.8959 0.7797    0.02495 0.05243         
#3   0.8939 0.7749    0.01899 0.04021         
#4   0.8929 0.7731    0.02392 0.05330         
#5   0.8999 0.7880    0.02927 0.06190         
#6   0.9049 0.7992    0.02676 0.05717        *
#  
#  The top 5 variables (out of 6):
#  Salary, Age, CreditLimit, CarType, Zipcode



# plot the results
plot(tbeRF, type=c("g", "o"))
# See the plot as the image file "tbeRF_Plot.png"

# show predictors used
predictors(tbeRF)
#[1] "Salary"         "Age"            "CreditLimit"    "CarType"        "Zipcode"        "EducationLevel" 

# Note results.  
varImp(tbeRF)
#Overall
#Salary         253.58846
#Age            196.03999
#CreditLimit     90.78268
#CarType         75.28792
#Zipcode         58.02317
#EducationLevel  45.40051

##--- create ds with features using varImp from top model ---##

# create ds with predictors from varImp   
responseCompRFE1k <- responseCompOOB1k[,predictors(tbeRF)]
str(responseCompRFE1k)
#'data.frame':	1000 obs. of  6 variables:
#  $ Salary        : num  64071 149568 62627 50517 21888 ...
#$ Age           : int  20 41 68 56 80 29 41 56 36 67 ...
#$ CreditLimit   : num  10309 271249 338220 51021 121554 ...
#$ CarType       : int  1 4 2 10 20 5 19 18 11 7 ...
#$ Zipcode       : int  0 2 2 6 4 2 0 3 2 1 ...
#$ EducationLevel: int  1 4 2 0 1 3 3 2 1 4 ...

# add dv
responseCompRFE1k$Brand <- responseCompOOB1k$Brand
# confirm new ds
str(responseCompRFE1k)
#'data.frame':	1000 obs. of  7 variables:
#  $ Salary      : num  64071 149568 62627 50517 21888 ...
#$ Age           : int  20 41 68 56 80 29 41 56 36 67 ...
#$ CreditLimit   : num  10309 271249 338220 51021 121554 ...
#$ CarType       : int  1 4 2 10 20 5 19 18 11 7 ...
#$ Zipcode       : int  0 2 2 6 4 2 0 3 2 1 ...
#$ EducationLevel: int  1 4 2 0 1 3 3 2 1 4 ...
#$ Brand         : Factor w/ 2 levels "0","1": 1 2 1 2 2 2 2 2 1 2 ...


##############################
# Feature engineering
##############################

# code goes here, when applicable




##################
# Train/test sets
##################

# responseCompOOB
set.seed(123) 
inTraining <- createDataPartition(responseCompOOB$Brand, p=0.75, list=FALSE)
oobTrain <- responseCompOOB[inTraining,]   
oobTest <- responseCompOOB[-inTraining,]   
# verify number of obs 
nrow(oobTrain) 
#[1] 7424
nrow(oobTest)  
#[1] 2474




# responseCompRFE1k
set.seed(123) 
rfe1kinTraining <- createDataPartition(responseCompRFE1k$Brand, p=0.75, list=FALSE)
rfe1kTrain <- responseCompRFE1k[rfe1kinTraining,]   
rfe1kTest <- responseCompRFE1k[-rfe1kinTraining,]   
# verify number of obs 
nrow(rfe1kTrain) 
# 751
nrow(rfe1kTest)  
# 249

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
# model parameter                         label forReg forClass probModel
#1    rf      mtry #Randomly Selected Predictors   TRUE     TRUE      TRUE

## ------- RF ------- ##

# default

set.seed(123)
oobRFfit <- train(Brand~., data=oobTrain, method="rf", importance=T, trControl=fitControl)


mtry <- sqrt(ncol(oobTrain)-1)
mtry

rfgrid <- expand.grid(.mtry=mtry)


set.seed(123)
# fit
oobRFfit <- train(Brand~.,data=oobTrain,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfgrid)


oobRFfit
#Random Forest 
#
#7424 samples
#6 predictor
#2 classes: '0', '1' 
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 5939, 5939, 5939, 5940, 5939 
#Resampling results:
#  
#  Accuracy   Kappa    
#0.9174315  0.8249274
#
#Tuning parameter 'mtry' was held constant at a value of 2.44949

#plot(oobRFfit)   
# See the image as file "oobRFfit_plot.png"

varImp(oobRFfit)
#rf variable importance
#
#Importance
#Salary           100.0000
#Age               69.4235
#CreditLimit        0.8292
#CarType            0.3991
#Zipcode            0.2233
#EducationLevel     0.0000




##  -----Bagged Tree-----------------##


# default

set.seed(123)
oobTBfit <- train(Brand~., data=oobTrain, method="treebag", importance=T, trControl=fitControl)

oobTBfit
#Bagged CART 
#
#7424 samples
#6 predictor
#2 classes: '0', '1' 
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 5938, 5940, 5939, 5940, 5939 
#Resampling results:
#  
#  Accuracy   Kappa    
#  0.9092155  0.8073282



varImp(oobTBfit)
#treebag variable importance
#
#Overall
#Salary         100.000
#Age             79.204
#CreditLimit     13.658
#CarType          8.184
#Zipcode          4.012
#EducationLevel   0.000


##################
# Model selection
##################

#-- responseCompOOB --# 

oobFitComp <- resamples(list(rf=oobRFfit, tb=oobTBfit))
# output summary metrics for tuned models 
summary(oobFitComp)

#Call:
#  summary.resamples(object = oobFitComp)
#
#Models: rf, tb 
#Number of resamples: 5 
#
#Accuracy 
#        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf 0.9111111 0.9144781 0.9144781 0.9174315 0.9185185 0.9285714    0
#tb 0.9010767 0.9037037 0.9090296 0.9092155 0.9151515 0.9171159    0
#
#Kappa 
#        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rf 0.8126212 0.8184028 0.8185311 0.8249274 0.8279392 0.8471426    0
#tb 0.7901836 0.7965147 0.8053916 0.8073282 0.8196447 0.8249067    0


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
# Accuracy     Kappa 
#0.9232013 0.8377535  


# plot predicted verses actual
plot(rfPred1, oobTest$Brand)
# See the plot image as "Plot_Brand_Prediction_vs_actual.png"

# -----predict with TreeBag----
tbPred1 <- predict(oobTBfit, oobTest)
# performace measurment
postResample(tbPred1, oobTest$Brand)
#Accuracy     Kappa 
#0.9143088 0.8177552 

# plot predicted verses actual
plot(tbPred1, oobTest$Brand)
# See the plot image as "TreeBag_predicted_vs_actual.png"

###############################
# Predict new data (Dataset 2)
###############################

# predict for new dataset with no values for DV

rfPredOOB <- predict(oobRFfit, surveyIncompOOB)

head(rfPredOOB)
#[1] 1 0 1 1 1 1
#Levels: 0 1

postResample(rfPredOOB, surveyIncompOOB$Brand)
#  Accuracy      Kappa 
#0.39100000 0.01230336 

# -------------Predict for new dataset with Treebag

# predict for new dataset with no values for DV

tbPredOOB <- predict(oobTBfit, surveyIncompOOB)

head(tbPredOOB)
#[1] 1 0 1 1 1 1
#Levels: 0 1

postResample(tbPredOOB, surveyIncompOOB$Brand)
#  Accuracy      Kappa 
#0.38760000 0.01272037

# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)


