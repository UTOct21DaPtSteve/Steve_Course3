# Title: Predict sales volume

# Last update: 2022.02

# File: Course3Task3_Steve_0305.R
# Project name: Multiple Regression in R

################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("doParallel")  # for Win parallel processing (see below) 
install.packages("gbm")
install.packages("e1071")
install.packages("randomForest")
install.packages("kernlab")
install.packages("plyr")
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)             # for Win
library(gbm)
library(e1071)
library(randomForest)
library(kernlab)
library(plyr)

#####################
# Parallel processing
#####################
#--- for Win ---#

detectCores()          # detect number of cores
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio



##############
# Import data 
##############

##-- Load Train/Existing data (Dataset 1) --##
existingProdAttrOOB <- read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE)
str(existingProdAttrOOB)

##--- Load Predict/New data (Dataset 2) [dv = NA, 0, or Blank] ---##

newProdAttrOOB <- read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE)
str(newProdAttrOOB)

################
# Evaluate data
################

##--- Dataset 1 ---##

# Review the statistics of data set
summary(existingProdAttrOOB)
#ProductType          ProductNum        Price         x5StarReviews   
#Length:80          Min.   :101.0   Min.   :   3.60   Min.   :   0.0  
#Class :character   1st Qu.:120.8   1st Qu.:  52.66   1st Qu.:  10.0  
#Mode  :character   Median :140.5   Median : 132.72   Median :  50.0  
#                   Mean   :142.6   Mean   : 247.25   Mean   : 176.2  
#                   3rd Qu.:160.2   3rd Qu.: 352.49   3rd Qu.: 306.5  
#                   Max.   :200.0   Max.   :2249.99   Max.   :2801.0 
#
#x4StarReviews    x3StarReviews    x2StarReviews    x1StarReviews    
#Min.   :  0.00   Min.   :  0.00   Min.   :  0.00   Min.   :   0.00  
#1st Qu.:  2.75   1st Qu.:  2.00   1st Qu.:  1.00   1st Qu.:   2.00  
#Median : 22.00   Median :  7.00   Median :  3.00   Median :   8.50  
#Mean   : 40.20   Mean   : 14.79   Mean   : 13.79   Mean   :  37.67  
#3rd Qu.: 33.00   3rd Qu.: 11.25   3rd Qu.:  7.00   3rd Qu.:  15.25  
#Max.   :431.00   Max.   :162.00   Max.   :370.00   Max.   :1654.00
#
#PositiveServiceReview NegativeServiceReview Recommendproduct
#Min.   :  0.00        Min.   :  0.000       Min.   :0.100   
#1st Qu.:  2.00        1st Qu.:  1.000       1st Qu.:0.700   
#Median :  5.50        Median :  3.000       Median :0.800   
#Mean   : 51.75        Mean   :  6.225       Mean   :0.745   
#3rd Qu.: 42.00        3rd Qu.:  6.250       3rd Qu.:0.900   
#Max.   :536.00        Max.   :112.000       Max.   :1.000
#
#BestSellersRank ShippingWeight     ProductDepth      ProductWidth   
#Min.   :    1   Min.   : 0.0100   Min.   :  0.000   Min.   : 0.000  
#1st Qu.:    7   1st Qu.: 0.5125   1st Qu.:  4.775   1st Qu.: 1.750  
#Median :   27   Median : 2.1000   Median :  7.950   Median : 6.800  
#Mean   : 1126   Mean   : 9.6681   Mean   : 14.425   Mean   : 7.819  
#3rd Qu.:  281   3rd Qu.:11.2050   3rd Qu.: 15.025   3rd Qu.:11.275  
#Max.   :17502   Max.   :63.0000   Max.   :300.000   Max.   :31.750  
#NA's   :15
#ProductHeight     ProfitMargin        Volume     
#Min.   : 0.000   Min.   :0.0500   Min.   :    0  
#1st Qu.: 0.400   1st Qu.:0.0500   1st Qu.:   40  
#Median : 3.950   Median :0.1200   Median :  200  
#Mean   : 6.259   Mean   :0.1545   Mean   :  705  
#3rd Qu.:10.300   3rd Qu.:0.2000   3rd Qu.: 1226  
#Max.   :25.800   Max.   :0.4000   Max.   :11204

# view first/last obs/rows
head(existingProdAttrOOB)

tail(existingProdAttrOOB)

# check for missing values 
anyNA(existingProdAttrOOB)
#[1] TRUE

# check for duplicates
anyDuplicated((existingProdAttrOOB))
#[1] 0

##--- Dataset 2 ---##

str(newProdAttrOOB)

# view first/last obs/rows
head(newProdAttrOOB)
tail(newProdAttrOOB)
# check for missing values 
anyNA(newProdAttrOOB)
#[1] FALSE
# check for duplicates
anyDuplicated(newProdAttrOOB)
#[1] 0


#############
# Preprocess
#############

##--- Dataset 1 ---##

# find which column(s) has NA value
colSums(is.na(existingProdAttrOOB))
#          ProductType            ProductNum                 Price 
#                    0                     0                     0 
#        x5StarReviews         x4StarReviews         x3StarReviews 
#                    0                     0                     0 
#        x2StarReviews         x1StarReviews PositiveServiceReview 
#                    0                     0                     0 
#NegativeServiceReview      Recommendproduct       BestSellersRank 
#                    0                     0                    15 
#       ShippingWeight          ProductDepth          ProductWidth 
#                    0                     0                     0 
#        ProductHeight          ProfitMargin                Volume 
#                    0                     0                     0 

# So, the "BestSellersRank" column has NA values, 15 rows
# Since 15 rows out of 80 rows is more than 10%, better not to
# remove those 15 rows. Instead, let's replace with the mean value.

existingProdAttrOOB$BestSellersRank[is.na(existingProdAttrOOB$BestSellersRank)]<-mean(existingProdAttrOOB$BestSellersRank,na.rm = TRUE)

# Now, let check again to see if there are still any NA values
anyNA(existingProdAttrOOB)
#[1] FALSE

# Check again which column(s) has NA value
colSums(is.na(existingProdAttrOOB))
#          ProductType            ProductNum                 Price 
#                    0                     0                     0 
#        x5StarReviews         x4StarReviews         x3StarReviews 
#                    0                     0                     0 
#        x2StarReviews         x1StarReviews PositiveServiceReview 
#                    0                     0                     0 
#NegativeServiceReview      Recommendproduct       BestSellersRank 
#                    0                     0                     0 
#       ShippingWeight          ProductDepth          ProductWidth 
#                    0                     0                     0 
#        ProductHeight          ProfitMargin                Volume 
#                    0                     0                     0 


# check for duplicates
anyDuplicated((existingProdAttrOOB))

# Good, let's keep a copy of the cleaned data set 1

existingProdAttrCleaned <- existingProdAttrOOB

#####################
# EDA/Visualizations
#####################

# plots
hist(existingProdAttrCleaned$Volume)
#  See the image file "Volume_Plot.png"

plot(existingProdAttrCleaned$Price, existingProdAttrCleaned$Volume)
#  See the image file "Price_vs_Volume_Plot.png"

plot(existingProdAttrCleaned$x5StarReviews, existingProdAttrCleaned$Volume)
#  See the image file "5star_vs_Volume_Plot.png"

plot(existingProdAttrCleaned$x4StarReviews, existingProdAttrCleaned$Volume)
#  See the image file "4star_vs_Volume_Plot.png"

plot(existingProdAttrCleaned$x3StarReviews, existingProdAttrCleaned$Volume)
#  See the image file "3star_vs_Volume_Plot.png"

plot(existingProdAttrCleaned$x2StarReviews, existingProdAttrCleaned$Volume)
#  See the image file "2star_vs_Volume_Plot.png"

plot(existingProdAttrCleaned$x1StarReviews, existingProdAttrCleaned$Volume)
#  See the image file "1star_vs_Volume_Plot.png"

plot(existingProdAttrCleaned$PositiveServiceReview, existingProdAttrCleaned$Volume)
#  See the image file "PositiveServiceReview_vs_Volume_Plot.png"

qqnorm(existingProdAttrCleaned$Volume)
#  See the image file "qqnorm_Volume.png"

#######################
# Feature selection
#######################

#######################
# Correlation analysis
#######################


# for regression problems, the below rules apply.
# 1) compare each IV to the DV, if cor > 0.95, remove IV
# 2) compare each pair of IVs, if cor > 0.90, remove the
#    IV that has the lowest cor to the DV. (see code
#    below for setting a threshold to automaticall select
#    IVs that are highly correlated)


# How many columns
ncol(existingProdAttrCleaned)
#[1] 18

existingProdAttrCleaned$ProductType <- as.factor(existingProdAttrCleaned$ProductType)
existingProdAttrCleaned$ProductType <- as.numeric(existingProdAttrCleaned$ProductType)

##Save a copy of the cleaned data set to the file

write.csv(existingProdAttrCleaned, file="OOBCleaned.csv", row.names = TRUE)

# calculate correlation matrix for all vars
corrAll <- cor(existingProdAttrCleaned[,1:18])
# view the correlation matrix
corrAll

#                      ProductType   ProductNum       Price x5StarReviews x4StarReviews x3StarReviews x2StarReviews x1StarReviews PositiveServiceReview NegativeServiceReview
#ProductType            1.00000000  0.335662230  0.21826235   -0.18858951 -0.1089635323   -0.02371002   0.087067778   0.151884525           -0.25370275            0.18111286
#ProductNum             0.33566223  1.000000000 -0.03974873    0.16612076  0.1194006067    0.09020064  -0.004533099  -0.063063850           -0.05774806           -0.01942716
#Price                  0.21826235 -0.039748728  1.00000000   -0.14234399 -0.1652836990   -0.15053761  -0.110681189  -0.083957332           -0.14214329           -0.06079037
#x5StarReviews         -0.18858951  0.166120763 -0.14234399    1.00000000  0.8790063940    0.76337319   0.487279328   0.255023904            0.62226022            0.30941899
#x4StarReviews         -0.10896353  0.119400607 -0.16528370    0.87900639  1.0000000000    0.93721418   0.679005621   0.444941717            0.48342128            0.53322218
#x3StarReviews         -0.02371002  0.090200642 -0.15053761    0.76337319  0.9372141751    1.00000000   0.861480050   0.679276158            0.41851739            0.68409662
#x2StarReviews          0.08706778 -0.004533099 -0.11068119    0.48727933  0.6790056214    0.86148005   1.000000000   0.951912978            0.30890137            0.86475481
#x1StarReviews          0.15188453 -0.063063850 -0.08395733    0.25502390  0.4449417168    0.67927616   0.951912978   1.000000000            0.20003529            0.88472832
#PositiveServiceReview -0.25370275 -0.057748062 -0.14214329    0.62226022  0.4834212832    0.41851739   0.308901370   0.200035288            1.00000000            0.26554975
#NegativeServiceReview  0.18111286 -0.019427155 -0.06079037    0.30941899  0.5332221777    0.68409662   0.864754808   0.884728323            0.26554975            1.00000000
#Recommendproduct      -0.13896289  0.003886211  0.06893036    0.16954126  0.0714153315   -0.05661326  -0.197917979  -0.246092974            0.23282881           -0.18832924
#BestSellersRank        0.27307570  0.052126800  0.16283736   -0.09923238 -0.1547135040   -0.11956304  -0.077463447  -0.048048569           -0.16621672           -0.10743229
#ShippingWeight         0.28241713  0.081238782  0.41677740   -0.18802398 -0.1949140938   -0.17184204  -0.128685586  -0.095656192           -0.27073854           -0.11179387
#ProductDepth          -0.13411905  0.036187970  0.01096765    0.06610525 -0.0317207111   -0.04937650  -0.042636007  -0.034639801           -0.05052659           -0.06741045
#ProductWidth           0.14938006  0.126793427  0.38239753   -0.14343661 -0.0006476125   -0.01883893  -0.065799979  -0.101139826           -0.33909373           -0.09720713
#ProductHeight          0.05283750 -0.046220225  0.29416060   -0.16000400 -0.0858559708   -0.06808141  -0.013774805   0.002517859           -0.31429444           -0.02073531
#ProfitMargin           0.23014358  0.039715141  0.09966941   -0.01344860 -0.1466538020   -0.12870692  -0.090093715  -0.031227760            0.42359172            0.04203563
#Volume                -0.18858951  0.166120763 -0.14234399    1.00000000  0.8790063940    0.76337319   0.487279328   0.255023904            0.62226022            0.30941899
#                      Recommendproduct BestSellersRank ShippingWeight ProductDepth  ProductWidth ProductHeight ProfitMargin      Volume
#ProductType               -0.138962889      0.27307570     0.28241713 -0.134119054  0.1493800591   0.052837500   0.23014358 -0.18858951
#ProductNum                 0.003886211      0.05212680     0.08123878  0.036187970  0.1267934273  -0.046220225   0.03971514  0.16612076
#Price                      0.068930357      0.16283736     0.41677740  0.010967649  0.3823975328   0.294160597   0.09966941 -0.14234399
#x5StarReviews              0.169541264     -0.09923238    -0.18802398  0.066105249 -0.1434366092  -0.160004003  -0.01344860  1.00000000
#x4StarReviews              0.071415331     -0.15471350    -0.19491409 -0.031720711 -0.0006476125  -0.085855971  -0.14665380  0.87900639
#x3StarReviews             -0.056613257     -0.11956304    -0.17184204 -0.049376503 -0.0188389256  -0.068081406  -0.12870692  0.76337319
#x2StarReviews             -0.197917979     -0.07746345    -0.12868559 -0.042636007 -0.0657999794  -0.013774805  -0.09009372  0.48727933
#x1StarReviews             -0.246092974     -0.04804857    -0.09565619 -0.034639801 -0.1011398264   0.002517859  -0.03122776  0.25502390
#PositiveServiceReview      0.232828810     -0.16621672    -0.27073854 -0.050526592 -0.3390937285  -0.314294445   0.42359172  0.62226022
#NegativeServiceReview     -0.188329242     -0.10743229    -0.11179387 -0.067410452 -0.0972071272  -0.020735305   0.04203563  0.30941899
#Recommendproduct           1.000000000     -0.23315579    -0.12604389  0.090358266  0.0110910859  -0.043715755   0.09576064  0.16954126
#BestSellersRank           -0.233155786      1.00000000     0.03749053 -0.068519830 -0.0110333499  -0.063849096   0.04567912 -0.09923238
#ShippingWeight            -0.126043887      0.03749053     1.00000000  0.065596924  0.6924735181   0.700311109  -0.07921538 -0.18802398
#ProductDepth               0.090358266     -0.06851983     0.06559692  1.000000000 -0.0060085117   0.025484993  -0.20717603  0.06610525
#ProductWidth               0.011091086     -0.01103335     0.69247352 -0.006008512  1.0000000000   0.566827113  -0.29143640 -0.14343661
#ProductHeight             -0.043715755     -0.06384910     0.70031111  0.025484993  0.5668271129   1.000000000  -0.28810629 -0.16000400
#ProfitMargin               0.095760642      0.04567912    -0.07921538 -0.207176026 -0.2914363968  -0.288106289   1.00000000 -0.01344860
#Volume                     0.169541264     -0.09923238    -0.18802398  0.066105249 -0.1434366092  -0.160004003  -0.01344860  1.00000000

# Generate the heat map of the correlation matrix
# Clean out the plot area first
plot.new()

# plot correlation matrix
corrplot(corrAll, method = "circle")
# See the image file "correlation_matrix_method_circle_plot.png"

corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
# see the image file "correlation_matrix_hclust_plot.png"

###  Remove some features to avoid overfitting.
#  From the correlation matrix, found the following:
#
#  1) 5-star review (IV) has correlation value 1 to Volume (DV), 
#  2) The pair of 4-star review (IV) and 3-star review (IV) has correlation value 0.937 which is greater than 0.9. 
#  3) The 3-star review (IV) has lower correlation value to Volume (DV) than the 4-star review (IV) correlation value to Volume (DV)
#  4) The pair of 2-star review (IV) and 1-star review (IV) has correlation value 0.95 which is greater than 0.9.
#  5) The 1-star review (IV) has lower correlation value to Volume (DV) than the 2-star review (IV) correlation value to Volume (DV)
#
#  Therefore, based on the Regression models Rule 1 and Rule 2, we decide to remove the features of 5-Star review (IV), 3-star Review, and 1-star review.

shortenedExistingProdAttrOOB <- subset(existingProdAttrOOB, select = - c(x5StarReviews, x3StarReviews, x1StarReviews))

str(shortenedExistingProdAttrOOB)


# find IVs that are highly corrected (ideally >0.90)

# Dummify the data to flatten the ProductType values to columns
existingDummified <- dummyVars(" ~ .", data = shortenedExistingProdAttrOOB)

readyData <- data.frame(predict(existingDummified, newdata = shortenedExistingProdAttrOOB))

str(readyData)

# Get the IVs
corrIV <- cor(readyData[,1:25])
# create object with indexes of highly corr features 
corrIVhigh <- findCorrelation(corrIV, cutoff=0.8)  # used 0.8 for illustration purposes  
# print indexes of highly correlated attributes
corrIVhigh
#[1]  3 16

# get var name of high corr IV
colnames(readyData[corrIVhigh])
#[1] "ProductTypeExtendedWarranty" "x2StarReviews"  



############
# caret RFE 
############


## ---- rf ---- ##

# define the control using a random forest selection function (regression or classification)
RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, repeats=1)

# run the RFE algorithm
set.seed(7)
rfeRF <- rfe(readyData[,1:25], readyData[,26], sizes=c(1:25), rfeControl=RFcontrol)
rfeRF 

#Recursive feature selection

#Outer resampling method: Cross-Validated (5 fold) 

#Resampling performance over subset size:
#  
#  Variables  RMSE Rsquared   MAE RMSESD RsquaredSD MAESD Selected
#          1 965.5   0.6459 344.7 1115.5     0.3708 348.2         
#          2 842.3   0.8026 287.3  739.3     0.1057 222.0        *
#          3 889.1   0.7666 322.3  735.0     0.1279 245.8         
#          4 902.8   0.7785 359.2  765.5     0.1213 219.3         
#          5 972.4   0.7292 376.8  817.8     0.1264 225.6         
#          6 918.9   0.7549 323.8  767.4     0.1004 230.1         
#          7 922.0   0.7680 330.5  796.8     0.1232 243.9         
#          8 928.9   0.7475 344.4  818.9     0.1224 244.5         
#          9 915.5   0.7522 327.9  798.0     0.1313 237.6         
#         10 916.9   0.7562 331.6  808.7     0.1271 233.8         
#         11 916.7   0.7515 334.2  816.1     0.1225 239.2         
#         12 900.4   0.7618 326.0  790.4     0.1242 232.6         
#         13 904.7   0.7488 328.3  816.5     0.1372 240.0         
#         14 908.2   0.7536 332.9  804.1     0.1294 236.1         
#         15 911.6   0.7661 326.0  806.7     0.1305 233.6         
#         16 908.4   0.7601 335.7  814.6     0.1337 238.9         
#         17 901.4   0.7581 337.7  808.0     0.1302 230.0         
#         18 909.2   0.7681 332.2  784.4     0.1280 228.4         
#         19 925.5   0.7489 342.3  811.2     0.1409 232.7         
#         20 916.6   0.7614 338.9  828.2     0.1259 234.7         
#         21 879.2   0.7872 326.7  813.1     0.1236 234.8         
#         22 907.1   0.7632 337.6  824.8     0.1189 230.2         
#         23 920.7   0.7417 343.7  789.4     0.1240 229.8         
#         24 907.6   0.7731 332.0  806.6     0.1207 230.8         
#         25 899.7   0.7690 335.8  813.4     0.1244 230.5         
#
#The top 2 variables (out of 2):
#   PositiveServiceReview, x4StarReviews

# plot the results
plot(rfeRF, type=c("g", "o"))
# See the image file "variable_RMSE_plot.png"

# show predictors used
predictors(rfeRF)
#[1] "PositiveServiceReview" "x4StarReviews"

# Note results.  
varImp(rfeRF)
#                       Overall
#PositiveServiceReview 15.84154
#x4StarReviews         10.03930




##############################
# Feature engineering
##############################


##################
# Train/test sets
##################

# Show the head of readyData
head(readyData)

# How many columns
ncol(readyData)
#[1] 26

set.seed(123) 
inTraining <- createDataPartition(readyData$Volume, p=0.9, list=FALSE)
readyDataTrain <- readyData[inTraining,]   
readyDataTest <- readyData[-inTraining,]   
# verify number of obs 
nrow(oobTrain) # 73
nrow(oobTest)  # 7


################
# Train control
################

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1) 



###############
# Train models
###############

#######################Gradient Boosting ##################

set.seed(123)
readyDataGBMfit <- train(Volume~., data=readyDataTrain, method="gbm", trControl=fitControl)

readyDataGBMfit
#Stochastic Gradient Boosting 
#
#73 samples
#25 predictors
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 58, 58, 58, 60, 58 
#Resampling results across tuning parameters:
#  
#  interaction.depth  n.trees  RMSE      Rsquared   MAE     
#1                   50      1030.158  0.6773655  490.7700
#1                  100      1144.684  0.6088361  599.0674
#1                  150      1162.245  0.5948104  620.2107
#2                   50      1039.145  0.6691593  489.3885
#2                  100      1096.035  0.6416665  562.9633
#2                  150      1134.855  0.6175434  596.9430
#3                   50      1046.154  0.6583355  478.0877
#3                  100      1078.609  0.6317835  537.0579
#3                  150      1113.030  0.5852837  570.6035
#
#Tuning parameter 'shrinkage' was held constant at a value of 0.1
#Tuning parameter 'n.minobsinnode' was held constant at a value of 10
#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

plot(readyDataGBMfit)
# See the image file "readyDataGBMfit_plot.png"

# eval variable importance
varImp(readyDataGBMfit)

#gbm variable importance
#
#only 20 most important variables shown (out of 25)
#
#                           Overall
#PositiveServiceReview      100.000
#NegativeServiceReview       35.179
#ProductNum                  26.314
#x4StarReviews               18.018
#x2StarReviews               15.291
#Price                       14.894
#ShippingWeight              11.230
#ProductWidth                 7.912
#BestSellersRank              6.667
#ProfitMargin                 4.957
#ProductTypeAccessories       3.208
#ProductTypeGameConsole       0.000
#ProductDepth                 0.000
#ProductTypePrinter           0.000
#ProductTypeSmartphone        0.000
#ProductTypePrinterSupplies   0.000
#Recommendproduct             0.000
#ProductHeight                0.000
#ProductTypeLaptop            0.000
#ProductTypeTablet            0.000


#############Support Vector Machines#############


set.seed(123)

readyDataSVMfit <- train(Volume ~., data = readyDataTrain, method = "svmLinear",
                               trControl=fitControl,
                               preProcess = c("center", "scale"),
                               tuneLength = 10)

readyDataSVMfit

#Support Vector Machines with Linear Kernel 
#
#73 samples
#25 predictors
#
#Pre-processing: centered (25), scaled (25) 
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 58, 58, 58, 60, 58 
#Resampling results:
#  
#  RMSE     Rsquared   MAE     
#1046.41  0.6941628  537.3439
#
#Tuning parameter 'C' was held constant at a value of 1


# eval variable importance
varImp(readyDataSVMfit)
#loess r-squared variable importance
#
#only 20 most important variables shown (out of 25)
#
#                            Overall
#x4StarReviews              100.0000
#PositiveServiceReview       88.0161
#x2StarReviews               70.9138
#NegativeServiceReview       34.1855
#ProductTypeGameConsole      16.6781
#ProductNum                  15.2686
#ProductWidth                 7.3325
#BestSellersRank              6.7229
#ProductHeight                6.6900
#ProductDepth                 6.2084
#ShippingWeight               6.1434
#Price                        3.9521
#ProfitMargin                 3.1279
#Recommendproduct             2.5675
#ProductTypePrinter           1.7711
#ProductTypeAccessories       1.5231
#ProductTypePC                1.2359
#ProductTypePrinterSupplies   0.9520
#ProductTypeLaptop            0.5744
#ProductTypeNetbook           0.5712


##############Random Forest #####################

set.seed(123)
readyDataRFfit <- train(Volume~., data=readyDataTrain, method="rf", importance=T, trControl=fitControl)

readyDataRFfit
#Random Forest 
#
#73 samples
#25 predictors
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 58, 58, 58, 60, 58 
#Resampling results across tuning parameters:
#  
#  mtry  RMSE      Rsquared   MAE     
#   2    956.7427  0.6792273  435.4334
#  13    952.7982  0.7179215  338.4548
#  25    947.2195  0.7393261  324.9800
#
#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 25.

# eval variable importance
varImp(readyDataRFfit)

#rf variable importance

#only 20 most important variables shown (out of 25)
#
#                           Overall
#PositiveServiceReview       100.000
#x4StarReviews                40.243
#x2StarReviews                20.933
#ProductTypeGameConsole       14.716
#NegativeServiceReview        13.019
#BestSellersRank              12.738
#ProductWidth                 11.559
#ProductHeight                10.605
#ProductDepth                 10.521
#ProductTypeExtendedWarranty   9.447
#ProductNum                    9.335
#ProductTypePC                 8.444
#ProductTypePrinter            7.824
#ProductTypePrinterSupplies    7.797
#ProductTypeLaptop             7.341
#ProfitMargin                  4.843
#Recommendproduct              4.677
#ProductTypeDisplay            4.310
#Price                         3.645
#ProductTypeAccessories        3.243

##################
# Model selection
##################

#  Listing out the models

readyDataFitListing <- resamples(list(gbm=readyDataGBMfit, svm=readyDataSVMfit, rf=readyDataRFfit))
# output summary metrics for tuned models 
summary(readyDataFitListing)

#Call:
#summary.resamples(object = readyDataFitListing)
#
#Models: gbm, svm, rf 
#Number of resamples: 5 
#
#MAE 
#Min.  1st Qu.   Median     Mean  3rd Qu.      Max. NA's
#gbm 233.63358 318.6297 460.8312 490.7700 666.8178  773.9378    0
#svm 324.50623 422.5252 423.5069 537.3439 747.4259  768.7554    0
#rf   53.09482 100.3770 135.6897 348.5825 384.8519 1068.8990    0
#
#RMSE 
#        Min.  1st Qu.   Median      Mean  3rd Qu.     Max. NA's
#gbm 273.7374 385.3949 589.5126 1030.1582 1426.042 2476.104    0
#svm 468.3767 660.6643 857.5997 1046.4101 1575.025 1670.384    0
#rf  131.0587 172.4706 253.7430  892.1462 1090.742 2812.716    0
#
#Rsquared 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#gbm 0.3718698 0.4547874 0.7710305 0.6773655 0.8499175 0.9392225    0
#svm 0.4460014 0.5198960 0.7321580 0.6941628 0.8777030 0.8950555    0
#rf  0.5747394 0.7326749 0.9180823 0.8150125 0.9237076 0.9258580    0



############################
# Predict testSet/validation
############################

# ##########predict with SVM  ######################
svmPred1 <- predict(readyDataSVMfit, readyDataTest)
# performace measurment
postResample(svmPred1, readyDataTest$Volume)
#       RMSE    Rsquared         MAE 
#413.1368199   0.9688884 276.0563568 

# plot predicted verses actual
plot(svmPred1, readyDataTest$Volume, xlab= "Predicted Values", ylab="Observed Values")
abline(a = 0, b = 1, lwd=2, col = "green")
# See the image file "svm_prediction_plot.png"


# ##########predict with RF   ###########################
rfPred1 <- predict(readyDataRFfit, readyDataTest)
# performace measurment
postResample(rfPred1, readyDataTest$Volume)
#       RMSE    Rsquared         MAE 
#334.6574967   0.7569658 191.7667429 


# plot predicted verses actual
plot(rfPred1, readyDataTest$Volume, xlab= "Predicted Values", ylab="Observed Values")
abline(a = 0, b = 1, lwd=2, col = "green")
# See the image file "RF_prediction_plot.png"


# ##########predict with GBM   ###########################
gbmPred1 <- predict(readyDataGBMfit, readyDataTest)

plot(gbmPred1, oobTest$Volume, xlab= "Predicted Values", ylab="Observed Values")
abline(a = 0, b = 1, lwd=2, col = "green")
# See the image file "GBM_prediction_plot.png"


# performace measurement
postResample(gbmPred1, readyDataTest$Volume)

#      RMSE   Rsquared        MAE 
#397.655568   0.870826 368.466654   




###===========Do the modeling and prediction focused on the 4 product types==============
set.seed(123)
svmfit4ProdTypes <- train(Volume ~. + ProductTypePC + ProductTypeLaptop + ProductTypeNetbook + ProductTypeSmartphone, data = readyDataTrain, method = "svmLinear",
                   trControl=fitControl,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)


svmPred4ProdTypes <- predict(svmfit4ProdTypes, readyDataTest)

postResample(svmPred4ProdTypes, readyData$Volume)
#   RMSE  Rsquared        MAE 
#1812.081        NA  1036.054

#----------------------------------

set.seed(123)
rffit4ProdTypes <- train(Volume ~. + ProductTypePC + ProductTypeLaptop + ProductTypeNetbook + ProductTypeSmartphone, data=readyDataTrain, method="rf", importance=T, trControl=fitControl)

rfPred4ProdTypes <- predict(rffit4ProdTypes, readyDataTest)

postResample(rfPred4ProdTypes, readyData$Volume)
#     RMSE Rsquared       MAE 
#1624.4298       NA  814.2963  

#-------------------------------------------

set.seed(123)

gbmfit4ProdTypes <- train(Volume ~. + ProductTypePC + ProductTypeLaptop + ProductTypeNetbook + ProductTypeSmartphone, data=readyDataTrain, method="gbm", trControl=fitControl)

gbmPred4ProdTypes <- predict(gbmfit4ProdTypes, readyDataTest)

postResample(gbmPred4ProdTypes, oobTest$Volume)
#       RMSE    Rsquared         MAE 
#402.1258285   0.8662395 310.5081175


###### Now, a quick review to select the proper model for product type sales prediction on new data set

#     Support Vector Machines
#   RMSE  Rsquared        MAE 
#1812.081        NA  1036.054

#     Random Forest
#     RMSE Rsquared       MAE 
#1624.4298       NA  814.2963 

#     Gradient Boosting
#       RMSE    Rsquared         MAE 
#402.1258285   0.8662395 310.5081175

# Gradient Boosting algorithm based model fits the best for proudct type sales prediction
# Therefore, Will use the Gradient Boosting algrithm based model for new data set's product type sales prediction


###############################
# Predict new data (Dataset 2)
###############################


# Use the chosen model to do the prediction for new dataset with no values for DV

# Dummify the data set 2
dataSet2Dummified <- dummyVars(" ~ .", data = newProdAttrOOB)

readyData2 <- data.frame(predict(dataSet2Dummified, newdata = newProdAttrOOB))

str(readyData2)

# Remove the same three features from the dataset as we did for the dataset 1
shortenedreadyData2 <- subset(readyData2, select = - c(x5StarReviews, x3StarReviews, x1StarReviews))

str(shortenedreadyData2)


#-------------------New data set's four product types sales prediction------
gbmPred4ProdTypes2 <- predict(gbmfit4ProdTypes, shortenedreadyData2)

# Capture and save the prediction values to CSV file
gbmoutputDataSetProdType2 <- shortenedreadyData2

gbmoutputDataSetProdType2$predictions <- gbmPred4ProdTypes2

write.csv(gbmoutputDataSetProdType2, file="C2.T3gbmProdTypePredication.csv", row.names = TRUE)

#--------------------Secure the reviews predication data----------


rfPred2 <- predict(readyDataRFfit, shortenedreadyData2)

# Capture and save the prediction values to CSV file
rfshortenedreadyData2 <- shortenedreadyData2

rfshortenedreadyData2$predictions <- rfPred2

write.csv(rfshortenedreadyData2, file="C2.T3rfNewDataSetPredication.csv", row.names = TRUE)



# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

