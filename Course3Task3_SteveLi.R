# Title: Predict sales volume

# Last update: 2022.02

# File: Course3Task3_Steve.R
# Project name: Multiple Regression in R

################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("doParallel")  # for Win parallel processing (see below) 
install.packages("Metrics")
install.packages("gbm")  # for fitting Gradient boosting model
install.packages("randomForest") # for fitting Random Forest model
library(caret) 
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)             # for Win
library(ggplot2)
library(scales)
library(e1071)
library(Metrics)
library(gbm)
library(randomForest)


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

# Find which column(s) has NA value
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

# Now, have a subset data frame just for PC, Laptop, Netbook, and Smartphone products

studyProductSubDF <- subset(existingProdAttrCleaned, ProductType=='PC' | ProductType=='Laptop' | ProductType=='Netbook' | ProductType=='Smartphone') 


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

# Stattered plots for the focused columns

ggplot(existingProdAttrCleaned,aes(x = existingProdAttrCleaned$x5StarReviews, y = existingProdAttrCleaned$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_5Star_Volume.png"

ggplot(existingProdAttrCleaned,aes(x = existingProdAttrCleaned$x4StarReviews, y = existingProdAttrCleaned$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_4Star_Volume.png"

ggplot(existingProdAttrCleaned,aes(x = existingProdAttrCleaned$x3StarReviews, y = existingProdAttrCleaned$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_3Star_Volume.png"

ggplot(existingProdAttrCleaned,aes(x = existingProdAttrCleaned$x2StarReviews, y = existingProdAttrCleaned$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_2Star_Volume.png"

ggplot(existingProdAttrCleaned,aes(x = existingProdAttrCleaned$x1StarReviews, y = existingProdAttrCleaned$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_1Star_Volume.png"

ggplot(existingProdAttrCleaned,aes(x = existingProdAttrCleaned$PositiveServiceReview, y = existingProdAttrCleaned$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_PositiveServiceReview_Volume.png"





#######################
# Correlation analysis
#######################


existingDummified <- dummyVars(" ~ .", data = existingProdAttrCleaned)

readyData <- data.frame(predict(existingDummified, newdata = existingProdAttrCleaned))

# Show the head of readyData
head(readyData)
#  ProductTypeAccessories ProductTypeDisplay ProductTypeExtendedWarranty ProductTypeGameConsole ProductTypeLaptop
#1                      0                  0                           0                      0                 0
#2                      0                  0                           0                      0                 0
#3                      0                  0                           0                      0                 0
#4                      0                  0                           0                      0                 1
#5                      0                  0                           0                      0                 1
#6                      1                  0                           0                      0                 0
#  ProductTypeNetbook ProductTypePC ProductTypePrinter ProductTypePrinterSupplies ProductTypeSmartphone
#1                  0             1                  0                          0                     0
#2                  0             1                  0                          0                     0
#3                  0             1                  0                          0                     0
#4                  0             0                  0                          0                     0
#5                  0             0                  0                          0                     0
#6                  0             0                  0                          0                     0
#  ProductTypeSoftware ProductTypeTablet ProductNum   Price x5StarReviews x4StarReviews x3StarReviews x2StarReviews
#1                   0                 0        101  949.00             3             3             2             0
#2                   0                 0        102 2249.99             2             1             0             0
#3                   0                 0        103  399.00             3             0             0             0
#4                   0                 0        104  409.99            49            19             8             3
#5                   0                 0        105 1079.99            58            31            11             7
#6                   0                 0        106  114.22            83            30            10             9
#  x1StarReviews PositiveServiceReview NegativeServiceReview Recommendproduct BestSellersRank ShippingWeight
#1             0                     2                     0              0.9            1967           25.8
#2             0                     1                     0              0.9            4806           50.0
#3             0                     1                     0              0.9           12076           17.4
#4             9                     7                     8              0.8             109            5.7
#5            36                     7                    20              0.7             268            7.0
#6            40                    12                     5              0.3              64            1.6
#  ProductDepth ProductWidth ProductHeight ProfitMargin Volume
#1        23.94         6.62         16.89         0.15     12
#2        35.00        31.75         19.00         0.25      8
#3        10.50         8.30         10.20         0.08     12
#4        15.00         9.90          1.30         0.08    196
#5        12.90         0.30          8.90         0.09    232
#6         5.80         4.00          1.00         0.05    332

# How many columns
ncol(readyData)
#[1] 29

ggplot(readyData,aes(x = readyData$ProductTypeLaptop, y = readyData$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_Laptop_Volume.png"

ggplot(readyData,aes(x = readyData$ProductTypeNetbook, y = readyData$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_Netbook_Volume.png"

ggplot(readyData,aes(x = readyData$ProductTypePC, y = readyData$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_PC_Volume.png"

ggplot(readyData,aes(x = readyData$ProductTypeSmartphone, y = readyData$Volume)) + geom_point() +geom_smooth(method = "lm")
# See the image file "scattered_plot_Smartphone_Volume.png"

################To prevent overfitting, develop some subset data frames####

# Have a subset data frame with customer reviews variables, service review, and sales volume
reviewsSubDF <- readyData[, c(3, 15, 16, 17, 18, 19, 20, 29)]

# Have a subset data frame with four product types and sales volume
productsSubDF <- readyData[, c(5, 6, 7, 10, 29)]

# Have a subset data frame with four product types, customer reviews, and sales volume
productsReviewsSubDF <- readyData[, c(3, 5, 6, 7, 10, 15, 16, 17, 18, 19, 20, 29)]




#####################Full data set correlation analysis####################


# calculate correlation matrix for all vars
corrAll <- cor(readyData[,1:29])
# view the correlation matrix
corrAll

# Generate the heat map of the correlation matrix
# Clean out the plot area first
plot.new()
dev.off()


corrplot(corrAll)
#  See the image file "heat_map_of_correlation.png"


corrIV <- cor(readyData[,1:28])
# create object with indexes of highly corr features 
corrIVhigh <- findCorrelation(corrIV, cutoff=0.8)  # used 0.8 for illustration purposes  
# print indexes of highly correlated attributes
corrIVhigh
#[1] 17 16 18 19  3

# get var name of high corr IV
colnames(readyData[corrIVhigh]) 
#[1] "x3StarReviews"               "x4StarReviews"              
#[3] "x2StarReviews"               "x1StarReviews"              
#[5] "ProductTypeExtendedWarranty"

str(readyData)
#'data.frame':	80 obs. of  29 variables:
#$ ProductTypeAccessories     : num  0 0 0 0 0 1 1 1 1 1 ...
#$ ProductTypeDisplay         : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypeExtendedWarranty: num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypeGameConsole     : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypeLaptop          : num  0 0 0 1 1 0 0 0 0 0 ...
#$ ProductTypeNetbook         : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypePC              : num  1 1 1 0 0 0 0 0 0 0 ...
#$ ProductTypePrinter         : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypePrinterSupplies : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypeSmartphone      : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypeSoftware        : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductTypeTablet          : num  0 0 0 0 0 0 0 0 0 0 ...
#$ ProductNum                 : num  101 102 103 104 105 106 107 108 109 110 ...
#$ Price                      : num  949 2250 399 410 1080 ...
#$ x5StarReviews              : num  3 2 3 49 58 83 11 33 16 10 ...
#$ x4StarReviews              : num  3 1 0 19 31 30 3 19 9 1 ...
#$ x3StarReviews              : num  2 0 0 8 11 10 0 12 2 1 ...
#$ x2StarReviews              : num  0 0 0 3 7 9 0 5 0 0 ...
#$ x1StarReviews              : num  0 0 0 9 36 40 1 9 2 0 ...
#$ PositiveServiceReview      : num  2 1 1 7 7 12 3 5 2 2 ...
#$ NegativeServiceReview      : num  0 0 0 8 20 5 0 3 1 0 ...
#$ Recommendproduct           : num  0.9 0.9 0.9 0.8 0.7 0.3 0.9 0.7 0.8 0.9 ...
#$ BestSellersRank            : num  1967 4806 12076 109 268 ...
#$ ShippingWeight             : num  25.8 50 17.4 5.7 7 1.6 7.3 12 1.8 0.75 ...
#$ ProductDepth               : num  23.9 35 10.5 15 12.9 ...
#$ ProductWidth               : num  6.62 31.75 8.3 9.9 0.3 ...
#$ ProductHeight              : num  16.9 19 10.2 1.3 8.9 ...
#$ ProfitMargin               : num  0.15 0.25 0.08 0.08 0.09 0.05 0.05 0.05 0.05 0.05 ...
#$ Volume                     : num  12 8 12 196 232 332 44 132 64 40 ...

#############################################
#   Focused, subset correlation analysis    #
#############################################

################Customer reviews/service review vs Volume subset correlation analysis####
reviewsCorr <- cor(reviewsSubDF[,1:8])

reviewsCorr
#                            ProductTypeExtendedWarranty x5StarReviews x4StarReviews x3StarReviews
#ProductTypeExtendedWarranty                  1.00000000    0.07086528   -0.09946665   -0.09934446
#x5StarReviews                                0.07086528    1.00000000    0.87900639    0.76337319
#x4StarReviews                               -0.09946665    0.87900639    1.00000000    0.93721418
#x3StarReviews                               -0.09934446    0.76337319    0.93721418    1.00000000
#x2StarReviews                               -0.09348376    0.48727933    0.67900562    0.86148005
#x1StarReviews                               -0.05189306    0.25502390    0.44494172    0.67927616
#PositiveServiceReview                        0.62710951    0.62226022    0.48342128    0.41851739
#Volume                                       0.07086528    1.00000000    0.87900639    0.76337319
#                            x2StarReviews x1StarReviews PositiveServiceReview     Volume
#ProductTypeExtendedWarranty   -0.09348376   -0.05189306             0.6271095 0.07086528
#x5StarReviews                  0.48727933    0.25502390             0.6222602 1.00000000
#x4StarReviews                  0.67900562    0.44494172             0.4834213 0.87900639
#x3StarReviews                  0.86148005    0.67927616             0.4185174 0.76337319
#x2StarReviews                  1.00000000    0.95191298             0.3089014 0.48727933
#x1StarReviews                  0.95191298    1.00000000             0.2000353 0.25502390
#PositiveServiceReview          0.30890137    0.20003529             1.0000000 0.62226022
#Volume                         0.48727933    0.25502390             0.6222602 1.00000000

#  Heat map of reviews vs sales volume 

corrplot(cor(reviewsSubDF[,1:8]))
#  See the image file "heat_map_of_reviews_volume_correlation.png"

################The four product type vs Volume subset correlation analysis####
productTypeCorr <- cor(productsSubDF[,1:5])

productTypeCorr

#                      ProductTypeLaptop ProductTypeNetbook ProductTypePC ProductTypeSmartphone
#ProductTypeLaptop            1.00000000        -0.03160698   -0.04528334           -0.04528334
#ProductTypeNetbook          -0.03160698         1.00000000   -0.03673592           -0.03673592
#ProductTypePC               -0.04528334        -0.03673592    1.00000000           -0.05263158
#ProductTypeSmartphone       -0.04528334        -0.03673592   -0.05263158            1.00000000
#Volume                      -0.06979958        -0.07001054   -0.10289168           -0.03850828
#                           Volume
#ProductTypeLaptop     -0.06979958
#ProductTypeNetbook    -0.07001054
#ProductTypePC         -0.10289168
#ProductTypeSmartphone -0.03850828
#Volume                 1.00000000

#  Heat map of the four product types vs sales volume 

corrplot(cor(productsSubDF[,1:5]))
#  See the image file "heat_map_of_productType_volume_correlation.png"

###########################################
#  Develop Multiple Regression Models     #
###########################################


# Split data into train and test
set.seed(123)
index <- createDataPartition(productsReviewsSubDF$Volume, p=0.70, list=FALSE)
train <- productsReviewsSubDF[index, ]
test <- productsReviewsSubDF[-index, ]


#########################################################################################
#   Generic linear regression models for Laptop, Netbook, PC, and Smartphone vs Volume  #
#########################################################################################

##############Laptop###################

model_Laptop <- lm(Volume ~ ProductTypeLaptop, data = train)

summary(model_Laptop)
#Call:
#  lm(formula = Volume ~ ProductTypeLaptop, data = train)
#
#Residuals:
#  Min      1Q  Median      3Q     Max 
#-760.7  -720.7  -556.7   463.3 10443.3 
#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)   
# (Intercept)          760.7      236.3   3.220  0.00216 **
#  ProductTypeLaptop   -546.7     1261.2  -0.433  0.66640   
#---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#
#Residual standard error: 1752 on 55 degrees of freedom
#Multiple R-squared:  0.003404,	Adjusted R-squared:  -0.01472 
#F-statistic: 0.1879 on 1 and 55 DF,  p-value: 0.6664

laptopPred <- predict(model_Laptop, newdata = test)

postResample(laptopPred, test$Volume)
#        RMSE     Rsquared          MAE 
#725.97207146   0.02402438 639.04031621

########Netbook#####################

model_Netbook <- lm(Volume ~ ProductTypeNetbook, data = train)

summary(model_Netbook)
#Call:
#  lm(formula = Volume ~ ProductTypeNetbook, data = train)
#
#Residuals:
#  Min      1Q  Median      3Q     Max 
#-766.8  -722.8  -534.8   457.2 10437.2 
#
#Coefficients:
#                  Estimate Std. Error t value Pr(>|t|)   
#(Intercept)           766.8      236.0   3.250  0.00197 **
#ProductTypeNetbook   -720.8     1259.6  -0.572  0.56952   
#---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#
#Residual standard error: 1750 on 55 degrees of freedom
#Multiple R-squared:  0.005918,	Adjusted R-squared:  -0.01216 
#F-statistic: 0.3274 on 1 and 55 DF,  p-value: 0.5695

netbookPred <- predict(model_Netbook, newdata = test)

postResample(netbookPred, test$Volume)
#    RMSE Rsquared      MAE 
#740.1617       NA 664.6672

################PC#######################


model_PC <- lm(Volume ~ ProductTypePC, data = train)

summary(model_PC)
#Call:
#  lm(formula = Volume ~ ProductTypePC, data = train)
#
#Residuals:
#  Min      1Q  Median      3Q     Max 
#-768.1  -708.1  -536.1   455.9 10435.9 
#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)   
#(Intercept)      768.1      235.9   3.256  0.00194 **
#  ProductTypePC   -758.1     1259.2  -0.602  0.54964   
#---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#
#Residual standard error: 1749 on 55 degrees of freedom
#Multiple R-squared:  0.006546,	Adjusted R-squared:  -0.01152 
#F-statistic: 0.3624 on 1 and 55 DF,  p-value: 0.5496

pcPred <- predict(model_PC, newdata = test)

postResample(pcPred, test$Volume)
#        RMSE     Rsquared          MAE 
#709.42057864   0.05827415 605.75494071

###########Smartphone####################

model_Smartphone <- lm(Volume ~ ProductTypeSmartphone, data = train)

summary(model_Smartphone)
#Call:
#  lm(formula = Volume ~ ProductTypeSmartphone, data = train)
#
#Residuals:
#  Min      1Q  Median      3Q     Max 
#-762.6  -722.6  -558.6   461.4 10441.4 
#
#Coefficients:
#                      Estimate Std. Error t value Pr(>|t|)   
#(Intercept)              762.6      236.2   3.229   0.0021 **
#ProductTypeSmartphone   -602.6     1260.8  -0.478   0.6346   
#---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#
#Residual standard error: 1751 on 55 degrees of freedom
#Multiple R-squared:  0.004137,	Adjusted R-squared:  -0.01397 
#F-statistic: 0.2285 on 1 and 55 DF,  p-value: 0.6346

smartphonePred <- predict(model_Smartphone, newdata = test)

postResample(smartphonePred, test$Volume)
#        RMSE     Rsquared          MAE 
#7.590928e+02 3.038923e-03 6.634055e+02 

#############5-Star Customer review###################
model_5Star <- lm(Volume ~ x5StarReviews, data = train)

summary(model_5Star)

fiveStarPred <- predict(model_5Star, newdata = test)

postResample(fiveStarPred, test$Volume)
#RMSE Rsquared      MAE 
#0        1        0

#############4-Star Customer review###################
model_4Star <- lm(Volume ~ x4StarReviews, data = train)

summary(model_4Star)

fourStarPred <- predict(model_4Star, newdata = test)

postResample(fourStarPred, test$Volume)
#RMSE     Rsquared          MAE 
#1346.9228104    0.6419054  806.1363444

#############3-Star Customer review###################
model_3Star <- lm(Volume ~ x3StarReviews, data = train)

summary(model_3Star)

threeStarPred <- predict(model_3Star, newdata = test)

postResample(threeStarPred, test$Volume)
#        RMSE     Rsquared          MAE 
#2477.7053427    0.5602049 1128.6911633 

#############2-Star Customer review###################
model_2Star <- lm(Volume ~ x2StarReviews, data = train)

summary(model_2Star)

twoStarPred <- predict(model_2Star, newdata = test)

postResample(twoStarPred, test$Volume)
#        RMSE     Rsquared          MAE 
#5637.4774846    0.3470071 1917.3718323

#############1-Star Customer review###################
model_1Star <- lm(Volume ~ x1StarReviews, data = train)

summary(model_1Star)

oneStarPred <- predict(model_1Star, newdata = test)

postResample(oneStarPred, test$Volume)
#        RMSE     Rsquared          MAE 
#1.436907e+04 2.450753e-01 3.714675e+03 

#############service review###################
model_service <- lm(Volume ~ PositiveServiceReview, data = train)

summary(model_service)

servicePred <- predict(model_service, newdata = test)

postResample(servicePred, test$Volume)
#       RMSE    Rsquared         MAE 
#786.9403399   0.2776466 488.0953580 

################################################################
# From applying the generic linear model on each individual    #
# features, we see the four product types do not have          #
# significant correlation and prediction accuracy to Volume.   #
# But, the 5-star, 4-star, and 3-star customer reviews have    #
# good correlation and prediction accuracy to Volume.          #
#                                                              #
# Time to do more prediction analysis with specific            #
# algorithms.                                                  #
################################################################



#########################################
#Fitting the model using SVM algorithm  #
#########################################

svmModel = svm(Volume~., data=train)

print(svmModel)
#Call:
#  svm(formula = Volume ~ ., data = train)
#
#
#Parameters:
#  SVM-Type:  eps-regression 
#SVM-Kernel:  radial 
#cost:  1 
#gamma:  0.09090909 
#epsilon:  0.1 
#
#
#Number of Support Vectors:  18


#  Do the predication with svmModel, using the test data#####


svmPred = predict(svmModel, test)

x = 1:length(test$Volume)
plot(x, test$Volume, pch=18, col="red")
lines(x, svmPred, lwd="1", col="blue")
# See the image file "svmModel_Predict_plot.png"

#### Accuracy checking for svmPred

postResample(svmPred, test$Volume)
#       RMSE    Rsquared         MAE 
#450.0488850   0.6276474 325.5436947

#########Use 4-star for IV svm model#######
svmModel_4star = svm(Volume~x4StarReviews, data=train)

svmPred_4star = predict(svmModel_4star, test)

postResample(svmPred_4star, test$Volume)
#      RMSE   Rsquared        MAE 
#635.350067   0.630964 454.411789 


######################################################
#Fitting the model using Gradient boosting algorithm  #
######################################################

#   Create data sets for gdm Model training and test use
gdm_test_x = test[, -12]
gdm_test_y = test[, 12]


# Fit the Gradient Boosting Model 
gbmModel = gbm(train$Volume ~., data=train, distribution="gaussian", cv.folds=10, shrinkage=.01, n.minobsinnode = 10, n.trees = 500)

print(gbmModel)
#gbm(formula = train$Volume ~ ., distribution = "gaussian", 
#    data = train, n.trees = 500, n.minobsinnode = 10, shrinkage = 0.01, 
#    cv.folds = 10)
#A gradient boosted model with gaussian loss function.
#500 iterations were performed.
#The best cross-validation iteration was 393.
#There were 11 predictors of which 6 had non-zero influence.

# Clear out the image screen 
plot.new()


summary(gbmModel)
#                                                    var   rel.inf
#x5StarReviews                             x5StarReviews 51.736588
#PositiveServiceReview             PositiveServiceReview 13.501981
#x1StarReviews                             x1StarReviews 12.167468
#x4StarReviews                             x4StarReviews  9.043362
#x2StarReviews                             x2StarReviews  6.858081
#x3StarReviews                             x3StarReviews  6.692520
#ProductTypeExtendedWarranty ProductTypeExtendedWarranty  0.000000
#ProductTypeLaptop                     ProductTypeLaptop  0.000000
#ProductTypeNetbook                   ProductTypeNetbook  0.000000
#ProductTypePC                             ProductTypePC  0.000000
#ProductTypeSmartphone             ProductTypeSmartphone  0.000000

##See the feature importance plot in image file "gdm_variable_importance_plot.png"

#  Let the gbmModel does the predication on the test data

gbmPred_test = predict.gbm(gbmModel, gdm_test_x)

gbmPred_test
#[1]  -33.27812   11.14265  -38.70881 1977.43888 2338.80749 2338.80749  -33.27812  -33.27812 1713.17159
#[10]  277.66409  122.48652 2091.82083 2338.80749  -52.68632 2111.52485  216.23323  -33.27812  545.73681
#[19]  -33.27812 2085.00952 1438.20887  -38.70881 2077.19330


#  Check the accuracy
# Manually calculate approch
residuals = gdm_test_y - gbmPred_test

RMSE_gdm = sqrt(mean((residuals)^2))
MAE_gdm = mean(abs(residuals))
rSquare = 1 - (sum((residuals)^2)/sum((gdm_test_y - mean(gdm_test_y))^2))

cat(" RMSE:", RMSE_gdm, "\n", "R-squared:", rSquare, "\n", "MAE:", MAE_gdm)
#RMSE: 577.2059 
#R-squared: 0.3650196 
#MAE: 389.1216


# Use the function to calculate
postResample(gbmPred_test, test$Volume)
#      RMSE   Rsquared        MAE 
#577.205922   0.801254 389.121582 


######################################################
#Fitting the model using Random Forest algorithm     #
######################################################

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1)

set.seed(123)
rfModelFit <- train(Volume~., data=train, method="rf", importance=T, trControl=fitControl)

rfModelFit
#Random Forest 
#
#57 samples
#11 predictors
#
#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 1 times) 
#Summary of sample sizes: 45, 45, 47, 46, 45 
#Resampling results across tuning parameters:
#  
#  mtry  RMSE      Rsquared   MAE     
#   2    627.1919  0.9608995  247.2738
#   6    562.5904  0.9730134  194.5295
#  11    506.6132  0.9737596  170.3834
#
#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 11.

plot(rfModelFit)
#  See the image file "rfMpdelFit_plot.png"

#  Let the random forest model does the predication on the test data
rfPred <- predict(rfModelFit, test)

# performace measurment
postResample(rfPred, test$Volume)
#        RMSE     Rsquared          MAE 
#1170.4643811    0.7605797  472.8964000 


#########Use 4-star IV for Random Forest model#######

set.seed(123)
rfModel_4star <- train(Volume~x4StarReviews, data=train, method="rf", importance=T, trControl=fitControl)

rfPred_4star = predict(rfModel_4star, test)

postResample(rfPred_4star, test$Volume)
#        RMSE     Rsquared          MAE 
#1183.9257799    0.6373683  521.1665456

#########################################################
#   Now, compare the 3 models to find out the best one  #
#########################################################

#  Random Forest
#        RMSE     Rsquared          MAE 
#1170.4643811    0.7605797  472.8964000 

# SVM
#       RMSE    Rsquared         MAE 
#450.0488850   0.6276474 325.5436947 

# Gradient boosting
#      RMSE   Rsquared        MAE 
#577.205922   0.801254 389.121582

# Based on the above matrix data, the Gradient Boosting algorithm has the highest
# Rsquared value with high coefficient of determination plus the lower error
# rate RMSE value, we choose the Gradient Boosting algorithm based model for the
# prediction on the new data set.

#######Apply the Gradient Boosting model on the new data set to do the prediction###


#   Need to dummify the data set 2 first

existingDummifiedNew <- dummyVars(" ~ .", data = newProdAttrOOB)

readyDataNew <- data.frame(predict(existingDummifiedNew, newdata = newProdAttrOOB))

# Have a subset data frame with customer reviews variables, service review, and sales volume
reviewsSubDFNew <- readyDataNew[, c(3, 15, 16, 17, 18, 19, 20, 29)]

# Have a subset data frame with four product types and sales volume
productsSubDFNew <- readyDataNew[, c(5, 6, 7, 10, 29)]

# Have a subset data frame with four product types, customer reviews, and sales volume
productsReviewsSubDFNew <- readyDataNew[, c(3, 5, 6, 7, 10, 15, 16, 17, 18, 19, 20, 29)]


#  Then do the predication for productsReviewsSubDFNew

gbmPredNew = predict(gbmModel, newdata = productsReviewsSubDFNew)

# Check the accuracy 

#### Accuracy checking for applying the gbmModel on the productsReviewsSubDFNew
mseNew = mean((gbmPredNew)^2)
maeNew = mean(abs(gbmPredNew))
rmseNew = sqrt(mseNew)
r2New = R2(productsReviewsSubDFNew$Volume, gbmPredNew, form = "traditional")

cat( " MAE:", maeNew, "\n", "MSE:", mseNew, "\n", "RMSE:", rmseNew, "\n", "R-squared:", r2New)
#MAE: 775.5961 
#MSE: 1357103 
#RMSE: 1164.948 
#R-squared: -0.7127143

plot(gbmPredNew, productsReviewsSubDFNew$Volume)

# Capture and save the prediction values to CSV file
output <- productsReviewsSubDFNew

output$predictions <- gbmPredNew

write.csv(output, file="C2.T3output.csv", row.names = TRUE)


# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

