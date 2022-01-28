#------------------------------------------------------------------
#   Course3 Task1: Get Started with R
#
#   Description:   In this task, will install necessary R packages,
#                  then use the library to read the data file to the
#                  data set.  Next, will review and pre-process the 
#                  data set, do visualization on the data set.
#                  Then split the data to training and test data sets.
#                  Build the machine learn model to analyze the
#                  data set. Then do the prediction.
#
#   Student Name:  Steve (Shifeng) Li
#
#   Date:          01/24/2022
#--------------------------------------------------------------------

# Install the "readr" package
install.packages("readr")

# Load the library "readr" into the runtime environment
library("readr")

#----Part One: Walk through the analysis steps for the cars data-----

# Create the CarsDataSet by reading the data from the iris.csv file
CarsDataSet <- read.csv("cars.csv")

# Check the data attributes in the CarsDataSet
attributes(CarsDataSet)
#$names
#[1] "name.of.car"     "speed.of.car"    "distance.of.car"
#
#$class
#[1] "data.frame"
#
#$row.names
#[1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#[24] 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46
#[47] 47 48 49 50

# Get the statistical key metrics of the CarsDataSet attributes
summary(CarsDataSet)
#name.of.car         speed.of.car  distance.of.car 
#Length:50          Min.   : 4.0   Min.   :  2.00  
#Class :character   1st Qu.:12.0   1st Qu.: 26.00  
#Mode  :character   Median :15.0   Median : 36.00  
#Mean   :15.4   Mean   : 42.98  
#3rd Qu.:19.0   3rd Qu.: 56.00  
#Max.   :25.0   Max.   :120.00  
 
# Display the structure of the CarsDataSet
str(CarsDataSet)
#'data.frame':	50 obs. of  3 variables:
#$ name.of.car    : chr  "Ford" "Jeep" "Honda" "KIA" ...
#$ speed.of.car   : int  4 4 7 7 8 9 10 10 10 11 ...
#$ distance.of.car: int  2 4 10 10 14 16 17 18 20 20 ...

# Show the names of the attributes within the Cars date set
names(CarsDataSet)
#[1] "name.of.car"     "speed.of.car"    "distance.of.car"

# Print out the instances within the name.of.car column in the Cars data set
CarsDataSet$name.of.car
#[1] "Ford"       "Jeep"       "Honda"      "KIA"        "Toyota "   
#[6] "BMW"        "Mercedes"   "GM"         "Hyundai"    "Infiniti"  
#[11] "Land Rover" "Lexus"      "Mazda"      "Mitsubishi" "Nissan"    
#[16] "GMC"        "Fiat"       "Chrysler"   "Dodge"      "Acura"     
#[21] "Audi"       "Chevrolet"  "Buick"      "Ford"       "Jeep"      
#[26] "Honda"      "KIA"        "Toyota "    "BMW"        "Mercedes"  
#[31] "GM"         "Hyundai"    "Infiniti"   "Land Rover" "Lexus"     
#[36] "Mazda"      "Mitsubishi" "Nissan"     "GMC"        "Fiat"      
#[41] "Chrysler"   "Dodge"      "Acura"      "Audi"       "Chevrolet" 
#[46] "Buick"      "Jeep"       "Honda"      "KIA"        "Dodge"     
 
# Print out the instances within the speed.of.car column in the Cars data set
CarsDataSet$speed.of.car
#[1]  4  4  7  7  8  9 10 10 10 11 11 12 12 12 12 13 13 13 13 14 14 14 14
#[24] 15 15 15 16 16 17 17 17 18 18 18 18 19 19 19 20 20 20 20 20 22 23 24
#[47] 24 24 24 25

# Print out the instances within the distance.of.car column in the Cars data set
CarsDataSet$distance.of.car
#[1]   2   4  10  10  14  16  17  18  20  20  22  24  26  26  26  26  28
#[18]  28  32  32  32  34  34  34  36  36  40  40  42  46  46  48  50  52
#[35]  54  54  56  56  60  64  66  68  70  76  80  84  85  92  93 120

# Do histogram plot for the name.of.car column in the Cars data set
# First, convert the string values of the car names to numeric code table with values 1 to 50 
CarNameCodeTable <- sample(c(CarsDataSet$name.of.car), 50, replace=TRUE)

# Then plot
hist(table(CarNameCodeTable))
# See the plot as the saved image file "Rplot_names_of_car.png"

# Do histogram plot for the speed.of.car column in the Cars data set
hist(CarsDataSet$speed.of.car)
# See the plot as the saved image file "Rplot_speed_of_car.png"

# Do histogram plot for the distance.of.car column in the Cars data set
hist(CarsDataSet$distance.of.car)
# See the plot as the saved image file "Rplot_distance_of_car.png"

# Scatter Plot for speed.of.car and distance.of.car
plot(CarsDataSet$speed.of.car, CarsDataSet$distance.of.car)
# See the plot as the saved image file "Rplot_speed_vs_distance"

# Normal Quantile Plot for speed of car
qqnorm(CarsDataSet$speed.of.car)
# See the plot as the saved image file "Rplot_Normal_Quantile_Plot_Speed_of_car.png"

# Normal Quantile Plot for distance of car
qqnorm(CarsDataSet$distance.of.car)
# See the plot as the saved image file "Rplot_Normal_Quantile_Plot_distance_of_car.png"

# Now, go through the data preprocess steps for the Cars data set

#CarsDataSet$name.of.car<-as.character.numeric_version(CarsDataSet$name.of.car)

# Count how mat NA's in the data set
summary(CarsDataSet)
#name.of.car         speed.of.car  distance.of.car 
#Length:50          Min.   : 4.0   Min.   :  2.00  
#Class :character   1st Qu.:12.0   1st Qu.: 26.00  
#Mode  :character   Median :15.0   Median : 36.00  
#Mean   :15.4   Mean   : 42.98  
#3rd Qu.:19.0   3rd Qu.: 56.00  
#Max.   :25.0   Max.   :120.00  

# Show if there is any missing data 
is.na(CarsDataSet)
#        name.of.car speed.of.car distance.of.car
#[1,]       FALSE        FALSE           FALSE
#[2,]       FALSE        FALSE           FALSE
#[3,]       FALSE        FALSE           FALSE
#[4,]       FALSE        FALSE           FALSE
#[5,]       FALSE        FALSE           FALSE
#[6,]       FALSE        FALSE           FALSE
#[7,]       FALSE        FALSE           FALSE
#[8,]       FALSE        FALSE           FALSE
#[9,]       FALSE        FALSE           FALSE
#[10,]       FALSE        FALSE           FALSE
#[11,]       FALSE        FALSE           FALSE
#[12,]       FALSE        FALSE           FALSE
#[13,]       FALSE        FALSE           FALSE
#[14,]       FALSE        FALSE           FALSE
#[15,]       FALSE        FALSE           FALSE
#[16,]       FALSE        FALSE           FALSE
#[17,]       FALSE        FALSE           FALSE
#[18,]       FALSE        FALSE           FALSE
#[19,]       FALSE        FALSE           FALSE
#[20,]       FALSE        FALSE           FALSE
#[21,]       FALSE        FALSE           FALSE
#[22,]       FALSE        FALSE           FALSE
#[23,]       FALSE        FALSE           FALSE
#[24,]       FALSE        FALSE           FALSE
#[25,]       FALSE        FALSE           FALSE
#[26,]       FALSE        FALSE           FALSE
#[27,]       FALSE        FALSE           FALSE
#[28,]       FALSE        FALSE           FALSE
#[29,]       FALSE        FALSE           FALSE
#[30,]       FALSE        FALSE           FALSE
#[31,]       FALSE        FALSE           FALSE
#[32,]       FALSE        FALSE           FALSE
#[33,]       FALSE        FALSE           FALSE
#[34,]       FALSE        FALSE           FALSE
#[35,]       FALSE        FALSE           FALSE
#[36,]       FALSE        FALSE           FALSE
#[37,]       FALSE        FALSE           FALSE
#[38,]       FALSE        FALSE           FALSE
#[39,]       FALSE        FALSE           FALSE
#[40,]       FALSE        FALSE           FALSE
#[41,]       FALSE        FALSE           FALSE
#[42,]       FALSE        FALSE           FALSE
#[43,]       FALSE        FALSE           FALSE
#[44,]       FALSE        FALSE           FALSE
#[45,]       FALSE        FALSE           FALSE
#[46,]       FALSE        FALSE           FALSE
#[47,]       FALSE        FALSE           FALSE
#[48,]       FALSE        FALSE           FALSE
#[49,]       FALSE        FALSE           FALSE
#[50,]       FALSE        FALSE           FALSE

# Create Testing and Training Sets for Cars Data Set

# Set the seed to creating a sequence of random numbers.

set.seed(123)

# Split the data into training and test sets with the split 70/30
CarsTrainSize<-round(nrow(CarsDataSet)*0.7)

CarsTestSize<-nrow(CarsDataSet)-CarsTrainSize

# Check the value of CarsTrainSize
CarsTrainSize
# 35

# Check the value of CarsTestSize
CarsTestSize
# 15

# Create the training and test sets
CarsTraining_indices<-sample(seq_len(nrow(CarsDataSet)), size=CarsTrainSize)

CarsTrainSet<-CarsDataSet[CarsTraining_indices,]

CarsTestSet<-CarsDataSet[-CarsTraining_indices,]

# Now, will go through the steps to run the Cars data set through modeling algorithm

# To create the linear model
CarsLRModel<-lm(distance.of.car~speed.of.car, CarsTrainSet)

# Check if the model is optimal
summary(CarsLRModel)
#Call:
#  lm(formula = distance.of.car ~ speed.of.car, data = CarsTrainSet)
#
#Residuals:
#  Min      1Q  Median      3Q     Max 
#-9.0012 -5.0012 -0.5603  2.1458 28.4109 
#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    

#(Intercept)  -35.2481     4.0712  -8.658 5.25e-10 ***
#  speed.of.car   5.0735     0.2519  20.143  < 2e-16 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#Residual standard error: 7.18 on 33 degrees of freedom
#Multiple R-squared:  0.9248,	Adjusted R-squared:  0.9225 
#F-statistic: 405.7 on 1 and 33 DF,  p-value: < 2.2e-16

# R-squared is 0.9225, so the regression line fits the data well.
# p-value < 2.2e-16, which is way less than 0.05, so the relationship
# between distance and speed is very significant.

# Now do the predication
CarsPrediction <- predict(CarsLRModel, CarsTestSet)

# To view the predication
CarsPrediction
#        1         2         6        16        18        20        22 
#-14.95415 -14.95415  10.41329  30.70724  30.70724  35.78073  35.78073 
#      23        34        35        38        39        44        46 
#35.78073  56.07468  56.07468  61.14817  66.22166  76.36864  86.51561 
#      47 
#86.51561 


#----Part Two:  Walk through the analysis steps for the iris data---

# Create the IrisDataSet by reading the data from the iris.csv file
IrisDataSet <- read.csv("iris.csv")

# Check the data attributes in the IrisDataSet
attributes(IrisDataSet)
#$names
#[1] "X"            "Sepal.Length" "Sepal.Width"  "Petal.Length"
#[5] "Petal.Width"  "Species"     
#
#$class
#[1] "data.frame"
#
#$row.names
#[1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
#[18]  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34
#[35]  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51
#[52]  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68
#[69]  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85
#[86]  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102
#[103] 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
#[120] 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136
#[137] 137 138 139 140 141 142 143 144 145 146 147 148 149 150

# Get the statistical key metrics of the IrisDataSet attributes
summary(IrisDataSet)
#X           Sepal.Length    Sepal.Width     Petal.Length  
#Min.   :  1.00   Min.   :4.300   Min.   :2.000   Min.   :1.000  
#1st Qu.: 38.25   1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600  
#Median : 75.50   Median :5.800   Median :3.000   Median :4.350  
#Mean   : 75.50   Mean   :5.843   Mean   :3.057   Mean   :3.758  
#3rd Qu.:112.75   3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100  
#Max.   :150.00   Max.   :7.900   Max.   :4.400   Max.   :6.900  
#Petal.Width      Species         
#Min.   :0.100   Length:150        
#1st Qu.:0.300   Class :character  
#Median :1.300   Mode  :character  
#Mean   :1.199                     
#3rd Qu.:1.800                     
#Max.   :2.500 

# Display the structure of the IrisDataSet
str(IrisDataSet)
#'data.frame':	150 obs. of  6 variables:
#$ X           : int  1 2 3 4 5 6 7 8 9 10 ...
#$ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
#$ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
#$ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
#$ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
#$ Species     : chr  "setosa" "setosa" "setosa" "setosa" ...

# Show the names of the attributes in the Iris data set
names(IrisDataSet)
#[1] "X"            "Sepal.Length" "Sepal.Width"  "Petal.Length"
#[5] "Petal.Width"  "Species"

# Print out the instances of the columns in the Iris Data Set

IrisDataSet$Sepal.Length
#[1] 5.1 4.9 4.7 4.6 5.0 5.4 4.6 5.0 4.4 4.9 5.4 4.8 4.8 4.3 5.8 5.7 5.4
#[18] 5.1 5.7 5.1 5.4 5.1 4.6 5.1 4.8 5.0 5.0 5.2 5.2 4.7 4.8 5.4 5.2 5.5
#[35] 4.9 5.0 5.5 4.9 4.4 5.1 5.0 4.5 4.4 5.0 5.1 4.8 5.1 4.6 5.3 5.0 7.0
#[52] 6.4 6.9 5.5 6.5 5.7 6.3 4.9 6.6 5.2 5.0 5.9 6.0 6.1 5.6 6.7 5.6 5.8
#[69] 6.2 5.6 5.9 6.1 6.3 6.1 6.4 6.6 6.8 6.7 6.0 5.7 5.5 5.5 5.8 6.0 5.4
#[86] 6.0 6.7 6.3 5.6 5.5 5.5 6.1 5.8 5.0 5.6 5.7 5.7 6.2 5.1 5.7 6.3 5.8
#[103] 7.1 6.3 6.5 7.6 4.9 7.3 6.7 7.2 6.5 6.4 6.8 5.7 5.8 6.4 6.5 7.7 7.7
#[120] 6.0 6.9 5.6 7.7 6.3 6.7 7.2 6.2 6.1 6.4 7.2 7.4 7.9 6.4 6.3 6.1 7.7
#[137] 6.3 6.4 6.0 6.9 6.7 6.9 5.8 6.8 6.7 6.7 6.3 6.5 6.2 5.9

IrisDataSet$Sepal.Width
#[1] 3.5 3.0 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 3.7 3.4 3.0 3.0 4.0 4.4 3.9
#[18] 3.5 3.8 3.8 3.4 3.7 3.6 3.3 3.4 3.0 3.4 3.5 3.4 3.2 3.1 3.4 4.1 4.2
#[35] 3.1 3.2 3.5 3.6 3.0 3.4 3.5 2.3 3.2 3.5 3.8 3.0 3.8 3.2 3.7 3.3 3.2
#[52] 3.2 3.1 2.3 2.8 2.8 3.3 2.4 2.9 2.7 2.0 3.0 2.2 2.9 2.9 3.1 3.0 2.7
#[69] 2.2 2.5 3.2 2.8 2.5 2.8 2.9 3.0 2.8 3.0 2.9 2.6 2.4 2.4 2.7 2.7 3.0
#[86] 3.4 3.1 2.3 3.0 2.5 2.6 3.0 2.6 2.3 2.7 3.0 2.9 2.9 2.5 2.8 3.3 2.7
#[103] 3.0 2.9 3.0 3.0 2.5 2.9 2.5 3.6 3.2 2.7 3.0 2.5 2.8 3.2 3.0 3.8 2.6
#[120] 2.2 3.2 2.8 2.8 2.7 3.3 3.2 2.8 3.0 2.8 3.0 2.8 3.8 2.8 2.8 2.6 3.0
#[137] 3.4 3.1 3.0 3.1 3.1 3.1 2.7 3.2 3.3 3.0 2.5 3.0 3.4 3.0

IrisDataSet$Petal.Length
#[1] 1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 1.5 1.6 1.4 1.1 1.2 1.5 1.3
#[18] 1.4 1.7 1.5 1.7 1.5 1.0 1.7 1.9 1.6 1.6 1.5 1.4 1.6 1.6 1.5 1.5 1.4
#[35] 1.5 1.2 1.3 1.4 1.3 1.5 1.3 1.3 1.3 1.6 1.9 1.4 1.6 1.4 1.5 1.4 4.7
#[52] 4.5 4.9 4.0 4.6 4.5 4.7 3.3 4.6 3.9 3.5 4.2 4.0 4.7 3.6 4.4 4.5 4.1
#[69] 4.5 3.9 4.8 4.0 4.9 4.7 4.3 4.4 4.8 5.0 4.5 3.5 3.8 3.7 3.9 5.1 4.5
#[86] 4.5 4.7 4.4 4.1 4.0 4.4 4.6 4.0 3.3 4.2 4.2 4.2 4.3 3.0 4.1 6.0 5.1
#[103] 5.9 5.6 5.8 6.6 4.5 6.3 5.8 6.1 5.1 5.3 5.5 5.0 5.1 5.3 5.5 6.7 6.9
#[120] 5.0 5.7 4.9 6.7 4.9 5.7 6.0 4.8 4.9 5.6 5.8 6.1 6.4 5.6 5.1 5.6 6.1
#[137] 5.6 5.5 4.8 5.4 5.6 5.1 5.1 5.9 5.7 5.2 5.0 5.2 5.4 5.1


IrisDataSet$Petal.Width
#[1] 0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 0.2 0.2 0.1 0.1 0.2 0.4 0.4
#[18] 0.3 0.3 0.3 0.2 0.4 0.2 0.5 0.2 0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.1 0.2
#[35] 0.2 0.2 0.2 0.1 0.2 0.2 0.3 0.3 0.2 0.6 0.4 0.3 0.2 0.2 0.2 0.2 1.4
#[52] 1.5 1.5 1.3 1.5 1.3 1.6 1.0 1.3 1.4 1.0 1.5 1.0 1.4 1.3 1.4 1.5 1.0
#[69] 1.5 1.1 1.8 1.3 1.5 1.2 1.3 1.4 1.4 1.7 1.5 1.0 1.1 1.0 1.2 1.6 1.5
#[86] 1.6 1.5 1.3 1.3 1.3 1.2 1.4 1.2 1.0 1.3 1.2 1.3 1.3 1.1 1.3 2.5 1.9
#[103] 2.1 1.8 2.2 2.1 1.7 1.8 1.8 2.5 2.0 1.9 2.1 2.0 2.4 2.3 1.8 2.2 2.3
#[120] 1.5 2.3 2.0 2.0 1.8 2.1 1.8 1.8 1.8 2.1 1.6 1.9 2.0 2.2 1.5 1.4 2.3
#[137] 2.4 1.8 1.8 2.1 2.4 2.3 1.9 2.3 2.5 2.3 1.9 2.0 2.3 1.8


IrisDataSet$Species
#[1] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[6] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[11] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[16] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[21] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[26] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[31] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[36] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[41] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[46] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
#[51] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[56] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[61] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[66] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[71] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[76] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[81] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[86] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[91] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[96] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
#[101] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[106] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[111] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[116] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[121] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[126] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[131] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[136] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[141] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
#[146] "virginica"  "virginica"  "virginica"  "virginica"  "virginica"


# Histogram Plots for the Iris data set

hist(IrisDataSet$Sepal.Length)
# View the plot with the saved image file "Hist_plot_Iris_Sepal_Length.png"

hist(IrisDataSet$Sepal.Width)
# View the plot with the saved image file "Hist_plot_Iris_Sepal_Width.png"

hist(IrisDataSet$Petal.Length)
# View the plot with the saved image file "Hist_plot_Iris_Petal_Length.png"

hist(IrisDataSet$Petal.Width)
# View the plot with the saved image file "Hist_plot_Iris_Petal_Width.png"

# For Species attribute plot, need to convert string to numeric value first

SpeciesCodeTable <- sample(c(IrisDataSet$Species), 150, replace=TRUE)

# Then plot
hist(table(SpeciesCodeTable))
# View the plot as the saved image file "Hist_plot_Iris_Species.png"

# Scatter plot for Sepal.Length
plot(IrisDataSet$Sepal.Length)
# View the plot as the saved image file "Scatter_plot_Iris_Sepal_Length.png"

# Scatter Plot for Petal.Width and Petal.Length
plot(IrisDataSet$Petal.Width, IrisDataSet$Petal.Length)
# See the plot as the saved image file "Scatter_plot_PetalWidth_vs_PetalLength.png"


# Normal Quantile plot for IrisDataSet
qqnorm(IrisDataSet)
#Error in FUN(X[[i]], ...) : 
#  only defined on a data frame with all numeric-alike variables

# Will fix the above error. The parameter of qqnorm needs to be column vector in numeric

IrisDataSet$Species<-as.numeric(IrisDataSet$Species)
#Warning message:
#NAs introduced by coercion

# Now, try to fix the coercion problem.

# First, need to re-run the code to get the orginal data back to the data set
IrisDataSet <- read.csv("iris.csv")

# Convert vector to factor
SpeciesFactor <- factor(IrisDataSet$Species)

# Convert factor into numeric value
SpeciesNumeric <- as.numeric(SpeciesFactor)

# Print the numeric value
SpeciesNumeric
#[1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#[45] 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#[89] 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#[133] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3

# Now assign the coverted numeric values back to IrisDataSet$Species
IrisDataSet$Species <- SpeciesNumeric

# Double check by printing out the IrisDataSet$SPecies
IrisDataSet$Species
#[1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#[45] 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#[89] 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#[133] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3


# Now we shall be able to do qqnorm(IrisDataSet$Species)
qqnorm(IrisDataSet$Species)
# See the plot as the saved image file "Normal_Q-Q_Plot_Iris.png"

# Set Seed
set.seed(123)

IrisTrainSize <- round(nrow(IrisDataSet) * 0.2)

IrisTestSize <- nrow(IrisDataSet) - IrisTrainSize

IrisTrainSize
# 30

IrisTestSize
# 120

# Create the training and test sets
IrisTraining_indices<-sample(seq_len(nrow(IrisDataSet)), size=IrisTrainSize)

IrisTrainSet <- IrisDataSet[IrisTraining_indices,]

IrisTestSet <- IrisDataSet[-IrisTraining_indices,]

# Build the model
IrisLinearModel <- lm(Petal.Width ~ Petal.Length, IrisTrainSet)

summary(IrisLinearModel)
#Call:
#  lm(formula = Petal.Width ~ Petal.Length, data = IrisTrainSet)
#
#Residuals:
#  Min       1Q   Median       3Q      Max 
#-0.36964 -0.10766  0.00591  0.08338  0.47607 
#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept)  -0.28053    0.07150  -3.923 0.000516 ***
#  Petal.Length  0.39365    0.01684  23.381  < 2e-16 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#Residual standard error: 0.1742 on 28 degrees of freedom
#Multiple R-squared:  0.9513,	Adjusted R-squared:  0.9495 
#F-statistic: 546.7 on 1 and 28 DF,  p-value: < 2.2e-16


# Prediction
IrisPrediction<-predict(IrisLinearModel, IrisTestSet)

# View the prediction result

IrisPrediction
#        1         2         3         4         5         6         8 
#0.2705864 0.2705864 0.2312211 0.3099518 0.2705864 0.3886824 0.3099518 
#       10        11        12        13        15        16        17 
#0.3099518 0.3099518 0.3493171 0.2705864 0.1918558 0.3099518 0.2312211 
#       18        19        20        21        22        24        25 
#0.2705864 0.3886824 0.3099518 0.3886824 0.3099518 0.3886824 0.4674131 
#       28        29        30        31        33        34        35 
#0.3099518 0.2705864 0.3493171 0.3493171 0.3099518 0.2705864 0.3099518 
#       36        37        38        39        40        42        44 
#0.1918558 0.2312211 0.2705864 0.2312211 0.3099518 0.2312211 0.3493171 
#       45        46        47        48        49        51        52 
#0.4674131 0.2705864 0.3493171 0.2705864 0.3099518 1.5696425 1.4909118 
#       53        54        55        56        57        58        59 
#1.6483732 1.2940852 1.5302772 1.4909118 1.5696425 1.0185278 1.5302772 
#       60        61        62        63        64        65        66 
#1.2547198 1.0972585 1.3728158 1.2940852 1.5696425 1.1366238 1.4515465 
#       67        68        69        70        71        73        75 
#1.4909118 1.3334505 1.4909118 1.2547198 1.6090078 1.6483732 1.4121812 
#       77        79        80        82        83        84        85 
#1.6090078 1.4909118 1.0972585 1.1759892 1.2547198 1.7271038 1.4909118 
#       86        87        88        89        93        94        95 
#1.4909118 1.5696425 1.4515465 1.3334505 1.2940852 1.0185278 1.3728158 
#       96        97        98       100       101       102       104 
#1.3728158 1.3728158 1.4121812 1.3334505 2.0813919 1.7271038 1.9239305 
#      105       107       108       110       111       112       113 
#2.0026612 1.4909118 2.1994879 2.1207572 1.7271038 1.8058345 1.8845652 
#      114       115       116       119       120       121       122
#1.6877385 1.7271038 1.8058345 2.4356799 1.6877385 1.9632959 1.6483732 
#      123       124       125       126       127       128       129 
#2.3569492 1.6483732 1.9632959 2.0813919 1.6090078 1.6483732 1.9239305 
#      130       131       132       133       134       135       138 
#2.0026612 2.1207572 2.2388532 1.9239305 1.7271038 1.9239305 1.8845652 
#      139       140       141       142       144       145       146 
#1.6090078 1.8451999 1.9239305 1.7271038 2.0420265 1.9632959 1.7664692 
#      149 
#1.8451999 


# Using different seed, to create different set of train and test sets
set.seed(405)

trainSet405 <- IrisDataSet[IrisTraining_indices,]

testSet405 <- IrisDataSet[-IrisTraining_indices,]

LinearModel405<-lm(trainSet405$Petal.Width ~ trainSet405$Petal.Length)

summary(LinearModel405)
#Call:
#  lm(formula = trainSet405$Petal.Width ~ trainSet405$Petal.Length)
#
#Residuals:
#  Min       1Q   Median       3Q      Max 
#-0.36964 -0.10766  0.00591  0.08338  0.47607 
#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept)              -0.28053    0.07150  -3.923 0.000516 ***
#  trainSet405$Petal.Length  0.39365    0.01684  23.381  < 2e-16 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#Residual standard error: 0.1742 on 28 degrees of freedom
#Multiple R-squared:  0.9513,	Adjusted R-squared:  0.9495 
#F-statistic: 546.7 on 1 and 28 DF,  p-value: < 2.2e-16

prediction405<-predict(LinearModel405, testSet405)
#Warning message:
#'newdata' had 120 rows but variables found have 30 rows

prediction405 
#        1         2         3         4         5         6         7         8 
#0.1524904 0.2705864 2.3569492 0.2312211 1.7271038 1.7664692 1.2940852 1.4515465 
#        9        10        11        12        13        14        15        16 
#1.7271038 1.5302772 1.9239305 0.9004318 1.2940852 0.3493171 0.2705864 1.6877385 
#       17        18        19        20        21        22        23        24 
#1.2153545 1.6877385 2.0420265 1.8845652 1.4515465 0.3099518 2.3175839 2.0026612 
#       25        26        27        28        29        30 
#2.1207572 0.2705864 0.2312211 1.5696425 0.1131251 0.3493171 
 

