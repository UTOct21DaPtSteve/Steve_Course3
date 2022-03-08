# Title: Predict sales volume

# Last update: 2022.03
# File: Course3Task4_Steve.R
# Project name: Market Basket Association in R

################
# Load packages
################

install.packages("caret")
install.packages("arules")
install.packages("arulesViz")
install.packages("readr")
install.packages("doParallel")  # for Win parallel processing (see below) 

library(caret)
library(arules)
library(arulesViz)
library(RColorBrewer)
library(doParallel)


##############################
# Enable Parallel processing
##############################
detectCores()          # detect number of cores
cl <- makeCluster(2)   # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()      # confirm number of cores being used by RStudio

# Get current directory
pwd <- getwd()

# Change directory to present work directory
setwd(pwd)

###############################
# Upload and inspect data set
###############################

# Upload the data set
originalTranDataset <- read.transactions("ElectronidexTransactions2017.csv", format = "basket", sep=",", rm.duplicates=TRUE)

# Inspect the data set
inspect (originalTranDataset)

# Number of transactions.
length (originalTranDataset) 
#[1] 9835

# Number of items per transaction

# Want to see all, so set the enough max.print value threshold
options(max.print= 10000)

# Now show the item size for all the 9835 transactions
size (originalTranDataset)

# Lists the item label details of each transaction by conversion (LIST must be capitalized)
LIST(originalTranDataset) 

# To see the item labels
itemLabels(originalTranDataset)
# Very cool!  Got exact 125 different merchandise items.

# Get the idea of the transaction data set with summary
summary(originalTranDataset)

#transactions as itemMatrix in sparse format with
# 9835 rows (elements/itemsets/transactions) and
# 125 columns (items) and a density of 0.03506172 
#
#most frequent items:
#                    iMac                HP Laptop CYBERPOWER Gamer Desktop            Apple Earpods 
#                    2519                     1909                     1809                     1715 
#       Apple MacBook Air                  (Other) 
#                    1530                    33622 
#
#element (itemset/transaction) length distribution:
#sizes
#   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20 
#   2 2163 1647 1294 1021  856  646  540  439  353  247  171  119   77   72   56   41   26   20   10   10 
#  21   22   23   25   26   27   29   30 
#  10    5    3    1    1    3    1    1 
#
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   2.000   3.000   4.383   6.000  30.000 
#
#includes extended item information - examples:
#  labels
#1 1TB Portable External Hard Drive
#2 2TB Portable External Hard Drive
#3                   3-Button Mouse

#############################################
#  Visualize the transaction dataset         
#############################################

# Visualize the top 10 item frequencies within the transactions as a bar chart. 


itemFrequencyPlot(originalTranDataset, topN = 10,
                  col = brewer.pal(8, 'Pastel2'),
                  main = 'Relative Item Frequency Plot',
                  type = "relative",
                  ylab = "Item Frequency (Relative)")
# See the image file "Top_10_Items_Frequency_Plot.png".

# Apply image function for the top 10 items
top10ItemsTranDataset <- originalTranDataset[, c("iMac", "HP Laptop", "CYBERPOWER Gamer Desktop", "Apple Earpods", "Apple MacBook Air", "Lenovo Desktop Computer", "Dell Desktop", "Apple MacBook Pro", "ViewSonic Monitor", "Acer Desktop")]

image(sample(top10ItemsTranDataset, 100))
# See the image file "Top_10_items_transactions_100_Sampling.png"

# See how iMac and HP Laptop are related to each other in the sampling transactions.
iMacHPLaptopTranDataset <- originalTranDataset[, c("iMac", "HP Laptop")]

image(sample(iMacHPLaptopTranDataset, 80))

##############################################################
#  Apply Apriori algorithm to find the association rules
##############################################################

# Have tried different combinations of supp/conf value pairs, then use supp=0.003 and conf=0.7 to have reasonable amount of rules, 23
rules <- apriori(originalTranDataset, parameter =  list(minlen=2, supp=0.003, conf=0.7))

summary(rules)
#set of 23 rules
#
#rule length distribution (lhs + rhs):sizes
# 3  4  5 
# 2 17  4 
#
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#3.000   4.000   4.000   4.087   4.000   5.000 
#
#summary of quality measures:
#   support           confidence        coverage             lift      
#Min.   :0.003050   Min.   :0.7018   Min.   :0.003864   Min.   :2.740  
#1st Qu.:0.003254   1st Qu.:0.7113   1st Qu.:0.004220   1st Qu.:2.777  
#Median :0.003559   Median :0.7381   Median :0.004575   Median :2.886  
#Mean   :0.003709   Mean   :0.7512   Mean   :0.004960   Mean   :3.234  
#3rd Qu.:0.003915   3rd Qu.:0.7831   3rd Qu.:0.005440   3rd Qu.:3.775  
#Max.   :0.005287   Max.   :0.9024   Max.   :0.007117   Max.   :4.649  
#    count      
#Min.   :30.00  
#1st Qu.:32.00  
#Median :35.00  
#Mean   :36.48  
#3rd Qu.:38.50  
#Max.   :52.00 
#mining info:
#               data ntransactions support confidence
#originalTranDataset          9835   0.003        0.7
#call
#apriori(data = originalTranDataset, parameter = list(minlen = 2, supp = 0.003, conf = 0.7))



inspect(rules)

#     lhs                                                                                     rhs         support     confidence coverage    lift     count
#[1]  {ASUS Monitor, LG Monitor}                                                           => {iMac}      0.003863752 0.7037037  0.005490595 2.747489 38   
#[2]  {ASUS 2 Monitor, ASUS Monitor}                                                       => {iMac}      0.005083884 0.7142857  0.007117438 2.788805 50   
#[3]  {ASUS Chromebook, Dell Desktop, ViewSonic Monitor}                                   => {HP Laptop} 0.003558719 0.7446809  0.004778851 3.836530 35   
#[4]  {Apple Magic Keyboard, ASUS 2 Monitor, Dell Desktop}                                 => {iMac}      0.003050330 0.7317073  0.004168785 2.856825 30   
#[5]  {Acer Desktop, ASUS 2 Monitor, Lenovo Desktop Computer}                              => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[6]  {ASUS 2 Monitor, Dell Desktop, Lenovo Desktop Computer}                              => {iMac}      0.005185562 0.7391304  0.007015760 2.885807 51   
#[7]  {HP Laptop, Logitech 3-button Mouse, ViewSonic Monitor}                              => {iMac}      0.003152008 0.7380952  0.004270463 2.881765 31   
#[8]  {Apple Magic Keyboard, ASUS Monitor, Dell Desktop}                                   => {HP Laptop} 0.003050330 0.7894737  0.003863752 4.067299 30   
#[9]  {Apple Magic Keyboard, ASUS Monitor, HP Laptop}                                      => {iMac}      0.004067107 0.7017544  0.005795628 2.739879 40   
#[10] {ASUS Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                           => {iMac}      0.003253686 0.7804878  0.004168785 3.047280 32   
#[11] {ASUS Monitor, Dell Desktop, Lenovo Desktop Computer}                                => {iMac}      0.003965430 0.7358491  0.005388917 2.872995 39   
#[12] {Apple Magic Keyboard, HP Laptop, Microsoft Office Home and Student 2016}            => {iMac}      0.003660397 0.7058824  0.005185562 2.755996 36   
#[13] {Dell Desktop, Microsoft Office Home and Student 2016, ViewSonic Monitor}            => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[14] {Lenovo Desktop Computer, Microsoft Office Home and Student 2016, ViewSonic Monitor} => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[15] {Dell Desktop, HP Monitor, ViewSonic Monitor}                                        => {HP Laptop} 0.003558719 0.7777778  0.004575496 4.007043 35   
#[16] {Acer Aspire, Apple Magic Keyboard, Dell Desktop}                                    => {iMac}      0.003050330 0.7692308  0.003965430 3.003329 30   
#[17] {Acer Aspire, Acer Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.003152008 0.7209302  0.004372140 3.714169 31   
#[18] {Acer Aspire, Dell Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.005287239 0.8125000  0.006507372 4.185928 52   
#[19] {Dell Desktop, Samsung Monitor, ViewSonic Monitor}                                   => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[20] {Acer Aspire, Dell Desktop, HP Laptop, ViewSonic Monitor}                            => {iMac}      0.003762074 0.7115385  0.005287239 2.778079 37   
#[21] {Acer Aspire, Dell Desktop, iMac, ViewSonic Monitor}                                 => {HP Laptop} 0.003762074 0.9024390  0.004168785 4.649286 37   
#[22] {Acer Desktop, Dell Desktop, iMac, ViewSonic Monitor}                                => {HP Laptop} 0.003253686 0.8000000  0.004067107 4.121530 32   
#[23] {Dell Desktop, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor}                => {iMac}      0.004372140 0.7049180  0.006202339 2.752231 43


##  Rules by lift
rules_by_lift <- sort(rules, by = "lift")

inspect (rules_by_lift)
#     lhs                                                                                     rhs         support     confidence coverage    lift     count
#[1]  {Acer Aspire, Dell Desktop, iMac, ViewSonic Monitor}                                 => {HP Laptop} 0.003762074 0.9024390  0.004168785 4.649286 37   
#[2]  {Acer Aspire, Dell Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.005287239 0.8125000  0.006507372 4.185928 52   
#[3]  {Acer Desktop, Dell Desktop, iMac, ViewSonic Monitor}                                => {HP Laptop} 0.003253686 0.8000000  0.004067107 4.121530 32   
#[4]  {Apple Magic Keyboard, ASUS Monitor, Dell Desktop}                                   => {HP Laptop} 0.003050330 0.7894737  0.003863752 4.067299 30   
#[5]  {Dell Desktop, HP Monitor, ViewSonic Monitor}                                        => {HP Laptop} 0.003558719 0.7777778  0.004575496 4.007043 35   
#[6]  {ASUS Chromebook, Dell Desktop, ViewSonic Monitor}                                   => {HP Laptop} 0.003558719 0.7446809  0.004778851 3.836530 35   
#[7]  {Acer Aspire, Acer Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.003152008 0.7209302  0.004372140 3.714169 31   
#[8]  {Dell Desktop, Microsoft Office Home and Student 2016, ViewSonic Monitor}            => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[9]  {Dell Desktop, Samsung Monitor, ViewSonic Monitor}                                   => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[10] {ASUS Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                           => {iMac}      0.003253686 0.7804878  0.004168785 3.047280 32   
#[11] {Acer Aspire, Apple Magic Keyboard, Dell Desktop}                                    => {iMac}      0.003050330 0.7692308  0.003965430 3.003329 30   
#[12] {ASUS 2 Monitor, Dell Desktop, Lenovo Desktop Computer}                              => {iMac}      0.005185562 0.7391304  0.007015760 2.885807 51   
#[13] {HP Laptop, Logitech 3-button Mouse, ViewSonic Monitor}                              => {iMac}      0.003152008 0.7380952  0.004270463 2.881765 31 
#[14] {ASUS Monitor, Dell Desktop, Lenovo Desktop Computer}                                => {iMac}      0.003965430 0.7358491  0.005388917 2.872995 39   
#[15] {Apple Magic Keyboard, ASUS 2 Monitor, Dell Desktop}                                 => {iMac}      0.003050330 0.7317073  0.004168785 2.856825 30   
#[16] {ASUS 2 Monitor, ASUS Monitor}                                                       => {iMac}      0.005083884 0.7142857  0.007117438 2.788805 50   
#[17] {Acer Aspire, Dell Desktop, HP Laptop, ViewSonic Monitor}                            => {iMac}      0.003762074 0.7115385  0.005287239 2.778079 37   
#[18] {Acer Desktop, ASUS 2 Monitor, Lenovo Desktop Computer}                              => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[19] {Lenovo Desktop Computer, Microsoft Office Home and Student 2016, ViewSonic Monitor} => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[20] {Apple Magic Keyboard, HP Laptop, Microsoft Office Home and Student 2016}            => {iMac}      0.003660397 0.7058824  0.005185562 2.755996 36   
#[21] {Dell Desktop, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor}                => {iMac}      0.004372140 0.7049180  0.006202339 2.752231 43   
#[22] {ASUS Monitor, LG Monitor}                                                           => {iMac}      0.003863752 0.7037037  0.005490595 2.747489 38   
#[23] {Apple Magic Keyboard, ASUS Monitor, HP Laptop}                                      => {iMac}      0.004067107 0.7017544  0.005795628 2.739879 40

##  Rules by support
rules_by_support <- sort(rules, by = "support")

inspect(rules_by_support)
#     lhs                                                                                     rhs         support     confidence coverage    lift     count
#[1]  {Acer Aspire, Dell Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.005287239 0.8125000  0.006507372 4.185928 52   
#[2]  {ASUS 2 Monitor, Dell Desktop, Lenovo Desktop Computer}                              => {iMac}      0.005185562 0.7391304  0.007015760 2.885807 51   
#[3]  {ASUS 2 Monitor, ASUS Monitor}                                                       => {iMac}      0.005083884 0.7142857  0.007117438 2.788805 50   
#[4]  {Dell Desktop, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor}                => {iMac}      0.004372140 0.7049180  0.006202339 2.752231 43   
#[5]  {Apple Magic Keyboard, ASUS Monitor, HP Laptop}                                      => {iMac}      0.004067107 0.7017544  0.005795628 2.739879 40   
#[6]  {ASUS Monitor, Dell Desktop, Lenovo Desktop Computer}                                => {iMac}      0.003965430 0.7358491  0.005388917 2.872995 39   
#[7]  {ASUS Monitor, LG Monitor}                                                           => {iMac}      0.003863752 0.7037037  0.005490595 2.747489 38   
#[8]  {Acer Aspire, Dell Desktop, HP Laptop, ViewSonic Monitor}                            => {iMac}      0.003762074 0.7115385  0.005287239 2.778079 37   
#[9]  {Acer Aspire, Dell Desktop, iMac, ViewSonic Monitor}                                 => {HP Laptop} 0.003762074 0.9024390  0.004168785 4.649286 37   
#[10] {Apple Magic Keyboard, HP Laptop, Microsoft Office Home and Student 2016}            => {iMac}      0.003660397 0.7058824  0.005185562 2.755996 36   
#[11] {ASUS Chromebook, Dell Desktop, ViewSonic Monitor}                                   => {HP Laptop} 0.003558719 0.7446809  0.004778851 3.836530 35   
#[12] {Dell Desktop, HP Monitor, ViewSonic Monitor}                                        => {HP Laptop} 0.003558719 0.7777778  0.004575496 4.007043 35   
#[13] {Dell Desktop, Microsoft Office Home and Student 2016, ViewSonic Monitor}            => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[14] {Dell Desktop, Samsung Monitor, ViewSonic Monitor}                                   => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[15] {Acer Desktop, ASUS 2 Monitor, Lenovo Desktop Computer}                              => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[16] {ASUS Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                           => {iMac}      0.003253686 0.7804878  0.004168785 3.047280 32   
#[17] {Lenovo Desktop Computer, Microsoft Office Home and Student 2016, ViewSonic Monitor} => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[18] {Acer Desktop, Dell Desktop, iMac, ViewSonic Monitor}                                => {HP Laptop} 0.003253686 0.8000000  0.004067107 4.121530 32   
#[19] {HP Laptop, Logitech 3-button Mouse, ViewSonic Monitor}                              => {iMac}      0.003152008 0.7380952  0.004270463 2.881765 31   
#[20] {Acer Aspire, Acer Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.003152008 0.7209302  0.004372140 3.714169 31   
#[21] {Apple Magic Keyboard, ASUS 2 Monitor, Dell Desktop}                                 => {iMac}      0.003050330 0.7317073  0.004168785 2.856825 30   
#[22] {Apple Magic Keyboard, ASUS Monitor, Dell Desktop}                                   => {HP Laptop} 0.003050330 0.7894737  0.003863752 4.067299 30   
#[23] {Acer Aspire, Apple Magic Keyboard, Dell Desktop}                                    => {iMac}      0.003050330 0.7692308  0.003965430 3.003329 30



##  Rules by confidence
rules_by_confidence <- sort(rules, by = "confidence")

inspect (rules_by_confidence)
#     lhs                                                                                     rhs         support     confidence coverage    lift     count
#[1]  {Acer Aspire, Dell Desktop, iMac, ViewSonic Monitor}                                 => {HP Laptop} 0.003762074 0.9024390  0.004168785 4.649286 37   
#[2]  {Acer Aspire, Dell Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.005287239 0.8125000  0.006507372 4.185928 52   
#[3]  {Acer Desktop, Dell Desktop, iMac, ViewSonic Monitor}                                => {HP Laptop} 0.003253686 0.8000000  0.004067107 4.121530 32   
#[4]  {Apple Magic Keyboard, ASUS Monitor, Dell Desktop}                                   => {HP Laptop} 0.003050330 0.7894737  0.003863752 4.067299 30   
#[5]  {Dell Desktop, Microsoft Office Home and Student 2016, ViewSonic Monitor}            => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[6]  {Dell Desktop, Samsung Monitor, ViewSonic Monitor}                                   => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[7]  {ASUS Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                           => {iMac}      0.003253686 0.7804878  0.004168785 3.047280 32   
#[8]  {Dell Desktop, HP Monitor, ViewSonic Monitor}                                        => {HP Laptop} 0.003558719 0.7777778  0.004575496 4.007043 35   
#[9]  {Acer Aspire, Apple Magic Keyboard, Dell Desktop}                                    => {iMac}      0.003050330 0.7692308  0.003965430 3.003329 30   
#[10] {ASUS Chromebook, Dell Desktop, ViewSonic Monitor}                                   => {HP Laptop} 0.003558719 0.7446809  0.004778851 3.836530 35   
#[11] {ASUS 2 Monitor, Dell Desktop, Lenovo Desktop Computer}                              => {iMac}      0.005185562 0.7391304  0.007015760 2.885807 51   
#[12] {HP Laptop, Logitech 3-button Mouse, ViewSonic Monitor}                              => {iMac}      0.003152008 0.7380952  0.004270463 2.881765 31   
#[13] {ASUS Monitor, Dell Desktop, Lenovo Desktop Computer}                                => {iMac}      0.003965430 0.7358491  0.005388917 2.872995 39   
#[14] {Apple Magic Keyboard, ASUS 2 Monitor, Dell Desktop}                                 => {iMac}      0.003050330 0.7317073  0.004168785 2.856825 30   
#[15] {Acer Aspire, Acer Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.003152008 0.7209302  0.004372140 3.714169 31   
#[16] {ASUS 2 Monitor, ASUS Monitor}                                                       => {iMac}      0.005083884 0.7142857  0.007117438 2.788805 50   
#[17] {Acer Aspire, Dell Desktop, HP Laptop, ViewSonic Monitor}                            => {iMac}      0.003762074 0.7115385  0.005287239 2.778079 37   
#[18] {Acer Desktop, ASUS 2 Monitor, Lenovo Desktop Computer}                              => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[19] {Lenovo Desktop Computer, Microsoft Office Home and Student 2016, ViewSonic Monitor} => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[20] {Apple Magic Keyboard, HP Laptop, Microsoft Office Home and Student 2016}            => {iMac}      0.003660397 0.7058824  0.005185562 2.755996 36   
#[21] {Dell Desktop, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor}                => {iMac}      0.004372140 0.7049180  0.006202339 2.752231 43   
#[22] {ASUS Monitor, LG Monitor}                                                           => {iMac}      0.003863752 0.7037037  0.005490595 2.747489 38   
#[23] {Apple Magic Keyboard, ASUS Monitor, HP Laptop}                                      => {iMac}      0.004067107 0.7017544  0.005795628 2.739879 40


# Check if there are any redundant rules
is.redundant(rules)
# No

## List out the redundant rules
inspect(rules[is.redundant(rules)])
# Not found 

# Remove the redundant rules and assign to a new variable 
nonRedundantRules <- rules[!is.redundant(rules)]

inspect(nonRedundantRules)

# There are 23 non-redundant rules
#     lhs                                                                                     rhs         support     confidence coverage    lift     count
#[1]  {ASUS Monitor, LG Monitor}                                                           => {iMac}      0.003863752 0.7037037  0.005490595 2.747489 38   
#[2]  {ASUS 2 Monitor, ASUS Monitor}                                                       => {iMac}      0.005083884 0.7142857  0.007117438 2.788805 50   
#[3]  {ASUS Chromebook, Dell Desktop, ViewSonic Monitor}                                   => {HP Laptop} 0.003558719 0.7446809  0.004778851 3.836530 35   
#[4]  {Apple Magic Keyboard, ASUS 2 Monitor, Dell Desktop}                                 => {iMac}      0.003050330 0.7317073  0.004168785 2.856825 30   
#[5]  {Acer Desktop, ASUS 2 Monitor, Lenovo Desktop Computer}                              => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[6]  {ASUS 2 Monitor, Dell Desktop, Lenovo Desktop Computer}                              => {iMac}      0.005185562 0.7391304  0.007015760 2.885807 51   
#[7]  {HP Laptop, Logitech 3-button Mouse, ViewSonic Monitor}                              => {iMac}      0.003152008 0.7380952  0.004270463 2.881765 31   
#[8]  {Apple Magic Keyboard, ASUS Monitor, Dell Desktop}                                   => {HP Laptop} 0.003050330 0.7894737  0.003863752 4.067299 30   
#[9]  {Apple Magic Keyboard, ASUS Monitor, HP Laptop}                                      => {iMac}      0.004067107 0.7017544  0.005795628 2.739879 40   
#[10] {ASUS Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                           => {iMac}      0.003253686 0.7804878  0.004168785 3.047280 32   
#[11] {ASUS Monitor, Dell Desktop, Lenovo Desktop Computer}                                => {iMac}      0.003965430 0.7358491  0.005388917 2.872995 39   
#[12] {Apple Magic Keyboard, HP Laptop, Microsoft Office Home and Student 2016}            => {iMac}      0.003660397 0.7058824  0.005185562 2.755996 36   
#[13] {Dell Desktop, Microsoft Office Home and Student 2016, ViewSonic Monitor}            => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[14] {Lenovo Desktop Computer, Microsoft Office Home and Student 2016, ViewSonic Monitor} => {iMac}      0.003253686 0.7111111  0.004575496 2.776410 32   
#[15] {Dell Desktop, HP Monitor, ViewSonic Monitor}                                        => {HP Laptop} 0.003558719 0.7777778  0.004575496 4.007043 35   
#[16] {Acer Aspire, Apple Magic Keyboard, Dell Desktop}                                    => {iMac}      0.003050330 0.7692308  0.003965430 3.003329 30   
#[17] {Acer Aspire, Acer Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.003152008 0.7209302  0.004372140 3.714169 31   
#[18] {Acer Aspire, Dell Desktop, ViewSonic Monitor}                                       => {HP Laptop} 0.005287239 0.8125000  0.006507372 4.185928 52   
#[19] {Dell Desktop, Samsung Monitor, ViewSonic Monitor}                                   => {iMac}      0.003355363 0.7857143  0.004270463 3.067686 33   
#[20] {Acer Aspire, Dell Desktop, HP Laptop, ViewSonic Monitor}                            => {iMac}      0.003762074 0.7115385  0.005287239 2.778079 37   
#[21] {Acer Aspire, Dell Desktop, iMac, ViewSonic Monitor}                                 => {HP Laptop} 0.003762074 0.9024390  0.004168785 4.649286 37   
#[22] {Acer Desktop, Dell Desktop, iMac, ViewSonic Monitor}                                => {HP Laptop} 0.003253686 0.8000000  0.004067107 4.121530 32   
#[23] {Dell Desktop, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor}                => {iMac}      0.004372140 0.7049180  0.006202339 2.752231 43


#####################################################################################
#  Visualize the rules
#####################################################################################

plot(rules)
#  See the image file "Scatter_plot_for_23_rules.png"

plot(rules[1:10], method = "graph", control=list(type="items"))
# See the image file "Graph_plot_for_10_rules.png"


# Relative item frequency plot for the top 5 items
arules::itemFrequencyPlot(top10ItemsTranDataset, topN=10, 
                  col = brewer.pal(8, 'Pastel2'),
                  main = 'Relative Item Frequency Plot',
                  type = "relative",
                  ylab = "Item Frequency (Relative)")


################################################
#   Analyze the rules to derive insights
################################################

#   First of all, let's sort the rules by lift and confidence 

df <- as(rules, "data.frame") 

#  Sort by lift (descending) and confidence (descending)
df[order(-df$lift, -df$confidence), ]

#                                                                                          rules     support confidence    coverage     lift count
#21                             {Acer Aspire,Dell Desktop,iMac,ViewSonic Monitor} => {HP Laptop} 0.003762074  0.9024390 0.004168785 4.649286    37
#18                                  {Acer Aspire,Dell Desktop,ViewSonic Monitor} => {HP Laptop} 0.005287239  0.8125000 0.006507372 4.185928    52
#22                            {Acer Desktop,Dell Desktop,iMac,ViewSonic Monitor} => {HP Laptop} 0.003253686  0.8000000 0.004067107 4.121530    32
#8                               {Apple Magic Keyboard,ASUS Monitor,Dell Desktop} => {HP Laptop} 0.003050330  0.7894737 0.003863752 4.067299    30
#15                                   {Dell Desktop,HP Monitor,ViewSonic Monitor} => {HP Laptop} 0.003558719  0.7777778 0.004575496 4.007043    35
#3                               {ASUS Chromebook,Dell Desktop,ViewSonic Monitor} => {HP Laptop} 0.003558719  0.7446809 0.004778851 3.836530    35
#17                                  {Acer Aspire,Acer Desktop,ViewSonic Monitor} => {HP Laptop} 0.003152008  0.7209302 0.004372140 3.714169    31
#13            {Dell Desktop,Microsoft Office Home and Student 2016,ViewSonic Monitor} => {iMac} 0.003355363  0.7857143 0.004270463 3.067686    33
#19                                   {Dell Desktop,Samsung Monitor,ViewSonic Monitor} => {iMac} 0.003355363  0.7857143 0.004270463 3.067686    33
#10                           {ASUS Monitor,Lenovo Desktop Computer,ViewSonic Monitor} => {iMac} 0.003253686  0.7804878 0.004168785 3.047280    32
#16                                    {Acer Aspire,Apple Magic Keyboard,Dell Desktop} => {iMac} 0.003050330  0.7692308 0.003965430 3.003329    30
#6                               {ASUS 2 Monitor,Dell Desktop,Lenovo Desktop Computer} => {iMac} 0.005185562  0.7391304 0.007015760 2.885807    51
#7                               {HP Laptop,Logitech 3-button Mouse,ViewSonic Monitor} => {iMac} 0.003152008  0.7380952 0.004270463 2.881765    31
#11                                {ASUS Monitor,Dell Desktop,Lenovo Desktop Computer} => {iMac} 0.003965430  0.7358491 0.005388917 2.872995    39
#4                                  {Apple Magic Keyboard,ASUS 2 Monitor,Dell Desktop} => {iMac} 0.003050330  0.7317073 0.004168785 2.856825    30
#2                                                       {ASUS 2 Monitor,ASUS Monitor} => {iMac} 0.005083884  0.7142857 0.007117438 2.788805    50
#20                             {Acer Aspire,Dell Desktop,HP Laptop,ViewSonic Monitor} => {iMac} 0.003762074  0.7115385 0.005287239 2.778079    37
#5                               {Acer Desktop,ASUS 2 Monitor,Lenovo Desktop Computer} => {iMac} 0.003253686  0.7111111 0.004575496 2.776410    32
#14 {Lenovo Desktop Computer,Microsoft Office Home and Student 2016,ViewSonic Monitor} => {iMac} 0.003253686  0.7111111 0.004575496 2.776410    32
#12            {Apple Magic Keyboard,HP Laptop,Microsoft Office Home and Student 2016} => {iMac} 0.003660397  0.7058824 0.005185562 2.755996    36
#23                 {Dell Desktop,HP Laptop,Lenovo Desktop Computer,ViewSonic Monitor} => {iMac} 0.004372140  0.7049180 0.006202339 2.752231    43
#1                                                           {ASUS Monitor,LG Monitor} => {iMac} 0.003863752  0.7037037 0.005490595 2.747489    38
#9                                       {Apple Magic Keyboard,ASUS Monitor,HP Laptop} => {iMac} 0.004067107  0.7017544 0.005795628 2.739879    40

###The Insights are observed, based on the confidence and life values in the rules listed by descending order of lift and confidencec:

#   1) When transactions have line items like desktops and monitors, the transaction may also have HP Laptop line item too.
#   2) When transactions have line items like desktops and monitors, the transaction may also have iMac line item too.
#   3) In the cases as the above two categories, the chance that the transactions have the HP Laptop as the result is higher than having the iMac as the result. 




################################################
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

