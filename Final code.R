
install.packages('fastAdaboost')


library(caret)
library(fastAdaboost) 
library(randomForest)

rm(list = ls())
rm(OutVals)


## Load Data
setwd("~/OneDrive - Drexel University/Spring Quarter/STAT 642/Project")

data_2018 <- read.csv("ks-projects-201801.csv")


index <- sample(1:nrow(data_2018),5000,replace=FALSE)
data_2018 <- data_2018[index,]

## Explore the data

str(data_2018)
summary(data_2018)


## Pre-processing the data


# remove the records which have "failed" and "successful" statuses and convert the state column into a
# factor column with two levels

print(unique(data_2018$state))
status_filter <- c("failed","successful")
ks_data <- data_2018[data_2018$state %in% status_filter ,]

ks_data$state <- as.character(ks_data$state) # changing the column to character
ks_data$state <- as.factor(ks_data$state) # changing it back to factor variable with 2 levels


# adding calculated column (duration of the project) to the data for analysis
ks_data$duration <- as.Date(as.character(ks_data$deadline),format = "%Y-%m-%d") -
                    as.Date(as.character(ks_data$launched),format = "%Y-%m-%d")

# keeping columns in the dataset that are necessary for our analysis
ks_data <- ks_data[,c("name","main_category","currency","state",
                                "usd_pledged_real","usd_goal_real","duration","backers")]

# converting the name column to character
ks_data$name <- as.character(ks_data$name) 

# checking for outliers in the data
summary(ks_data)
boxplot(ks_data$usd_goal_real)


OutVals = boxplot(ks_data$usd_goal_real)$out

ks_data_out <- which(ks_data$usd_goal_real %in% OutVals)

ks_data <- ks_data[-ks_data_out,]





# checking for class imbalance
barplot(table(ks_data$state), main="status ")

# sampling the data
set.seed(712)
samp <- createDataPartition(ks_data$state, p=.60, list=FALSE)
train_data = ks_data[samp, ] 
test_data = ks_data[-samp, ]



# check for class imbalance
barplot(table(train_data$state), main="status")





## AdaBoost
# We use the adaboost() function from the fastAdaboost package
# to perform the basic Boosting classification. We need to
# specify the number of classifiers, using nIter. Later, 
# we can tune this as a hyperparameter.

boost_model <- adaboost(state ~main_category+usd_pledged_real+usd_goal_real+duration+backers, 
                      data = train_data,
                      nIter=10)

## Training Performance
boost.train <- predict(boost_model, 
                       train_data[,-4], 
                       type="class")


confusionMatrix(boost.train$class, 
                train_data$state,
                mode="prec_recall",
                positive="successful")

boost.test <- predict(boost_model,
                      test_data[,-4])

confusionMatrix(boost.test$class, 
                test_data$state,
                mode="prec_recall",
                positive="successful")

## Hyper-Parameter Tuning

ctrl <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 3)



set.seed(712)


fit.adaboost <- train(state~ main_category+usd_pledged_real+usd_goal_real+duration+backers, 
                      data=train_data, 
                      method="adaboost", 
                      trControl=ctrl)

print(fit.adaboost)
confusionMatrix(fit.adaboost)

# We can apply our best fitting model to our training data
# to obtain predictions
inpredsAda <- predict(object=fit.adaboost, 
                      newdata=train_data)

confusionMatrix(inpredsAda,
                train_data$state,
                positive="successful",
                mode="prec_recall")

# Finally, we can apply our model to our testing data to 
# obtain our outsample predictions.
outpredsboost <- predict(fit.adaboost, 
                         newdata=test_data)

# We can view the default confusionMatrix() information
# and the precision and recall information
confusionMatrix(outpredsboost, 
                test_data$state, 
                positive="successful", 
                mode="prec_recall")



#################################################################################################


## Random Forest
# We use the randomForest() function in the randomForest package to 
# build the model. 
# We specify that 500 trees will be used with the ntree argument.
# By default mtry is equal to the square root of our number of 
# predictor variables. After building a default model, we will
# tune the model by changing the mtry value.


summary(train_data)
set.seed(712)
rf_mod <- randomForest(state~ main_category+usd_pledged_real+usd_goal_real+duration+backers,
                       data=train_data,
                       importance=TRUE, 
                       proximity=TRUE, 
                       ntree=500)
rf_mod

# Variable Importance Plot
# We can view the most important variables in the random forest model.
varImpPlot(rf_mod, main="Variable Importance Plot")

## Training Performance
rf.train <- predict(rf_mod, 
                    train_data[,-4], 
                    type="class")
confusionMatrix(rf.train, 
                train_data$state,
                positive="successful",
                mode="prec_recall")

## Testing Performance


rf.test <- predict(rf_mod, 
                   test_data[,-4])

confusionMatrix(rf.test, 
                test_data$state,
                positive="successful",
                mode="prec_recall")


## Hyperparameter Tuning
# We can perform a grid search using the caret package to tune the 
# mtry hyperparameter, or the number of variables to randomly sample 
# as potential variables to split on.
# By default, mtry=sqrt(n_features), or:
sqrt(ncol(train_data)-1)

## Training the model using Repeated 5-fold Cross Validation and
# grid search

grids = expand.grid(mtry = seq(from = 1, to = 10, by = 1))

grid_ctrl <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3,
                          search="grid")

set.seed(712)
fit.rf <- train(state~ main_category+usd_pledged_real+usd_goal_real+duration+backers, 
                data=train_data, 
                method="rf", 
                trControl=grid_ctrl,
                tuneGrid=grids)
print(fit.rf)
confusionMatrix(fit.rf)

## Training Performance
rf.train1 <- predict(fit.rf, 
                     train_data[,-4])
confusionMatrix(rf.train1, 
                train_data$state,
                positive="successful",
                mode="prec_recall")


## Testing Performance
rf.test <- predict(fit.rf, 
                   test_data[,-4])
confusionMatrix(rf.test, 
                test_data$state,
                positive="successful",
                mode="prec_recall")

