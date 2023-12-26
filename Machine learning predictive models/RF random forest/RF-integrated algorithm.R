rm(list=ls())
dev.off()
dev.new()

####Load package
library(rpart) #classification and regression trees
library(partykit) #treeplots
library(MASS) #breast and pima indian data
library(ElemStatLearn) #prostate data
library(randomForest) #random forests
library(xgboost) #gradient boosting 
library(caret) #tune hyper-parameters



###########CART first
#Read the above data set
prostate <- read.csv('input.csv', header= TRUE)
pros.train <- subset(prostate, train == TRUE)[, 1:15]
pros.test = subset(prostate, train == FALSE)[, 1:15]

set.seed(123)
tree.pros <- rpart(output ~ ., data = pros.train)
tree.pros$cptable
plotcp(tree.pros)
cp <- min(tree.pros$cptable[3, ])
prune.tree.pros <- prune(tree.pros, cp = cp)
plot(as.party(tree.pros))
plot(as.party(prune.tree.pros))
party.pros.test <- predict(prune.tree.pros, 
                           newdata = pros.test)
rpart.resid <- party.pros.test - pros.test$output #calculate residual
mean(rpart.resid^2)

################RF
set.seed(123)
rf.pros <- randomForest(output ~ ., data = pros.train)
rf.pros
plot(rf.pros)
#Derived prediction#
write.csv(rf.pros$predicted,file = "RFpredicted-230.csv")

which.min(rf.pros$mse)
set.seed(123)
rf.pros.2 <- randomForest(output ~ ., data = pros.train, ntree = 200)
rf.pros.2
plot(rf.pros.2)

varImpPlot(rf.pros.2, scale = TRUE,
           main = "Variable Importance Plot - PSA Score")
importance(rf.pros.2)
rf.pros.test <- predict(rf.pros.2, newdata = pros.test)
#plot(rf.pros.test, pros.test$output)
rf.resid <- rf.pros.test - pros.test$output #calculate residual
mean(rf.resid^2)


set.seed(123)
rf.biop <- randomForest(output ~ ., data = pros.train)
rf.biop
plot(rf.biop)
which.min(rf.biop$err.rate[, 1])
set.seed(123)
rf.biop.2 <- randomForest(output ~ ., data = pros.train, ntree = 500)
#getTree(rf.biop,1)
rf.biop.2
rf.biop.test <- predict(rf.biop.2, 
                        newdata = pros.test, 
                        type = "response")

varImpPlot(rf.biop.2)
#End#
