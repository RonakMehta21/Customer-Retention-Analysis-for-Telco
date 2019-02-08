## Import Data
library(h2o)
library(data.table)
library(purrr)
library(ggplot2)
library(hrbrthemes)
library(lattice)
library(plyr)
library(scales)
#getwd('C:/Users/home/Documents/Predicting Churn with H2O')
DT  <- fread("WA_Fn-UseC_-Telco-Customer-Churn.csv")
fCols <- map_lgl(DT, is.character) %>% names(DT)[.] %>% .[-1]
#DT[, fCols, with=F] %>% head %>% knitr::kable() 
DT[, (fCols) := lapply(.SD, factor), .SDcols=fCols]
nCols <- map_lgl(DT, is.numeric) %>% names(DT)[.]
#DT[, nCols, with=F] %>% map(summary)
DT[, SeniorCitizen := factor(SeniorCitizen)]
with(DT, {cor(MonthlyCharges, TotalCharges, use = "pairwise.complete.obs")})
h2o.init(nthreads = -1)
y <- "TotalCharges"
x <- setdiff(names(DT), c(y, "customerID"))

train <- DT[!is.na(TotalCharges)] %>% as.h2o()
test <- DT[is.na(TotalCharges)] %>% as.h2o()

dl.mod <- h2o.deeplearning(x, y, 
                           model_id = "impute_TotalCharges",
                           training_frame = train)

preds <- h2o.predict(dl.mod, test)
h2o.cbind(test["customerID"], preds)
DT <- h2o.rbind(train, test) %>% as.data.table()
pmap(
  list(col = c("tenure", "MonthlyCharges", "TotalCharges"),
       fill = c("#EF6461", "#FFA737", "#93B5C6"),
       title = c("Tenure has many null value counts as well as many high values",
                 "Majority of customers have low monthly charges",
                 "Right skewed distribution shows that majority of customers have low charges")),
  function(col, fill, title) {
    ggplot(DT, aes_string(col)) +
      geom_histogram(bins = 15, colour= "white", fill = fill) +
      ggtitle(label = col, subtitle = title) +
      theme_ipsum_tw()
  })
data <- as.h2o(DT)
result <- lm(tenure ~ TotalCharges,DT)
ggplot(DT, aes(x=TotalCharges,y=tenure))+geom_point()+stat_smooth(method = "lm",col="red")
a = count(DT, 'PhoneService')
b = count(DT, 'MultipleLines')
c = count(DT, 'OnlineSecurity')
d = count(DT,'OnlineBackup')
e = count(DT,'DeviceProtection')
f = count(DT,'TechSupport')
g = count(DT,'StreamingTV')
h = count(DT,'StreamingMovies')
count <- c(a[2,2],b[3,2],c[3,2],d[3,2],e[3,2],f[3,2],g[3,2],h[3,2])
service <- c("PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies")
barplot(count,main = "Services",xlab="Type of Service",ylab="Counts",col=rainbow(8),legend = service,beside=TRUE,args.legend=list(fill = rainbow(8), ncol = 2, title="Services",cex = 0.75))
tenure_group <- vector(mode="character",length = length(DT$tenure))
tenure_group[DT$tenure<15] <- "<15"
tenure_group[DT$tenure>=15 & DT$tenure<30] <- ">=15 and <30"
tenure_group[DT$tenure>=30 & DT$tenure<45] <- ">=30 and <45"
tenure_group[DT$tenure>=45 & DT$tenure<60] <- ">=45 and <60"
tenure_group[DT$tenure>=60 & DT$tenure<75] <- ">=60 and <75"
tenure_graph <- factor(tenure_group,levels = c("<15",">=15 and <30",">=30 and <45",">=45 and <60",">=60 and <75"),ordered = TRUE)
#DT <- cbind(DT,tenure_graph)
counts <- table(DT$Contract,tenure_graph)
churn <- table(DT$Churn,tenure_graph)
contract <- table(DT$Contract,DT$PaymentMethod)
barplot(counts,main = "Tenure vs Contract",xlab="Tenure",ylab="Counts",col=c("#0000FFFF","#0080FFFF","#00FFFFFF"),legend = rownames(counts),beside=TRUE,args.legend=list(x="top",title="Contract",cex= 0.75))
barplot(churn,main = "Tenure vs Churn",xlab="Tenure",ylab="Counts",col=terrain.colors(2),legend = rownames(churn),beside=TRUE,args.legend=list(x="top",title="Churn",cex= 0.75))
barplot(contract,main = "Payment Mode vs Type of Contract",xlab="Contract",ylab="Counts",col=terrain.colors(3),legend = rownames(contract),beside=TRUE,args.legend=list(x="topleft",title="Contract"))
j = count(DT, 'PaymentMethod')
n<-length(DT$PaymentMethod)
bp<- ggplot(j, aes(x="", y=freq, fill=PaymentMethod))+geom_bar(width = 1, stat = "identity")
pie <- bp + coord_polar("y", start=0)+geom_text(aes(y = j$freq/n + c(0, cumsum(j$freq)[-length(j$freq)]), label = percent((j$freq)/n)), size=5)
pie

splits <- h2o.splitFrame(data, ratios = c(0.6, 0.2), seed = 123)

train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

# Baseline if Churn = 0 always
nos <- test["Churn"] %>% as.data.table() %>% .[, .N, by=Churn] %>% .[,N] %>% .[2]
baseline <- nos / h2o.dim(test)[1]
y <- "Churn"
x <- setdiff(names(DT), c(y, "customerID"))
rf.mod <- h2o.randomForest(x, y,
                           training_frame = train,
                           model_id = "RandomForest_Model",
                           validation_frame = valid,
                           nfolds = 5,
                           seed = 123)
rf.cm <- h2o.confusionMatrix(rf.mod, test)
rf.cm %>% knitr::kable()
gbm.mod <- h2o.gbm(x, y,
                   training_frame = train,
                   validation_frame = valid,
                   nfolds = 5,
                   seed = 123)
gbm.cm <- h2o.confusionMatrix(gbm.mod, test)
gbm.cm %>% knitr::kable()
dl.mod <- h2o.deeplearning(x, y,
                           training_frame = train,
                           model_id = "DeepLearning_Model",
                           validation_frame = valid,
                           nfolds = 5,
                           seed = 123)
dl.cm <- h2o.confusionMatrix(dl.mod, test)
dl.cm %>% knitr::kable()
#h2o.shutdown(prompt = FALSE)