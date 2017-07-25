#IMPORT THE owl DATASET IN R and install dplyr function to run the code. The dplyr is non generic function used for selecting the training dataset randomly
library(readxl)
library (dplyr)
#i <- as.matrix(owls15, 135, 5)
owlsBarny <- owls15[owls15$V5 == "BarnOwl", ]
owlsSnowy<- owls15[owls15$V5 == "SnowyOwl", ]
owlsLongEared<- owls15[owls15$V5 == "LongEaredOwl", ]
#dataframes usually have original indexes . That s why it s need to be rescaled
rownames(owlsBarny) <- 1:nrow(owlsBarny)
rownames(owlsSnowy) <- 1:nrow(owlsSnowy)
rownames(owlsLongEared) <- 1:nrow(owlsLongEared)
acclayer<-c()
for (v in 1: 10)
{
  #70% for training 30% for testing
  
  
  trainBarny <- sample_frac(owlsBarny, 0.7) # random selection of database for training
  trainSnowy <- sample_frac(owlsSnowy, 0.7)
  trainLongEared <- sample_frac(owlsLongEared, 0.7)
  # extracting numericalindexes of training set
  kidBarny <- as.numeric(rownames(trainBarny))
  kidSnowy <- as.numeric(rownames(trainSnowy))
  kidLongEared <- as.numeric(rownames(trainLongEared))
  #removing train indexes from main data
  testBarny <- owlsBarny[-kidBarny, ]
  testSnowy <- owlsSnowy[-kidSnowy, ]
  testLongEared<-owlsLongEared[-kidLongEared, ]
  testeddata<-rbind.data.frame(testBarny,testSnowy,testLongEared) #complete testing data
  bigtrain<-rbind.data.frame(trainBarny,trainSnowy,trainLongEared) #complete training data
  ################################main algortihm#######################################
  indexBarnSnow <- bigtrain$V5              #index of training data
  traineddata1 <- bigtrain[, -(ncol = 5)] #removing labels from training set
  #converting data frame to matrix for matrix operations
  i9 <- matrix(as.numeric(unlist(traineddata1)), nrow = nrow(traineddata1))
  meanarray <- apply(i9, 2, mean)    #mean and standard deviation for z normalisation
  sdarray <- apply(i9, 2, sd)
  i10 <- apply(i9, 1, '-', meanarray)
  ihh<-t(apply(i10,2,'/',sdarray))
  b <- matrix(1, 93, 1)             # was done for testing the data set even with bias inputs but bias input didnt give good results 
  biasinput <- cbind(ihh, b) #X
  convertbin <- cbind.data.frame(indexBarnSnow, b)
  convertbin1 <- data.frame(indexBarnSnow)
  convertbin1$BarnOwl <- ifelse(convertbin$index == "BarnOwl" , 1, 0) 
  convertbin1$SnowyOwl<-ifelse(convertbin$index=="SnowyOwl",1,0)
  convertbin1$LongEaredOwl<-ifelse(convertbin$index=="LongEaredOwl",1,0)
  # weight matrixes the rows represent number of nodes in precedin layer and columns represent number of nodes in the next layer
  weightmatrix<-matrix(qnorm(runif(64,min=pnorm(0),max=pnorm(1))),4,16)
  weightmatrix2<-matrix(qnorm(runif(256,min=pnorm(0),max=pnorm(1))),16,16)#W'2
  weightmatrix3<-matrix(qnorm(runif(48,min=pnorm(0),max=pnorm(1))),16,3) 
  #  weightmatrix <- matrix(rnorm(64, mean = 0, sd = 1), 4, 16)  #W1 rows 4 column 16(input layer 4 nodes and hidden layer 16 nodes )
  # weightmatrix2 <- matrix(rnorm(256, mean = 0, sd = 1), 16, 16) # W2 hidden layer 1 16 nodes hidden layer 2 16 nodes
  #weightmatrix3 <- matrix(rnorm(48, mean = 0, sd = 1), 16, 3)   # W3 hidden layer 2 16 nodes output layer 3 nodes
  real_val <- select(convertbin1, BarnOwl,SnowyOwl,LongEaredOwl)
  for (i in 1:500) # 70 number of iterations
  {
    ipforactivationlayer1 <- ihh %*% weightmatrix #z2=X*W1 ...matrix multiplication
    k2 <- apply(ipforactivationlayer1, 2, function(z) { 1 / (1 + exp(-z))}) #a2=f(z2)
    
    #biashidden<-cbind(k2,b) # incase for bias layers
    ipforactivationlayer2 <- k2 %*% weightmatrix2#Z3 = a2*W2
    k3 <-
      apply(ipforactivationlayer2, 2, function(z) {
        1 / (1 + exp(-z))
      }) # a3
    ipforactivationlayer3 <- k3 %*% weightmatrix3 #Z4=a3*W3
    finalvalue1 <- apply(ipforactivationlayer3, 2, function(z) { 1 / (1 + exp(-z))}) #y^=f(z4)official estimate of our test score
    reactivefinal <-apply(ipforactivationlayer3, 2, function(z) {exp(-z) / ((1 + exp(-z)) ^ 2)
    })#f'(z4)
    
    error <- -finalvalue1 + real_val
    delta3 <--(as.matrix(error * reactivefinal))#backpropagating error1 ->error*f'(z4)
    readyforback <-t(k3) %*% delta3 #should be equal to number of weights W3 backpropagating error1*t(a3) 
    ipforreactivation <-
      apply(ipforactivationlayer2, 2, function(z) {
        exp(-z) / ((1 + exp(-z)) ^ 2)
      })#f'(z3)
    ipforreactivation2<-apply(ipforactivationlayer1, 2, function(z) {
      exp(-z) / ((1 + exp(-z)) ^ 2)
    })
    # weightmatrixforlearningrate<-weightmatrix2[-(nrow=7),] for bias inputs considerations
    
    getreadyforlearnrate <- delta3 %*% (t(weightmatrix3))  
    backrate2 <- getreadyforlearnrate * ipforreactivation #backpropagating error2
    finalback <- t(k2)%*%(backrate2) # backpropagating error2*t(a2)
    
    getdelta<-learningrate2%*%t(weightmatrix2) #backpropagating error3
    backrate3<- getdelta*ipforreactivation2
    finalbackfinal<- t(ihh)%*%(backrate3)  
    #updating weight matrixes
    weightmatrix <- weightmatrix - (0.01*finalbackfinal/i) #learning rate= 0.301
    weightmatrix2 <- weightmatrix2 - (0.01* finalback/i)
    weightmatrix3 <- weightmatrix3 - (0.01* readyforback/i)
  }
  testindex<-data.frame(testeddata$V5)
  testeddata1<-testeddata[,-(ncol=5)]
  i99 <- matrix(as.numeric(unlist(testeddata1)), nrow = nrow(testeddata1))
  i111 <- apply(i99, 1, '-', meanarray) # normalising the data using mean and std deviation value of training database
  ihh11<-t(apply(i111,2,'/',sdarray))
  checktestactivation<-ihh11%*%weightmatrix
  activatedtest<- apply(checktestactivation, 2, function(z) { 1 / (1 + exp(-z))})
  checktestactivation2<-activatedtest %*% weightmatrix2
  activatedtest2<- apply(checktestactivation, 2, function(z) { 1 / (1 + exp(-z))})
  ipforfinallayer <- activatedtest2 %*% weightmatrix3
  finalpredicted <-
    apply(ipforfinallayer, 2, function(z) {
      1 / (1 + exp(-z))
    }) #y^=f(z'3)official estimate of our test score
  finalpredicted1<-apply(finalpredicted,1,max)   # calculating the max which would inheritably our predicted class
  pred<-c()
  for (i in 1:42) # total number of test classes 
  {
    for (j in 1:3) # total number of rows in test classes
    {
      if ((finalpredicted[i,1]==finalpredicted1[i]) && j==1) # for printing predicted labels
      {
        predictedclass=1
        pred<-c(pred,predictedclass)
      }
      if ((finalpredicted[i,2]==finalpredicted1[i]) && j==2)
      {
        predictedclass=2
        pred<-c(pred,predictedclass)
      }
      if ((finalpredicted[i,3]==finalpredicted1[i])  && j==3)
      {
        predictedclass=3
        pred<-c(pred,predictedclass)
      }
      
    }
  }
  testindex$actual <-ifelse(testeddata$V5 == "BarnOwl" ,1,
                            ifelse(testeddata$V5 == "SnowyOwl", 2, 3))
  actuallabels<-testindex$actual
  acc=0
  
  for (b in 1: 42)   # 42 is number of testing data
  {
    if (actuallabels[b]==pred[b])
    { 
      acc=acc+1 
    }
  }
  acc=(acc/42)*100           # accuracy
  acclayer<-c(acclayer,acc)  # accuracy over 10 random observations
  
}

sum(acclayer)/10
