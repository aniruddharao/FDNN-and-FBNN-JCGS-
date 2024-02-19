library(ggplot2)
library(reshape2)
library(MASS)
library(mice)
library(lattice)
library(missForest)
library(fdapace)
library(refund)
library(Metrics)
library(grf)
library(e1071)
library(splines)
library(ggfortify)
library(neuralnet)
library(sigmoid)
library(splines2)



set.seed(735)




n=500
m=100
M=m
my=75
MY=my


frun=100



relud=function(aa){
  aa[which(aa>0)]=1
  aa[which(aa<0)]=0
  
  return(aa)
}





fdata=list()
for (i in 1:frun) {
  
  
  x = seq(0,1,len=m)        #gird values
  
  cov<-function(t,s,sig2=1,rho=0.5){ # matern ftn nu = 5/2
    d <- abs(outer(t,s,"-"))
    tmp2 <- sig2*(1+sqrt(5)*d/rho + 5*d^2/(3*rho^2))*exp(-sqrt(5)*d/rho)}
  COV = cov(x, x)
  
  
  rdata= mvrnorm(n, rep(0, length=length(x)), COV)
  
  
  #data= mvrnorm(n, rep(0, length=length(x)), COV) #+ .3*mvrnorm(n, rep(0, length=length(x)), .1*diag(length(x))) #error
  
  
  #data1=data.frame(data)
  
  
  rdat=data.frame(x=x,t(rdata))
  rdat=melt(rdat,id='x')
  #dat= data.frame(x=x, t(data))
  #dat= melt(dat, id="x")
  
  
  
  
  ey=matrix(rnorm(n*my),n,my)
  
  xy= seq(0,1,len=my) 
  beta1=5*sin(x*2*pi)
  beta2=3*sin(xy*3*pi)
  
  beta=(beta1)%*%t(beta2)
  
  fx=0.5*rdata%*%beta/m
  
  yy=fx+ey
  
  c1=colMeans(fx)
  c2=colMeans(ey)
  
  c11=fx-c1
  c111=(c11^2)
  c1111=(colSums(c111))/(n-1)
  
  c22=ey-c2
  c222=c22^2
  c2222=(colSums(c222))/(n-1)
  
  sn=sum(c1111)/sum(c2222)
  sn
  
  y=yy[1:(n/2),]
  xt=rdata[1:(n/2),]
  
  yp=yy[(n/2+1):n,]
  xtp=rdata[(n/2+1):n,]
  
  
  
  
  fdata[[i]]=list(y,xt,yp,xtp)
  
  
}




fdnn4.4 <- function(X, Y, S, step_size, niteration){
  
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  MY=ncol(y)
  # initialize parameters randomly
  W1_1 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_1 <- matrix(0, nrow = N, ncol = S)
  W1_2 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_2 <- matrix(0, nrow = N, ncol = S)
  
  W1_3 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_3 <- matrix(0, nrow = N, ncol = S)
  W1_4 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_4 <- matrix(0, nrow = N, ncol = S)
  
  W2_11 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_21 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_31 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_41 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  W2_12 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_22 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_32 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_42 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  W2_13 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_23 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_33 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_43 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  W2_14 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_24 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_34 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_44 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  
  
  
  b2_1 <- matrix(0, nrow = N, ncol = S)
  b2_2 <- matrix(0, nrow = N, ncol = S)
  b2_3 <- matrix(0, nrow = N, ncol = S)
  b2_4 <- matrix(0, nrow = N, ncol = S)
  
  #second layer
  b3_1 <- matrix(0, nrow = N, ncol = MY)
  
  W3_1 <- 0.01 * matrix(rnorm(S*MY), nrow = S)
  W3_2 <- 0.01 * matrix(rnorm(S*MY), nrow = S)
  W3_3 <- 0.01 * matrix(rnorm(S*MY), nrow = S)
  W3_4 <- 0.01 * matrix(rnorm(S*MY), nrow = S)
  
  
  
  
  # gradient descent loop to update weight and bias
  for (i in 0:niteration){
    # hidden layer, ReLU activation
    
    hidden_layer11 <- relu(b1_1+xt%*%W1_1/M)
    hidden_layer11 <- matrix(hidden_layer11, nrow  = N)
    
    
    hidden_layer12 <- relu(b1_2+xt%*%W1_2/M)
    hidden_layer12 <- matrix(hidden_layer12, nrow  = N)
    
    
    hidden_layer13 <- relu(b1_3+xt%*%W1_3/M)
    hidden_layer13 <- matrix(hidden_layer13, nrow  = N)
    
    
    hidden_layer14 <- relu(b1_4+xt%*%W1_4/M)
    hidden_layer14 <- matrix(hidden_layer14, nrow  = N)
    
    # class score
    
    
    hidden_layer21 <- relu(b2_1+(hidden_layer11%*%W2_11/S)+(hidden_layer12%*%W2_21/S)+(hidden_layer13%*%W2_31/S)+(hidden_layer14%*%W2_41/S))
    hidden_layer21 <- matrix(hidden_layer21, nrow  = N)
    
    hidden_layer22 <- relu(b2_2+(hidden_layer11%*%W2_12/S)+(hidden_layer12%*%W2_22/S)+(hidden_layer13%*%W2_32/S)+(hidden_layer14%*%W2_42/S))
    hidden_layer22 <- matrix(hidden_layer22, nrow  = N)
    
    hidden_layer23 <- relu(b2_3+(hidden_layer11%*%W2_13/S)+(hidden_layer12%*%W2_23/S)+(hidden_layer13%*%W2_33/S)+(hidden_layer14%*%W2_43/S))
    hidden_layer23 <- matrix(hidden_layer23, nrow  = N)
    
    hidden_layer24 <- relu(b2_4+(hidden_layer11%*%W2_14/S)+(hidden_layer12%*%W2_24/S)+(hidden_layer13%*%W2_34/S)+(hidden_layer14%*%W2_44/S))
    hidden_layer24 <- matrix(hidden_layer24, nrow  = N)
    
    
    # compute the loss: sofmax and regularization
    yhat=(b3_1+(hidden_layer21%*%W3_1/S)+(hidden_layer22%*%W3_2/S)+(hidden_layer23%*%W3_3/S)+(hidden_layer24%*%W3_4/S))
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    # check progress
    if (i%%1000 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    
    dW3_10=dscores
    
    dW3_1= t(hidden_layer21)%*%dW3_10
    
    db3_1=colSums(dW3_10)
    
    dW3_2= t(hidden_layer22)%*%dW3_10
    
    dW3_3= t(hidden_layer23)%*%dW3_10
    
    dW3_4= t(hidden_layer24)%*%dW3_10
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    dscores= -2*(Y-yhat)/N
    
    #dstep0=t(as.matrix(hidden_layer21*(1-hidden_layer21)))
    dW2_11.1=(as.matrix(relud(hidden_layer21)))
    dW2_11.2=as.matrix(dscores)%*%t(as.matrix(W3_1))
    dW2_11.3=dW2_11.2*dW2_11.1
    dW2_11.4=t(hidden_layer11)%*%dW2_11.3
    dW2_11=(dW2_11.4)
    
    
    db2_1=colSums(dW2_11.3)
    
    
    dW2_21.4=t(hidden_layer12)%*%dW2_11.3
    dW2_21=(dW2_21.4)    
    dW2_31.4=t(hidden_layer13)%*%dW2_11.3
    dW2_31=(dW2_31.4)    
    dW2_41.4=t(hidden_layer14)%*%dW2_11.3
    dW2_41=(dW2_41.4)
    
    
    
    
    dW2_12.1=(as.matrix(relud(hidden_layer22)))
    dW2_12.2=as.matrix(dscores)%*%t(as.matrix(W3_2))
    dW2_12.3=dW2_12.2*dW2_12.1
    dW2_12.4=t(hidden_layer11)%*%dW2_12.3
    dW2_12=(dW2_12.4)
    
    
    db2_2=colSums(dW2_12.3)
    
    
    dW2_22.4=t(hidden_layer12)%*%dW2_12.3
    dW2_22=(dW2_22.4)    
    dW2_32.4=t(hidden_layer13)%*%dW2_12.3
    dW2_32=(dW2_32.4)    
    dW2_42.4=t(hidden_layer14)%*%dW2_12.3
    dW2_42=(dW2_42.4)
    
    
    
    
    dW2_13.1=(as.matrix(relud(hidden_layer23)))
    dW2_13.2=as.matrix(dscores)%*%t(as.matrix(W3_3))
    dW2_13.3=dW2_13.2*dW2_13.1
    dW2_13.4=t(hidden_layer11)%*%dW2_13.3
    dW2_13=(dW2_13.4)
    
    
    db2_3=colSums(dW2_13.3)
    
    
    dW2_23.4=t(hidden_layer12)%*%dW2_13.3
    dW2_23=(dW2_23.4)    
    dW2_33.4=t(hidden_layer13)%*%dW2_13.3
    dW2_33=(dW2_33.4)    
    dW2_43.4=t(hidden_layer14)%*%dW2_13.3
    dW2_43=(dW2_43.4)
    
    
    
    dW2_14.1=(as.matrix(relud(hidden_layer24)))
    dW2_14.2=as.matrix(dscores)%*%t(as.matrix(W3_4))
    dW2_14.3=dW2_14.2*dW2_14.1
    dW2_14.4=t(hidden_layer11)%*%dW2_14.3
    dW2_14=(dW2_14.4)
    
    
    db2_4=colSums(dW2_14.3)
    
    
    dW2_24.4=t(hidden_layer12)%*%dW2_14.3
    dW2_24=(dW2_24.4)    
    dW2_34.4=t(hidden_layer13)%*%dW2_14.3
    dW2_34=(dW2_34.4)    
    dW2_44.4=t(hidden_layer14)%*%dW2_14.3
    dW2_44=(dW2_44.4)
    
    
    
    
    ########################
    dW1_1.1=dW2_11.3%*%t(W2_11)+dW2_12.3%*%t(W2_12)+dW2_13.3%*%t(W2_13)+dW2_14.3%*%t(W2_14)
    dW1_1.2=(as.matrix(relud(hidden_layer11)))
    dW1_1.3=dW1_1.1*dW1_1.2
    dW1_1.4=t(xt)%*%dW1_1.3
    dW1_1=dW1_1.4
    
    db1_1=colSums(dW1_1.3)
    
    
    dW1_2.1=dW2_11.3%*%t(W2_21)+dW2_12.3%*%t(W2_22)+dW2_13.3%*%t(W2_23)+dW2_14.3%*%t(W2_24)
    dW1_2.2=(as.matrix(relud(hidden_layer12)))
    dW1_2.3=dW1_2.1*dW1_2.2
    dW1_2.4=t(xt)%*%dW1_2.3
    dW1_2=dW1_2.4
    
    db1_2=colSums(dW1_2.3)
    
    
    
    dW1_3.1=dW2_11.3%*%t(W2_31)+dW2_12.3%*%t(W2_32)+dW2_13.3%*%t(W2_33)+dW2_14.3%*%t(W2_34)
    dW1_3.2=(as.matrix(relud(hidden_layer13)))
    dW1_3.3=dW1_3.1*dW1_3.2
    dW1_3.4=t(xt)%*%dW1_3.3
    dW1_3=dW1_1.4
    
    db1_3=colSums(dW1_3.3)
    
    
    dW1_4.1=dW2_11.3%*%t(W2_41)+dW2_12.3%*%t(W2_42)+dW2_13.3%*%t(W2_43)+dW2_14.3%*%t(W2_44)
    dW1_4.2=(as.matrix(relud(hidden_layer14)))
    dW1_4.3=dW1_4.1*dW1_4.2
    dW1_4.4=t(xt)%*%dW1_4.3
    dW1_4=dW1_4.4
    
    db1_4=colSums(dW1_4.3)
    
    # update parameter 
    
    W1_1=W1_1-step_size*dW1_1
    b1_1<- b1_1[1,]-step_size*db1_1
    b1_1=matrix(rep(b1_1,N),N,S, byrow = T)
    W1_2=W1_2-step_size*dW1_2
    b1_2<- b1_2[1,]-step_size*db1_2
    b1_2=matrix(rep(b1_2,N),N,S, byrow = T)
    W1_3=W1_3-step_size*dW1_3
    b1_3<- b1_3[1,]-step_size*db1_3
    b1_3=matrix(rep(b1_3,N),N,S, byrow = T)
    W1_4=W1_4-step_size*dW1_4
    b1_4<- b1_4[1,]-step_size*db1_4
    b1_4=matrix(rep(b1_4,N),N,S, byrow = T)
    
    W2_11 <- W2_11-step_size*dW2_11
    W2_12 <- W2_12-step_size*dW2_12
    W2_13 <- W2_13-step_size*dW2_13
    W2_14 <- W2_14-step_size*dW2_14 
    W2_21 <- W2_21-step_size*dW2_21
    W2_22 <- W2_22-step_size*dW2_22
    W2_23 <- W2_23-step_size*dW2_23
    W2_24 <- W2_24-step_size*dW2_24    
    W2_31 <- W2_31-step_size*dW2_31
    W2_32 <- W2_32-step_size*dW2_32
    W2_33 <- W2_33-step_size*dW2_33
    W2_34 <- W2_34-step_size*dW2_34    
    W2_41 <- W2_41-step_size*dW2_41
    W2_42 <- W2_42-step_size*dW2_42
    W2_43 <- W2_43-step_size*dW2_43
    W2_44 <- W2_44-step_size*dW2_44
    
    
    b2_1<- b2_1[1,]-step_size*db2_1
    b2_1=matrix(rep(b2_1,N),N,S, byrow = T)
    b2_2<- b2_2[1,]-step_size*db2_2
    b2_2=matrix(rep(b2_2,N),N,S, byrow = T)
    b2_3<- b2_3[1,]-step_size*db2_3
    b2_3=matrix(rep(b2_3,N),N,S, byrow = T)
    b2_4<- b2_4[1,]-step_size*db2_4
    b2_4=matrix(rep(b2_4,N),N,S, byrow = T)
    
    W3_1 <- W3_1-step_size*dW3_1
    W3_2 <- W3_2-step_size*dW3_2   
    W3_3 <- W3_3-step_size*dW3_3
    W3_4 <- W3_4-step_size*dW3_4
    
    b3_1 <- b3_1[1,]-step_size*db3_1
    b3_1=matrix(rep(b3_1,N),N,MY, byrow = T)   
    
    
    
    
  }
  return(list(W1_1, W1_2, W1_3, W1_4, b1_1, b1_2, b1_3, b1_4, 
              W2_11, W2_12, W2_13, W2_14, W2_21, W2_22, W2_23, W2_24, W2_31, W2_32, W2_33, W2_34, W2_41, W2_42, W2_43, W2_44,
              b2_1, b2_2, b2_3, b2_4, 
              W3_1, W3_2, W3_3, W3_4, b3_1,
              sqrt(loss),yhat))
}


fdnn.pred4.4 <- function(X, S, para = list()){
  
  
  xt=X
  N <- nrow(X)
  M <- ncol(xt) # number of timepoints for xt
  
  
  W1_1 <-para[[1]]
  W1_2 <-para[[2]]  
  W1_3 <-para[[3]]
  W1_4 <-para[[4]]
  
  
  b1_1 <-para[[5]]
  b1_2 <-para[[6]]  
  b1_3 <-para[[7]]
  b1_4 <-para[[8]]
  
  
  #second layer
  
  W2_11 <-para[[9]]
  W2_12 <-para[[10]]
  W2_13 <-para[[11]]
  W2_14 <-para[[12]]  
  W2_21 <-para[[13]]
  W2_22 <-para[[14]]
  W2_23 <-para[[15]]
  W2_24 <-para[[16]]  
  W2_31 <-para[[17]]
  W2_32 <-para[[18]]
  W2_33 <-para[[19]]
  W2_34 <-para[[20]]  
  W2_41 <-para[[21]]
  W2_42 <-para[[22]]
  W2_43 <-para[[23]]
  W2_44 <-para[[24]]
  
  
  b2_1 <-para[[25]]
  b2_2 <-para[[26]]  
  b2_3 <-para[[27]]
  b2_4 <-para[[28]]
  
  
  W3_1 <-para[[29]]
  W3_2 <-para[[30]]  
  W3_3 <-para[[31]]
  W3_4 <-para[[32]]
  
  b3_1 <-para[[33]]
  
  
  hidden_layer11 <- relu(b1_1+xt%*%W1_1/M)
  hidden_layer11 <- matrix(hidden_layer11, nrow  = N)
  
  
  hidden_layer12 <- relu(b1_2+xt%*%W1_2/M)
  hidden_layer12 <- matrix(hidden_layer12, nrow  = N)
  
  
  hidden_layer13 <- relu(b1_3+xt%*%W1_3/M)
  hidden_layer13 <- matrix(hidden_layer13, nrow  = N)
  
  
  hidden_layer14 <- relu(b1_4+xt%*%W1_4/M)
  hidden_layer14 <- matrix(hidden_layer14, nrow  = N)
  
  # class score
  
  
  hidden_layer21 <- relu(b2_1+(hidden_layer11%*%W2_11/S)+(hidden_layer12%*%W2_21/S)+(hidden_layer13%*%W2_31/S)+(hidden_layer14%*%W2_41/S))
  hidden_layer21 <- matrix(hidden_layer21, nrow  = N)
  
  hidden_layer22 <- relu(b2_2+(hidden_layer11%*%W2_12/S)+(hidden_layer12%*%W2_22/S)+(hidden_layer13%*%W2_32/S)+(hidden_layer14%*%W2_42/S))
  hidden_layer22 <- matrix(hidden_layer22, nrow  = N)
  
  hidden_layer23 <- relu(b2_3+(hidden_layer11%*%W2_13/S)+(hidden_layer12%*%W2_23/S)+(hidden_layer13%*%W2_33/S)+(hidden_layer14%*%W2_43/S))
  hidden_layer23 <- matrix(hidden_layer23, nrow  = N)
  
  hidden_layer24 <- relu(b2_4+(hidden_layer11%*%W2_14/S)+(hidden_layer12%*%W2_24/S)+(hidden_layer13%*%W2_34/S)+(hidden_layer14%*%W2_44/S))
  hidden_layer24 <- matrix(hidden_layer24, nrow  = N)
  
  
  # compute the loss: sofmax and regularization
  yhat=(b3_1+(hidden_layer21%*%W3_1/S)+(hidden_layer22%*%W3_2/S)+(hidden_layer23%*%W3_3/S)+(hidden_layer24%*%W3_4/S))
  
  
  
  return(yhat)  
}



fnnit=function(y,xt,yp,xtp,S,vi,vj,vk,step_size,niteration){
  
  
  Y=y
  X=xt
  
  
  fdnn.model <- fdnn4.4(X=xt, Y=Y,S=S, step_size = step_size, niteration = niteration)
  
  fdnn.pred.model <- fdnn.pred4.4(X=xtp, S=30, fdnn.model)
  t4p4=rmse(as.numeric(yp),as.numeric(fdnn.pred.model))
  fdnn.pred.model <- fdnn.pred4.4(X=xt, S=30, fdnn.model)
  t44=rmse(as.numeric(Y),as.numeric(fdnn.pred.model))
  t44
  t4p4
  
  
  
  rr=as.matrix(c(t44,t4p4))
  
  return(rr)
  
}



fnnrun=frun
finalresm=matrix(c(1,2),nrow=2)
for (i in 1:fnnrun){
  runs=fnnit(y=fdata[[i]][[1]],xt=fdata[[i]][[2]],yp=fdata[[i]][[3]],xtp=fdata[[i]][[4]],S=30,vi=5,vj=6,vk=7,step_size=.001,niteration=10000)
  finalresm=cbind(finalresm,runs)
  
}

finalresm=finalresm[,-1]

kk1=rowMeans(finalresm)

kk1
