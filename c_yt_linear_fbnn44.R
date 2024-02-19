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



fbnn4.4 <- function(X, Y, S, vi, vj, vm,vn,vk1,vk2, step_size = 0.5, niteration){
  
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  MY=ncol(y)
  
  
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  
  
  vm1 <- seq(0, 1, by=1/(S-1))
  vm_v <- t(as.matrix(bSpline(vm1,df=vm)))
  
  vn1 <- seq(0, 1, by=1/(S-1))
  vn_v <- t(as.matrix(bSpline(vn1,df=vn)))
  
  vks <- seq(0, 1, by=1/(S-1))
  vks_v <- t(as.matrix(bSpline(vks,df=vk1)))
  
  vky <- seq(0, 1, by=1/(MY-1))
  vky_v <- t(as.matrix(bSpline(vky,df=vk2)))
  
  
  
  # initialize parameters randomly
  W1_1 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_1 <- matrix(0, nrow = N, ncol = S)
  W1_2 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_2 <- matrix(0, nrow = N, ncol = S)
  W1_3 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_3 <- matrix(0, nrow = N, ncol = S)
  W1_4 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_4 <- matrix(0, nrow = N, ncol = S)
  
  
  W2_11 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_12 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_13 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_14 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_21 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_22 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_23 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_24 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_31 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_32 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_33 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_34 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_41 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_42 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_43 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_44 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  
  
  b2_1 <- matrix(0, nrow = N, ncol = S)
  b2_2 <- matrix(0, nrow = N, ncol = S)  
  b2_3 <- matrix(0, nrow = N, ncol = S)
  b2_4 <- matrix(0, nrow = N, ncol = S)
  
  
  
  W3_1 <- 0.01 * matrix(rnorm(vk1*vk2), nrow = vk1)
  b3_1 <- matrix(0, nrow = N, ncol = MY)
  W3_2 <- 0.01 * matrix(rnorm(vk1*vk2), nrow = vk1)
  W3_3 <- 0.01 * matrix(rnorm(vk1*vk2), nrow = vk1)
  W3_4 <- 0.01 * matrix(rnorm(vk1*vk2), nrow = vk1)
  
  
  
  
  
  # gradient descent loop to update weight and bias
  for (r in 0:niteration){
    # hidden layer, ReLU activation
    
    ai1_1=t(as.matrix(vi_v)%*%t(xt)/M)
    
    
    awv1_1.1=ai1_1%*%W1_1
    awv1_1=awv1_1.1%*%vj_v
    
    hidden_layer11 <- relu(b1_1+awv1_1)
    hidden_layer11<- matrix(hidden_layer11, nrow  = N)
    
    
    
    awv1_2.1=ai1_1%*%W1_2
    awv1_2=awv1_2.1%*%vj_v
    
    hidden_layer12 <- relu(b1_2+awv1_2)
    hidden_layer12<- matrix(hidden_layer12, nrow  = N)
    
    awv1_3.1=ai1_1%*%W1_3
    awv1_3=awv1_3.1%*%vj_v
    
    hidden_layer13 <- relu(b1_3+awv1_3)
    hidden_layer13<- matrix(hidden_layer13, nrow  = N)
    
    awv1_4.1=ai1_1%*%W1_4
    awv1_4=awv1_4.1%*%vj_v
    
    hidden_layer14 <- relu(b1_4+awv1_4)
    hidden_layer14<- matrix(hidden_layer14, nrow  = N)
    
    
    #layer 2
    ai2_11=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_11.1=ai2_11%*%W2_11
    awv2_11=awv2_11.1%*%vn_v
    
    ai2_21=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_21.1=ai2_21%*%W2_21
    awv2_21=awv2_21.1%*%vn_v
    
    ai2_31=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_31.1=ai2_31%*%W2_31
    awv2_31=awv2_31.1%*%vn_v
    
    ai2_41=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_41.1=ai2_41%*%W2_41
    awv2_41=awv2_41.1%*%vn_v
    
    
    hidden_layer21 <- relu(b2_1+awv2_11+awv2_21+awv2_31+awv2_41)
    hidden_layer21<- matrix(hidden_layer21, nrow  = N)
    
    
    
    ai2_12=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_12.1=ai2_12%*%W2_12
    awv2_12=awv2_12.1%*%vn_v
    
    ai2_22=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_22.1=ai2_22%*%W2_22
    awv2_22=awv2_22.1%*%vn_v
    
    ai2_32=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_32.1=ai2_32%*%W2_32
    awv2_32=awv2_32.1%*%vn_v
    
    ai2_42=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_42.1=ai2_42%*%W2_42
    awv2_42=awv2_42.1%*%vn_v
    
    
    hidden_layer22 <- relu(b2_2+awv2_12+awv2_22+awv2_32+awv2_42)
    hidden_layer22<- matrix(hidden_layer22, nrow  = N)
    
    
    
    ai2_13=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_13.1=ai2_13%*%W2_13
    awv2_13=awv2_13.1%*%vn_v
    
    ai2_23=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_23.1=ai2_23%*%W2_23
    awv2_23=awv2_23.1%*%vn_v
    
    ai2_33=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_33.1=ai2_33%*%W2_33
    awv2_33=awv2_33.1%*%vn_v
    
    ai2_43=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_43.1=ai2_43%*%W2_43
    awv2_43=awv2_43.1%*%vn_v
    
    
    hidden_layer23 <- relu(b2_3+awv2_13+awv2_23+awv2_33+awv2_43)
    hidden_layer23<- matrix(hidden_layer23, nrow  = N)
    
    
    ai2_14=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_14.1=ai2_14%*%W2_14
    awv2_14=awv2_14.1%*%vn_v
    
    ai2_24=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_24.1=ai2_24%*%W2_24
    awv2_24=awv2_24.1%*%vn_v
    
    ai2_34=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_34.1=ai2_34%*%W2_34
    awv2_34=awv2_34.1%*%vn_v
    
    ai2_44=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_44.1=ai2_44%*%W2_44
    awv2_44=awv2_44.1%*%vn_v
    
    
    hidden_layer24 <- relu(b2_4+awv2_14+awv2_24+awv2_34+awv2_44)
    hidden_layer24<- matrix(hidden_layer24, nrow  = N)
    
    #last layer
    ai3_1=t(as.matrix(vks_v)%*%t(hidden_layer21)/S)
    
    awv3_1.1=ai3_1%*%W3_1
    awv3_1=awv3_1.1%*%vky_v
    
    
    ai3_2=t(as.matrix(vks_v)%*%t(hidden_layer22)/S)
    
    awv3_2.1=ai3_2%*%W3_2
    awv3_2=awv3_2.1%*%vky_v
    
    
    ai3_3=t(as.matrix(vks_v)%*%t(hidden_layer23)/S)
    
    awv3_3.1=ai3_3%*%W3_3
    awv3_3=awv3_3.1%*%vky_v
    
    
    ai3_4=t(as.matrix(vks_v)%*%t(hidden_layer24)/S)
    
    awv3_4.1=ai3_4%*%W3_4
    awv3_4=awv3_4.1%*%vky_v
    
    
    
    hidden_layer3 <- (b3_1+awv3_1+awv3_2+awv3_3+awv3_4)
    hidden_layer3<- matrix(hidden_layer3, nrow  = N)
    
    
    
    # compute the loss: sofmax and regularization
    yhat=hidden_layer3
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    # check progress
    if (r%%1000 == 0 | r == niteration){
      print(paste("iteration", r,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    dW3_1.1=t(ai3_1)%*%dscores
    dW3_1=dW3_1.1%*%t(vky_v)
    
    dW3_2.1=t(ai3_2)%*%dscores
    dW3_2=dW3_2.1%*%t(vky_v)
    
    dW3_3.1=t(ai3_3)%*%dscores 
    dW3_3=dW3_3.1%*%t(vky_v)
    
    dW3_4.1=t(ai3_4)%*%dscores
    dW3_4=dW3_4.1%*%t(vky_v)
    
    db3_1=colSums(dscores)
    
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    
    dW2_11.0=(W3_1)%*%vky_v
    dW2_11.05=t(dW2_11.0)%*%vks_v
    dW2_11.1=dscores%*%dW2_11.05
    dW2_11.2=(as.matrix(relud(hidden_layer21)))
    dW2_11.3=dW2_11.1*dW2_11.2
    dW2_11.4=vn_v%*%t(dW2_11.3)
    dW2_11.5=dW2_11.4%*%(ai2_11)
    dW2_11=t(dW2_11.5) 
    
    
    db2_1=colSums(dW2_11.3)
    
    
    dW2_21.5=dW2_11.4%*%(ai2_21)
    dW2_21=t(dW2_21.5) 
    
    dW2_31.5=dW2_11.4%*%(ai2_31)
    dW2_31=t(dW2_31.5) 
    
    dW2_41.5=dW2_11.4%*%(ai2_41)
    dW2_41=t(dW2_41.5) 
    
    
    
    dW2_12.0=(W3_2)%*%vky_v
    dW2_12.05=t(dW2_12.0)%*%vks_v
    dW2_12.1=dscores%*%dW2_12.05
    dW2_12.2=(as.matrix(relud(hidden_layer22)))
    dW2_12.3=dW2_12.1*dW2_12.2
    dW2_12.4=vn_v%*%t(dW2_12.3)
    dW2_12.5=dW2_12.4%*%(ai2_12)
    dW2_12=t(dW2_12.5) 
    
    
    db2_2=colSums(dW2_12.3)
    
    
    dW2_22.5=dW2_12.4%*%(ai2_22)
    dW2_22=t(dW2_22.5) 
    
    dW2_32.5=dW2_12.4%*%(ai2_23)
    dW2_32=t(dW2_32.5) 
    
    dW2_42.5=dW2_12.4%*%(ai2_24)
    dW2_42=t(dW2_42.5) 
    
    
    
    
    
    
    dW2_13.0=(W3_3)%*%vky_v
    dW2_13.05=t(dW2_13.0)%*%vks_v
    dW2_13.1=dscores%*%dW2_13.05
    dW2_13.2=(as.matrix(relud(hidden_layer23)))
    dW2_13.3=dW2_13.1*dW2_13.2
    dW2_13.4=vn_v%*%t(dW2_13.3)
    dW2_13.5=dW2_13.4%*%(ai2_13)
    dW2_13=t(dW2_13.5) 
    
    
    db2_3=colSums(dW2_13.3)
    
    
    dW2_23.5=dW2_13.4%*%(ai2_23)
    dW2_23=t(dW2_23.5) 
    
    dW2_33.5=dW2_13.4%*%(ai2_33)
    dW2_33=t(dW2_33.5) 
    
    dW2_43.5=dW2_13.4%*%(ai2_43)
    dW2_43=t(dW2_43.5) 
    
    
    
    dW2_14.0=(W3_4)%*%vky_v
    dW2_14.05=t(dW2_14.0)%*%vks_v
    dW2_14.1=dscores%*%dW2_14.05
    dW2_14.2=(as.matrix(relud(hidden_layer24)))
    dW2_14.3=dW2_14.1*dW2_14.2
    dW2_14.4=vn_v%*%t(dW2_14.3)
    dW2_14.5=dW2_14.4%*%(ai2_14)
    dW2_14=t(dW2_14.5) 
    
    
    db2_4=colSums(dW2_14.3)
    
    
    dW2_24.5=dW2_14.4%*%(ai2_24)
    dW2_24=t(dW2_24.5)
    
    dW2_34.5=dW2_14.4%*%(ai2_34)
    dW2_34=t(dW2_34.5) 
    
    dW2_44.5=dW2_14.4%*%(ai2_44)
    dW2_44=t(dW2_44.5) 
    
    
    
    
    
    dW1_1.0=t(vm_v)%*%W2_11
    dW1_1.0=dW1_1.0%*%vn_v
    dW1_1.1=t(vm_v)%*%W2_12
    dW1_1.1=dW1_1.1%*%vn_v
    dW1_1.11=t(vm_v)%*%W2_13
    dW1_1.11=dW1_1.11%*%vn_v
    dW1_1.111=t(vm_v)%*%W2_14
    dW1_1.111=dW1_1.111%*%vn_v
    dW1_1.2=dW2_11.3%*%dW1_1.0+dW2_12.3%*%dW1_1.1+dW2_13.3%*%dW1_1.11+dW2_14.3%*%dW1_1.111
    dW1_1.3=(as.matrix(relud(hidden_layer11)))
    dW1_1.4=dW1_1.3*dW1_1.2
    dW1_1.5=vj_v%*%t(dW1_1.4)
    dW1_1.6=dW1_1.5%*%ai1_1
    dW1_1=t(dW1_1.6)
    
    db1_1=colSums(dW1_1.4)
    
    
    
    dW1_2.0=t(vm_v)%*%W2_21
    dW1_2.0=dW1_2.0%*%vn_v
    dW1_2.1=t(vm_v)%*%W2_22
    dW1_2.1=dW1_2.1%*%vn_v
    dW1_2.11=t(vm_v)%*%W2_23
    dW1_2.11=dW1_2.11%*%vn_v
    dW1_2.111=t(vm_v)%*%W2_24
    dW1_2.111=dW1_2.111%*%vn_v
    dW1_2.2=dW2_11.3%*%dW1_2.0+dW2_12.3%*%dW1_2.1+dW2_13.3%*%dW1_2.11+dW2_14.3%*%dW1_2.111
    dW1_2.3=(as.matrix(relud(hidden_layer12)))
    dW1_2.4=dW1_2.3*dW1_2.2
    dW1_2.5=vj_v%*%t(dW1_2.4)
    dW1_2.6=dW1_2.5%*%ai1_1
    dW1_2=t(dW1_2.6)
    
    db1_2=colSums(dW1_2.4)
    
    
    
    dW1_3.0=t(vm_v)%*%W2_31
    dW1_3.0=dW1_3.0%*%vn_v
    dW1_3.1=t(vm_v)%*%W2_32
    dW1_3.1=dW1_3.1%*%vn_v
    dW1_3.11=t(vm_v)%*%W2_33
    dW1_3.11=dW1_3.11%*%vn_v
    dW1_3.111=t(vm_v)%*%W2_34
    dW1_3.111=dW1_3.111%*%vn_v
    dW1_3.2=dW2_11.3%*%dW1_3.0+dW2_12.3%*%dW1_3.1+dW2_13.3%*%dW1_3.11+dW2_14.3%*%dW1_3.111
    dW1_3.3=(as.matrix(relud(hidden_layer13)))
    dW1_3.4=dW1_3.3*dW1_3.2
    dW1_3.5=vj_v%*%t(dW1_3.4)
    dW1_3.6=dW1_3.5%*%ai1_1
    dW1_3=t(dW1_3.6)
    
    db1_3=colSums(dW1_3.4)
    
    
    
    dW1_4.0=t(vm_v)%*%W2_41
    dW1_4.0=dW1_4.0%*%vn_v
    dW1_4.1=t(vm_v)%*%W2_42
    dW1_4.1=dW1_4.1%*%vn_v
    dW1_4.11=t(vm_v)%*%W2_43
    dW1_4.11=dW1_4.11%*%vn_v
    dW1_4.111=t(vm_v)%*%W2_44
    dW1_4.111=dW1_4.111%*%vn_v
    dW1_4.2=dW2_11.3%*%dW1_4.0+dW2_12.3%*%dW1_4.1+dW2_13.3%*%dW1_4.11+dW2_14.3%*%dW1_4.111
    dW1_4.3=(as.matrix(relud(hidden_layer14)))
    dW1_4.4=dW1_4.3*dW1_4.2
    dW1_4.5=vj_v%*%t(dW1_4.4)
    dW1_4.6=dW1_4.5%*%ai1_1
    dW1_4=t(dW1_4.6)
    
    db1_4=colSums(dW1_4.4)
    
    
    
    
    # update parameter
    W1_1 <- W1_1-step_size*dW1_1
    W1_2 <- W1_2-step_size*dW1_2
    b1_1 <- b1_1[1,]-step_size*db1_1
    b1_1=matrix(rep(b1_1,N),N,S, byrow = T)
    b1_2 <- b1_2[1,]-step_size*db1_2
    b1_2=matrix(rep(b1_2,N),N,S, byrow = T)
    
    W1_3 <- W1_3-step_size*dW1_3
    W1_4 <- W1_4-step_size*dW1_4
    b1_3 <- b1_3[1,]-step_size*db1_3
    b1_3=matrix(rep(b1_3,N),N,S, byrow = T)
    b1_4 <- b1_4[1,]-step_size*db1_4
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
    
    b2_1 <- b2_1[1,]-step_size*db2_1
    b2_1=matrix(rep(b2_1,N),N,S, byrow = T)
    b2_2 <- b2_2[1,]-step_size*db2_2
    b2_2=matrix(rep(b2_2,N),N,S, byrow = T)
    
    b2_3 <- b2_3[1,]-step_size*db2_3
    b2_3=matrix(rep(b2_3,N),N,S, byrow = T)
    b2_4 <- b2_4[1,]-step_size*db2_4
    b2_4=matrix(rep(b2_4,N),N,S, byrow = T)
    
    
    W3_1 <- W3_1-step_size*dW3_1
    W3_2 <- W3_2-step_size*dW3_2    
    W3_3 <- W3_3-step_size*dW3_3
    W3_4 <- W3_4-step_size*dW3_4
    b3_1 <- b3_1[1,]-step_size*db3_1
    b3_1=matrix(rep(b3_1,N),N,MY, byrow = T)
    
    
    
    
    
  }
  return(list(W1_1,W1_2,W1_3,W1_4,b1_1,b1_2,b1_3,b1_4,
              W2_11,W2_12,W2_13,W2_14,W2_21,W2_22,W2_23,W2_24,W2_31,W2_32,W2_33,W2_34,W2_41,W2_42,W2_43,W2_44,b2_1,b2_2,b2_3,b2_4,
              W3_1,W3_2,W3_3,W3_4,b3_1,sqrt(loss)))
}


fbnn.pred4.4 <- function(X, S, vi, vj, vm,vn,vk1,vk2, para = list()){
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
  
  
  
  
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  MY=ncol(y)
  
  
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  
  
  vm1 <- seq(0, 1, by=1/(S-1))
  vm_v <- t(as.matrix(bSpline(vm1,df=vm)))
  
  vn1 <- seq(0, 1, by=1/(S-1))
  vn_v <- t(as.matrix(bSpline(vn1,df=vn)))
  
  vks <- seq(0, 1, by=1/(S-1))
  vks_v <- t(as.matrix(bSpline(vks,df=vk1)))
  
  vky <- seq(0, 1, by=1/(MY-1))
  vky_v <- t(as.matrix(bSpline(vky,df=vk2)))
  
  
  
  
  
  ai1_1=t(as.matrix(vi_v)%*%t(xt)/M)
  
  
  awv1_1.1=ai1_1%*%W1_1
  awv1_1=awv1_1.1%*%vj_v
  
  hidden_layer11 <- relu(b1_1+awv1_1)
  hidden_layer11<- matrix(hidden_layer11, nrow  = N)
  
  
  
  awv1_2.1=ai1_1%*%W1_2
  awv1_2=awv1_2.1%*%vj_v
  
  hidden_layer12 <- relu(b1_2+awv1_2)
  hidden_layer12<- matrix(hidden_layer12, nrow  = N)
  
  awv1_3.1=ai1_1%*%W1_3
  awv1_3=awv1_3.1%*%vj_v
  
  hidden_layer13 <- relu(b1_3+awv1_3)
  hidden_layer13<- matrix(hidden_layer13, nrow  = N)
  
  awv1_4.1=ai1_1%*%W1_4
  awv1_4=awv1_4.1%*%vj_v
  
  hidden_layer14 <- relu(b1_4+awv1_4)
  hidden_layer14<- matrix(hidden_layer14, nrow  = N)
  
  
  #layer 2
  ai2_11=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_11.1=ai2_11%*%W2_11
  awv2_11=awv2_11.1%*%vn_v
  
  ai2_21=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_21.1=ai2_21%*%W2_21
  awv2_21=awv2_21.1%*%vn_v
  
  ai2_31=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_31.1=ai2_31%*%W2_31
  awv2_31=awv2_31.1%*%vn_v
  
  ai2_41=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_41.1=ai2_41%*%W2_41
  awv2_41=awv2_41.1%*%vn_v
  
  
  hidden_layer21 <- relu(b2_1+awv2_11+awv2_21+awv2_31+awv2_41)
  hidden_layer21<- matrix(hidden_layer21, nrow  = N)
  
  
  
  ai2_12=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_12.1=ai2_12%*%W2_12
  awv2_12=awv2_12.1%*%vn_v
  
  ai2_22=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_22.1=ai2_22%*%W2_22
  awv2_22=awv2_22.1%*%vn_v
  
  ai2_32=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_32.1=ai2_32%*%W2_32
  awv2_32=awv2_32.1%*%vn_v
  
  ai2_42=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_42.1=ai2_42%*%W2_42
  awv2_42=awv2_42.1%*%vn_v
  
  
  hidden_layer22 <- relu(b2_2+awv2_12+awv2_22+awv2_32+awv2_42)
  hidden_layer22<- matrix(hidden_layer22, nrow  = N)
  
  
  
  ai2_13=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_13.1=ai2_13%*%W2_13
  awv2_13=awv2_13.1%*%vn_v
  
  ai2_23=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_23.1=ai2_23%*%W2_23
  awv2_23=awv2_23.1%*%vn_v
  
  ai2_33=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_33.1=ai2_33%*%W2_33
  awv2_33=awv2_33.1%*%vn_v
  
  ai2_43=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_43.1=ai2_43%*%W2_43
  awv2_43=awv2_43.1%*%vn_v
  
  
  hidden_layer23 <- relu(b2_3+awv2_13+awv2_23+awv2_33+awv2_43)
  hidden_layer23<- matrix(hidden_layer23, nrow  = N)
  
  
  ai2_14=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_14.1=ai2_14%*%W2_14
  awv2_14=awv2_14.1%*%vn_v
  
  ai2_24=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_24.1=ai2_24%*%W2_24
  awv2_24=awv2_24.1%*%vn_v
  
  ai2_34=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_34.1=ai2_34%*%W2_34
  awv2_34=awv2_34.1%*%vn_v
  
  ai2_44=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_44.1=ai2_44%*%W2_44
  awv2_44=awv2_44.1%*%vn_v
  
  
  hidden_layer24 <- relu(b2_4+awv2_14+awv2_24+awv2_34+awv2_44)
  hidden_layer24<- matrix(hidden_layer24, nrow  = N)
  
  #last layer
  ai3_1=t(as.matrix(vks_v)%*%t(hidden_layer21)/S)
  
  awv3_1.1=ai3_1%*%W3_1
  awv3_1=awv3_1.1%*%vky_v
  
  
  ai3_2=t(as.matrix(vks_v)%*%t(hidden_layer22)/S)
  
  awv3_2.1=ai3_2%*%W3_2
  awv3_2=awv3_2.1%*%vky_v
  
  
  ai3_3=t(as.matrix(vks_v)%*%t(hidden_layer23)/S)
  
  awv3_3.1=ai3_3%*%W3_3
  awv3_3=awv3_3.1%*%vky_v
  
  
  ai3_4=t(as.matrix(vks_v)%*%t(hidden_layer24)/S)
  
  awv3_4.1=ai3_4%*%W3_4
  awv3_4=awv3_4.1%*%vky_v
  
  
  
  hidden_layer3 <- (b3_1+awv3_1+awv3_2+awv3_3+awv3_4)
  hidden_layer3<- matrix(hidden_layer3, nrow  = N)
  
  
  
  # compute the loss: sofmax and regularization
  yhat=hidden_layer3
  
  
  
  return(yhat)  
}



fnnit=function(y,xt,yp,xtp,S, vi, vj, vm,vn,vk1,vk2,step_size,niteration){
  
  Y=y
  X=xt
  N <- nrow(xt) # number of examples
  n=N
  fbnn.model <- fbnn4.4(X=xt, Y=Y, S=S, vi=vi, vj=vj, vm=vm,vn=vn,vk1=vk1,vk2=vk2, step_size=step_size, niteration =niteration)
  
  
  fbnn.pred.model <- fbnn.pred4.4(X=xtp, S=S, vi=vi, vj=vj, vm=vm,vn=vn,vk1=vk1,vk2=vk2, fbnn.model)
  t4p_i2=rmse(as.numeric(yp),as.numeric(fbnn.pred.model))
  
  fbnn.pred.model <- fbnn.pred4.4(X=xt, S=S,vi=vi, vj=vj, vm=vm,vn=vn,vk1=vk1,vk2=vk2, fbnn.model)
  t4_i2=rmse(as.numeric(Y),as.numeric(fbnn.pred.model))
  
  
  
  
  rr=as.matrix(c(t4_i2,t4p_i2))
  
  return(rr)
  
}



fnnrun=frun
finalresm=matrix(c(1,2),nrow=2)
for (i in 1:fnnrun){
  runs=fnnit(y=fdata[[i]][[1]],xt=fdata[[i]][[2]],yp=fdata[[i]][[3]],xtp=fdata[[i]][[4]],S=30,
             vi=5, vj=6, vm=7,vn=8,vk1=9,vk2=10,step_size=.01,niteration=10000)
  finalresm=cbind(finalresm,runs)
  
}

finalresm=finalresm[,-1]

kk1=rowMeans(finalresm)

kk1
