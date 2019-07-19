
# Question3
### Curve fitting ###

#Data Generation
getwd()
setwd('D:/Georgian tech/Courses/ISYE8803/HW1')
data=read.csv('P04.csv', stringsAsFactors = FALSE, header=FALSE)
X = seq(0,0.68,0.01)
Y = data$V2
plot(X,Y)
CV=rep(0,10)
#Buliding up the LOOCV model
for (n in 6:15){
mse=0
for( i in 1:length(X)){
Y_train=Y[-i]
Y_test=Y[i]
X_train=X[-i]
X_test=X[i]

#Define knots and basis
#training
# n=6
k = seq(min(X),max(X),length.out = n+2)
k=k[2:(n+1)]
h1 = rep(1,length(X_train))
h2 = X_train
h3 = X_train^2
h4 = X_train^3
H=cbind(h1,h2,h3,h4)
for (i in 5:(4+n)){
  A=list()
  print(k[i-4])
  A=(X_train-k[i-4])^3
  A[A <= 0] = 0
  H=cbind(H,A)
  assign(paste0('h',i),A)
}


#Least square estimates
B=solve(t(H)%*%H)%*%t(H)%*%Y_train # refer to formula on slides No.16

#Validation
h1_test = rep(1,length(X_test))
h2_test = X_test
h3_test = X_test^2
h4_test = X_test^3
H_test=cbind(h1_test,h2_test,h3_test,h4_test)
for (i in 5:(4+n)){
  VT=list()
  VT=(X_test-k[i-4])^3
  VT[VT <= 0] = 0
  H_test=cbind(H_test,VT)
  assign(paste0('h_test',i),VT)
}
# calculate mse
mse=mse+as.numeric((Y_test-H_test%*%B))^2

}
CV[n-5]=mse
}

# B spline fitting
# Generate data:

CV=rep(0,10)
for (n in 6:15){
mse=0
for( i in 1:length(X)){
Y_train=Y[-i]
Y_test=Y[i]
X_train=X[-i]
X_test=X[i]
# Generate B-spline basis:
knots = seq(0,0.68,length.out =n+2) # create knots within the same range of X 
B_train = bs(X_train, knots = knots, degree = 3,intercept = FALSE)[,1:(n+4)]
B_test = bs(X, knots = knots, degree = 3,intercept = FALSE)[i,1:(n+4)]
# Least square estimation
yhat = B_test%*%solve(t(B_train)%*%B_train)%*%t(B_train)%*%Y_train  # same way of making the prediction as curve fitting. Only different way of creating the matrix
mse=mse+(yhat-Y_test)^2

}
CV[n-5]=mse
}


#smoothing spline
# Different lambda selection
k=40
allspar = seq(0,1,length.out = 1000)
p = length(allspar)
RSS = rep(0,p)
df = rep(0,p)
for(i in 1:p)  # using a for loop to test the change of GCV value
{
  yhat = smooth.spline(Y_train, df = k+1, spar = allspar[i])
  df[i] = yhat$df
  yhat = yhat$y
  RSS[i] = sum((yhat-Y_train)^2)
}
# GCV criterion
GCV = (RSS/n)/((1-df/n)^2)
plot(allspar,GCV,type = "l", lwd = 3)
spar = allspar[which.min(GCV)]
points(spar,GCV[which.min(GCV)],col = "red",lwd=5)
yhat = smooth.spline(Y_train, df = k+1, spar = spar)
yhat = yhat$y
plot(X_train,Y_train,col = "red",lwd=3)
lines(X_train,Y_train,col = "black",lwd=3)

### RBF Kernel ###

# Data Genereation 
getwd()
setwd('D:/Georgian tech/Courses/ISYE8803/HW1')
data=read.csv('P04.csv', stringsAsFactors = FALSE, header=FALSE)
x = seq(0,0.68,0.01)
y = data$V2 
kerf = function(z){exp(-z*z/2)/sqrt(2*pi)} # the Gaussian kernel function
# leave-one-out CV
h1=seq(0.01,0.2,0.005)
er = rep(0, length(y))
mse = rep(0, length(h1))
for(j in 1:length(h1))  # use the for loop to find out the h value that minimize the prediction error
{
  h=h1[j]
  for(i in 1:length(y))
  {
    X1=x;
    Y1=y;
    X1=x[-i];
    Y1=y[-i];
    z=kerf((x[i]-X1)/h)
    yke=sum(z*Y1)/sum(z) # critical step to calculate the value of yke
    er[i]=y[i]-yke
  }
  mse[j]=sum(er^2) # calculate the error 
}
plot(h1,mse,type = "l")
h = h1[which.min(mse)]
points(h,mse[which.min(mse)],col = "red", lwd=5)


### Slide 8 ###

# Interpolation for N values
N=1000
xall = seq(min(x),max(x),length.out = N)
f = rep(0,N);
for(k in 1:N)
{
  z=kerf((xall[k]-x)/h)
  f[k]=sum(z*y)/sum(z);
}
ytrue = sin(xall/10)+(xall/50)^2
plot(x,y,col = "black")
lines(xall,ytrue,col = "red")
lines(xall, f, col = "blue")


# Question 4


### Functional data classification ###
setwd('D:/Georgian tech/Courses/ISYE8803/HW1')
library(randomForest)
# Data generation
set.seed(123)
data1=read.table("ECG200TRAIN",sep=',',header=TRUE)
data2=read.table("ECG200TEST",sep=',',header=TRUE)
# Train and test data sets

# Option 1: B-splines
library(splines)
X=data1[,2:97]
Xt=data2[,2:97]
x=seq(0,0.95,0.01)
knots = seq(0,0.95,length.out =8)
B = bs(x, knots = knots, degree = 3)[,1:10]
Bcoef = matrix(0,dim(X)[1],10)
Bcoef_t = matrix(0,dim(X)[1],10)
for(i in 1:dim(X)[1])
{
  Bcoef[i,] = solve(t(B)%*%B)%*%t(B)%*%t(as.matrix(X[i,]))
  Bcoef_t[i,] = solve(t(B)%*%B)%*%t(B)%*%t(as.matrix(Xt[i,]))
}

data=cbind.data.frame(Bcoef,data1[,1])
names(data)=c( 'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','y')

data$y=as.factor(data$y)
fit = randomForest(y~.,data)

pred = predict(fit,Bcoef_t)
table(data2[,1],pred)
Xtest = Xt
matplot(x,t(Xtest[pred==-1,]),type="l",col = "blue",ylab = "y",main="Classification using B-spline coefficients")
X2 = Xtest[pred == 1,]
for(i in 1:length(pred[pred==1]))
{
  lines(x,X2[i,],col = "red")
}


# Option 2: Functional principal components
library(fda)
X=data1[,2:97]
Xt=data2[,2:97]
x=seq(0,1,length=96)



splinebasis = create.bspline.basis(c(0,1),20)
smooth = smooth.basis(x,t(X),splinebasis)
smooth_t= smooth.basis(x,t(Xt),splinebasis)
Xfun = smooth$fd
Xfun_t=smooth_t$fd
pca = pca.fd(Xfun, 20)
pca_t=pca.fd(Xfun_t, 20)
var.pca = cumsum(pca$varprop)

nharm = sum(var.pca < 0.95) + 1
pc = pca.fd(Xfun, nharm)
pc_t = pca.fd(Xfun_t, nharm)
plot(pc$scores[data1$X.1==-1,],xlab = "FPC-score 1", ylab = "FPC-score 2",col = "blue")
points(pc$scores[data1$X.1==1,],col = "red")
FPCcoef = pc$scores
FPCcoef_t=pc_t$scores
data=cbind.data.frame(as.data.frame(FPCcoef),data1$X.1)
names(data)[9]='y'

data$y=as.factor(data$y)

fit = randomForest(y ~ .,data)
pred = predict(fit,FPCcoef_t)
table(data2$X1,pred)
Xtest = Xt
matplot(x,t(Xtest[pred==-1,]),type="l",col = "blue",ylab = "y",main="Classification using FPCA scores")
X2 = Xtest[pred == 1,]
for(i in 1:length(pred[pred==1]))
  
{
  lines(x,X2[i,],col = "red")
}
