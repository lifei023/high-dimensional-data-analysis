
##Question 2
# Library
#install.packages("glmnet")

library(glmnet)
# load data
set.seed(7)
data=read.csv('./HW4/concrete.csv',stringsAsFactors = FALSE,header = TRUE)
head(data)
#split the data into training and test sets
s=sample.int(1030,0.8*nrow(data))
train=data[s,]
test=data[-s,]

################################lasso############################ 
# build Lasso model on the training set
X=data.matrix(train[,1:8])
y=data.matrix(train[,9])
lasso = cv.glmnet(X, y, family = "gaussian", alpha = 1, intercept = TRUE,
                  nfolds = 5,type.measure="mse")
lambda_l = lasso$lambda.min
coef.lasso = matrix(round(coef(lasso, s = lambda_l),3))[2:9]
lasso = glmnet(X, y, family = "gaussian", alpha = 1, intercept = TRUE)
plot(lasso, xvar = "lambda", label = TRUE)
abline(v = log(lambda_l))

#test the lasso model on the test set
X_test=data.matrix(test[,1:8])
y_test=data.matrix(test[,9])
y_lasso = predict(lasso, X_test, s = lambda_l)
mse_lasso = sum((y_test-y_lasso)^2)/nrow(test)

################################ridge############################# 
# build ridge model on the training set
ridge = cv.glmnet(X, y, family = "gaussian", alpha = 0, intercept = TRUE,
                  nfolds = 5,type.measure="mse")
lambda_r = ridge$lambda.min

coef.ridge = matrix(round(coef(ridge, s = lambda_r),3))[2:9]
ridge = glmnet(X, y, family = "gaussian", alpha = 0, intercept = TRUE)
plot(ridge, xvar = "lambda", label = TRUE)
abline(v = log(lambda_r))

#test the ridge model on the test set
X_test=data.matrix(test[,1:8])
y_test=data.matrix(test[,9])
y_ridge = predict(ridge, X_test, s = lambda_r)
mse_ridge = sum((y_test-y_ridge)^2)/nrow(test)


################################elastic net############################# 
#find the optimal alpha value
R2=c()

for (i in 0:10) {
  mod_elastic = cv.glmnet(X,y,alpha=i/10,nfolds = 5,type.measure="mse",family="gaussian")
  R2 = cbind(R2,mod_elastic$glmnet.fit$dev.ratio[which(mod_elastic$glmnet.fit$lambda == mod_elastic$lambda.min)])
  
}

alpha_best = (which.max(R2)-1)/10
#find the optimal lambda value for elastic net
E_net=cv.glmnet(X,y,alpha=alpha_best,
                      nfolds = 5,type.measure="mse",family="gaussian")
lambda_e =E_net $lambda.min
coef.enet = matrix(round(coef(E_net, s = lambda_e),3))[2:9]
E_net = glmnet(X, y, family = "gaussian", alpha = alpha_best, intercept = TRUE)
plot(E_net, xvar = "lambda", label = TRUE)
abline(v = log(lambda_e))

#test the elastic net model on the test set
X_test=data.matrix(test[,1:8])
y_test=data.matrix(test[,9])
y_enet = predict(E_net, X_test, s = lambda_e)
mse_enet = sum((y_test-y_enet)^2)/nrow(test)



################################Adaptive lasso############################# 
gamma = 2
b.ols = round(solve(t(X)%*%X)%*%t(X)%*%y,3)
ridge = cv.glmnet(X, y, family = "gaussian", alpha = 0, intercept = TRUE)
l.ridge = ridge$lambda.min
b.ridge = matrix(round(coef(ridge, s = l.ridge),3))[2:9]
w1 = 1/abs(b.ols)^gamma
w2 = 1/abs(b.ridge)^gamma
alasso1 = cv.glmnet(X, y, family = "gaussian", alpha = 1, intercept = TRUE, penalty.factor = w1)
alasso2 = cv.glmnet(X, y, family = "gaussian", alpha = 1, intercept = TRUE, penalty.factor = w2)
lambda1 = alasso1$lambda.min
lambda2 = alasso2$lambda.min
coef.alasso1 = matrix(round(coef(alasso1, s = lambda1),3))[2:9]
coef.alasso2 = matrix(round(coef(alasso2, s = lambda2),3))[2:9]
alasso1 = glmnet(X, y, family = "gaussian", alpha = 1, intercept = TRUE, penalty.factor = w1)
alasso2 = glmnet(X, y, family = "gaussian", alpha = 1, intercept = TRUE, penalty.factor = w2)
plot(alasso1, xvar = "lambda", label = TRUE)
abline(v=log(lambda1))
plot(alasso2, xvar = "lambda", label = TRUE)
abline(v=log(lambda2))
View(cbind.data.frame(b.ols, b.ridge, coef.lasso, coef.alasso1, coef.alasso2,coef.enet))
y_alasso1 = predict(alasso1, X_test, s = lambda1)
mse_alasso1 = sum((y_test-y_alasso1)^2)/nrow(test)
y_alasso2 = predict(alasso2, X_test, s = lambda2)
mse_alasso2 = sum((y_test-y_alasso2)^2)/nrow(test)


##############################problem 3######################
# install.packages("gglasso")
#install.packages("fda")
# install.packages("pracma")
library(fda)
library(pracma)
library(gglasso)

#load data
# install.packages('R.matlab')
library(R.matlab)
data=readMat('./HW4/NSC.mat')
X=data$x
y=data$y
m=dim(y)[1]
n=dim(y)[2]
p=length(X)
#plot the data x
par(mfrow=c(2,5))# set graphic parameters
data1 = list()
for(i in 1:p)
{
  data1[[i]]=as.matrix(X[[i]][[1]])
  matplot(t(data1[[i]]), type = "l", xlab = i,ylab = "")  # matrix plot with type 'line'
}

#plot the data y
dev.off()
plot(y, xlab = "")

# reduce dimension
spb = 10
x = seq(0,1,length=n)
splinebasis_B=create.bspline.basis(c(0,1),spb)
base_B=eval.basis(as.vector(x),splinebasis_B)
P = t(base_B)
X = array(dim=c(m,n,p))
for(i in 1:p)
{
  X[,,i] = data1[[i]]
}
Z = array(dim=c(dim(X)[1],spb,p))
for(i in 1:p)
{
  Z[,,i] = X[,,i]%*%base_B/n 
}
Z = matrix(Z,m,spb*p)
y= y%*%base_B/n

#group lasso
I=diag(10)
Z1=kronecker(I,Z)
y=matrix(y,1500,1)
group = rep(rep(1:p,each=spb),10)
glasso = cv.gglasso(Z1,y,group,loss = "ls")
lambda = glasso$lambda.min
coef = matrix(coef(glasso,s="lambda.1se")[2:1001],spb**2,p)
View(coef)
# coef = base_B%*%t(coef)
x1=seq(1,100,1)
matplot(x1,coef,col=c(5,1,6,2,7,3,8,4,9,10),lty=rep(1,10),type="l",ylim=c(-0.00003,0.00002),lwd=1)
legend(90,0.00002,c(1:10),col=c(5,1,6,2,7,3,8,4,9,10),lty = rep(1,10),lwd=3,cex=0.3)
glasso = gglasso(Z1,y,group,loss = "ls", lambda=lambda)

#prediction on the test set
#import test dataset
data_t=readMat('./HW4/NSC.test.mat')
Xt=data_t$x.test
yt=data_t$y.test
mt=dim(yt)[1]
nt=dim(yt)[2]
pt=length(Xt)
data1_t=list()
for(i in 1:pt)
{
  data1_t[[i]]=as.matrix(Xt[[i]][[1]])
}
#get the B spline coefficient of the sensor data in the test set
X2 = array(dim=c(mt,nt,pt))
for(i in 1:pt)
{
  X2[,,i] = data1_t[[i]]
}
Z2 = array(dim=c(dim(X2)[1],spb,p))
for(i in 1:pt)
{
  Z2[,,i] = X2[,,i]%*%base_B/nt 
}
Z2 = matrix(Z2,mt,spb*pt)
Z2=kronecker(I,Z2)
#make prediction
y_pred=predict(glasso, Z2)
yt= matrix(yt%*%base_B/nt,500,1)
#calculate mse
mse_glasso=sum((yt-y_pred)**2)/nt
