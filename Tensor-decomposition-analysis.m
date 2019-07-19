%%%%%%%%%%%%%%%%%% Q3 %%%%%%%%%%%%%%%%%
%%% CP Decomposition
rng(7)
%matrial 1
X=tensor(T1);
X1 = tenmat(X,1);
err = zeros(10,1);
for i = 1:10
    P = cp_als(X,i);
    P1 = tenmat(P,1);
    dif = tensor(X1-P1);
    err(i) = innerprod(dif,dif);
end
AIC =  2*err + (2*[1:10])';
[min,wmin] = min(AIC);
plot(AIC)
xlabel('R')
ylabel('AIC')
P = cp_als(X,4);
figure
plot(P.U{3})
legend('1','2','3','4')
xlabel('time')
for i = 1:4
    XY = kron(P.U{1}(:,i),P.U{2}(:,i)')*P.lambda(i);
    figure;
    ScaledXY = XY*16;
    image(ScaledXY);
    xlabel('x')
    ylabel('y')
    colormap hot
end

%matrial 2
rng(7)
X=tensor(T2)
X1 = tenmat(X,1);
err = zeros(10,1);
for i = 1:10
    P = cp_als(X,i);
    P1 = tenmat(P,1);
    dif = tensor(X1-P1);
    err(i) = innerprod(dif,dif);
end
AIC =  2*err + (2*[1:10])';
[min,wmin] = min(AIC);
plot(AIC)
xlabel('R')
ylabel('AIC')
P = cp_als(X,4);
figure
plot(P.U{3})
legend('1','2','3','4')
xlabel('time')
for i = 1:4
    XY = kron(P.U{1}(:,i),P.U{2}(:,i)')*P.lambda(i);
    figure;
    ScaledXY = XY*16;
    image(ScaledXY);
    xlabel('x')
    ylabel('y')
    colormap hot
end

%matrial 3
rng(7)
X=tensor(T3)
X1 = tenmat(X,1);
err = zeros(10,1);
for i = 1:10
    P = cp_als(X,i);
    P1 = tenmat(P,1);
    dif = tensor(X1-P1);
    err(i) = innerprod(dif,dif);
end
AIC =  2*err + (2*[1:10])';
[min,wmin] = min(AIC);
plot(AIC)
xlabel('R')
ylabel('AIC')
P = cp_als(X,4);
figure
plot(P.U{3})
legend('1','2','3','4')
xlabel('time')
for i = 1:4
    XY = kron(P.U{1}(:,i),P.U{2}(:,i)')*P.lambda(i);
    figure;
    ScaledXY = XY*16;
    image(ScaledXY);
    xlabel('x')
    ylabel('y')
    colormap hot
end


%%%%%%%%%%%%%%%%%% Q4 Part a %%%%%%%%%%%%%%%%%%%
%load images for training
data=double(imread('./CatsBirds/train28.jpg'));
data_train=tensor(ones(500,500,3,28));
for i =1:28
    name=strcat('./CatsBirds/train',num2str(i),'.jpg');
    data_train(:,:,:,i)=imread(name);
end

%%% Tucker Decomposition
%decomposition
X=data_train
X1 = tenmat(X,1);
err = zeros(6,6,3);
AIC = zeros(6,6,3);
for i = 5:10
    for j = 5:10
       for k=1:3
            T = tucker_als(X,[i,j,k,28]);
            T1 = tenmat(T,1);
            dif = tensor(X1-T1);
            err(i-4,j-4,k) = innerprod(dif,dif);
            AIC(i-4,j-4,k) = 2*err(i-4,j-4,k) + 2*(i+j+k);
       end 
    end    
end
% we find [10,10,3,28] to be the optimal value
T = tucker_als(X,[10,10,3,28]);
test=T.core
DT=ones(28,300)
%convert core tensor into vectors
for i =1:28
    A=test(:,:,:,i);
    V1=tenmat(A,1:3,'t');
    DT(i,:)=V1;
end
Y=ones(28,1)
Y(15:28,1)=0
%build up tree model
fit=TreeBagger(100,DT,Y)

%Decomposition of test dataset
data_test=tensor(ones(500,500,3,12));
for i =1:12
    name=strcat('./CatsBirds/test',num2str(i),'.jpg');
    data_test(:,:,:,i)=imread(name);
end
T2 = tucker_als(data_test,[10,10,3,12]);
test2=T2.core
DT2=ones(12,300)
%convert core tensor into vectors
for i =1:12
    A=test2(:,:,:,i);
    V1=tenmat(A,1:3,'t');
    DT2(i,:)=V1;
end
%make prediction
Y2=ones(12,1)
Y2([1 2 4 6 9 12],1)=0
Yfit=predict(fit,DT2)
result=double(cell2mat(Yfit))-48
%check the prediction result
TN=0
TP=0
FN=0
FP=0
for i =1:12
    
    if Y2(i,1)==result(i,1)& result(i,1)==0
        TN=TN+1
    elseif Y2(i,1)==result(i,1)& result(i,1)==1
        TP=TP+1
    elseif ne(Y2(i,1),result(i,1))& result(i,1)==1
        FP=FP+1
    else
        FN=FN+1
    end
end
   
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1=precision*recall/(precision+recall)
A=categorical(Y2)
B=categorical(result)
plotconfusion(A,B)


%%%%%%%%%%%%%%%%%% Q4 Part b %%%%%%%%%%%%%%%%%%%
rng(7)
rgb2gray(imread('./CatsBirds/train28.jpg'));
data_train=tensor(ones(500,500,28));
for i =1:28
    name=strcat('./CatsBirds/train',num2str(i),'.jpg');
    data_train(:,:,i)=rgb2gray(imread(name));
end

%%% Tucker Decomposition
%decomposition
X=data_train
X1 = tenmat(X,1);
err = zeros(10,10);
AIC = zeros(10,10);
for i = 1:10
    for j = 1:10
            T = tucker_als(X,[i,j,28]);
            T1 = tenmat(T,1);
            dif = tensor(X1-T1);
            err(i,j) = innerprod(dif,dif);
            AIC(i,j) = 2*err(i,j) + 2*(i+j);
   
    end    
end
% we find [10,10,28] to be the optimal value
T = tucker_als(X,[10,10,28]);
test=T.core
DT=ones(28,100)
%convert core tensor into vectors
for i =1:28
    A=test(:,:,i);
    V1=tenmat(A,1:2,'t');
    DT(i,:)=V1;
end
Y=ones(28,1)
Y(15:28,1)=0
%build up tree model
fit=TreeBagger(100,DT,Y)

%Decomposition of test dataset
data_test=tensor(ones(500,500,12));
for i =1:12
    name=strcat('./CatsBirds/test',num2str(i),'.jpg');
    data_test(:,:,i)=rgb2gray(imread(name));
end
T2 = tucker_als(data_test,[10,10,12]);
test2=T2.core
DT2=ones(12,100)
%convert core tensor into vectors
for i =1:12
    A=test2(:,:,i);
    V1=tenmat(A,1:2,'t');
    DT2(i,:)=V1;
end
%make prediction
Y2=ones(12,1)
Y2([1 2 4 6 9 12],1)=0
Yfit=predict(fit,DT2)
result=double(cell2mat(Yfit))-48
%check the prediction result
TN=0
TP=0
FN=0
FP=0
for i =1:12
    
    if Y2(i,1)==result(i,1)& result(i,1)==0
        TN=TN+1
    elseif Y2(i,1)==result(i,1)& result(i,1)==1
        TP=TP+1
    elseif ne(Y2(i,1),result(i,1))& result(i,1)==1
        FP=FP+1
    else
        FN=FN+1
    end
end
   
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1=precision*recall/(precision+recall)
A1=categorical(Y2)
B1=categorical(result)
plotconfusion(A1,B1)
