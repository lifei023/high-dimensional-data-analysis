% % EXAMPLES OF IMAGE ANALYSIS

%%%%%%%%%%%%%%%%%% IMAGE TRANSFORMATION %%%%%%%%%%%%%%%%%%

%%% Read and show image %%%
I = imread('horse1.jpg');
figure
imshow(I)

%%% Resize the image %%%
J =imresize (I,0.33)
figure, imshow (J);

%%% Convert to gray image %%%
I2= rgb2gray(I);
figure
imshow (I2)

%%% Convert to black and white image %%%
BW= im2bw(I,0.5)
imshow (BW)

%%% Image histogram %%%
N = 256;
imhist(I2,N)

%%% Gray level transformation  %%%
%%% linear  %%%
figure
imshow(uint8(256-1-double(I2)))
%%% log transformation with c=20  %%%
figure
imshow(uint8(20*log(double(I2)+1)))
%%% log transformation with c=40  %%%
figure
imshow(uint8(40*log(double(I2)+1)))
%%% transformation with threshold 100 %%%
figure
imshow(I2>100)
%%% transformation with threshold 150 %%%
figure
imshow(I2>150)


%%% Histogram shift %%%
figure
imshow(I2+50)
imhist(I2+50,N)
I3=ones(563,1000)
for c = 1:563
    for r = 1:1000
        if (I2(c,r)+50) < 25
            I3(c,r) = 25;
        elseif (I2(c,r)+50) > 225
            I3(c,r) = 225;
        else
            I3(c,r)=(I2(c,r)+50);
        end
        
    end
end
imshow(uint8(I3))
imhist(uint8(I3),N)

%%% Histogram stretching %%%
%%%the initial contrast%%%
Cons_ini=max(max(I2))-min(min(I2))
D=double(I2);
I4=uint8((D-min(min(D)))/double(Cons_ini)*200)
imshow(I4)
Cons_new=max(max(I4))-min(min(I4))
imhist(I4,N)

%%% image denoising  %%%
K = [ 1 1 1;1 1 1;1 1 1]/9
I5 = uint8(double(I2)+normrnd(0,10,563,1000));
I6 = imfilter(I5,K);
figure
imshow(I5)
figure
imshow(I6)

%%% image sharpening  %%%
K2 = [ -1 -1 -1;-1 9 -1;-1 -1 -1]
I7 = imfilter(I,K2);
figure
imshow(I7)

%%%%%%%%%%%%%%%%%% IMAGE SEGMENTATION %%%%%%%%%%%%%%%%%%

%%% Otsu's Method %%%
%%one level%%
I8 = I2;
imshow(I8)
level = multithresh(I8);
seg_I = imquantize(I8,level);
figure
imshow(seg_I,[])

%%two levels%%
level = multithresh(I8,2);
seg_I = imquantize(I8,level);
figure
imshow(seg_I,[])

%%three levels%%
level = multithresh(I8,3);
seg_I = imquantize(I8,level);
figure
imshow(seg_I,[])

%%four levels%%
level = multithresh(I8,4);
seg_I = imquantize(I8,level);
figure
imshow(seg_I,[])

%%five levels%%
level = multithresh(I8,5);
seg_I = imquantize(I8,level);
figure
imshow(seg_I,[])

%%% k-means clustering %%%
% input image
X=reshape(I,size(I,1)*size(I,2),size(I,3));
% segmentation with different K values
K=[2 3 4 5];
for i = 1:4
[L,Centers] = kmeans(double(X),K(i));
Y = reshape(L,size(I,1),size(I,2)); 
B = labeloverlay(I,Y);
subplot (2,2,i);
imshow(B) 
end

%%%%%%%%%%%%%%%%%% EDGE DETECTION %%%%%%%%%%%%%%%%%%

BW2 = edge(double(rgb2gray(I)),'Prewitt',2)
figure;imagesc(BW2)
BW2 = edge(double(rgb2gray(I)),'Prewitt',10)
figure;imagesc(BW2)
BW2 = edge(double(rgb2gray(I)),'Prewitt',40)
figure;imagesc(BW2)


BW3 = edge(double(rgb2gray(I)),'Sobel',2)
figure;imagesc(BW3)
BW3 = edge(double(rgb2gray(I)),'Sobel',10)
figure;imagesc(BW3)
BW3 = edge(double(rgb2gray(I)),'Sobel',40)
figure;imagesc(BW3)


%%% Q4 %%%
%a%
I=imread('horse1.jpg')
% figure
% imshow(I)
I2= rgb2gray(I);
k=[1 4 7 10 7 4 1; 4 12 26 33 26 12 4;7 26 55 71 55 26 7;
    10 33 71 91 71 33 10; 7 26 55 71 55 26 7; 
    4 12 26 33 26 12 4; 1 4 7 10 7 4 1]/1115
s = imfilter(I2,k,'replicate');
imshow(s)

%b%
[w,h]=size(s)
dx=zeros(w,h);
dy=zeros(w,h);
Gx=zeros(w,h);
theta=zeros(w,h);

% calculate dx,dy%
for i = 1:(w-1)
    for j = 1:(h-1)      
         dx(i,j)=(s(i+1,j)-s(i,j)+s(i+1,j+1)-s(i,j+1))/2;
         dy(i,j)=(s(i,j+1)-s(i,j)+s(i+1,j+1)-s(i+1,j))/2;
         Gx(i,j)=sqrt(dx(i,j).^2+dy(i,j).^2);
         theta(i,j)=atan(dx(i,j)/dy(i,j));
    end
end

theta(isnan(theta))=0;

%part c%
A=0;
B=0;
Phi=Gx
for i = 2:(w-1)
    for j = 2:(h-1)
        
if theta(i,j)>-pi/8 & theta(i,j)<=pi/8 
   theta(i,j)=0;
   A=Gx(i,j-1);
   B=Gx(i,j+1);
elseif theta(i,j)>pi/8 & theta(i,j)<=pi*3/8 
   theta(i,j)=pi/4;
   A=Gx(i+1,j-1);
   B=Gx(i-1,j+1);
elseif theta(i,j)>-3*pi/8 & theta(i,j)<=-pi/8 
   theta(i,j)=-pi/4;
   A=Gx(i-1,j-1);
   B=Gx(i+1,j+1);
elseif theta(i,j)>3*pi/8 & theta(i,j)<=pi/2 | theta(i,j)>-pi/2 & theta(i,j)<=-3*pi/8  
   theta(i,j)=pi/2;
   A=Gx(i-1,j);
   B=Gx(i+1,j);
else
 
    
end

if Gx(i,j)>=A & Gx(i,j)<=B
    Phi(i,j)=Gx(i,j);
else
    Phi(i,j)=0;
end
end
end

%part d%
count=5
TL=1
TU=10
E=zeros(563,1000)
while count~=0
    count=0;
    for i = 2:(w-1)
        for j = 2:(h-1)
            if Phi(i,j)>TU & E(i,j)==0
                E(i,j)=1;
                count=count+1;
                
            elseif Phi(i,j)>=TL &E(i,j)==0
                for k = i-1:i+1
                    for l=j-1:j+1
                        if E(k,l)==1;
                            E(i,j)=1;
                            count=count+1;
                   
                        end
                    end
                end
            end
        end
    end
end
figure
imshow(E)
figure
BW=edge(I2,'Canny')
imshow(BW)