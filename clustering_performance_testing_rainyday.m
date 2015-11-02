%% Code by Gaurav Kumar Singh
% This code plays the next part of the evaluation of clustering's
% performance. In this code we train the model on regular day data and test
% the model on the rainy data. 
% This code demonstrates how to perform clustering on driving data and
% generate useful predictions. The data used for training is contained in
% the file "driver_1_trip_1_regular.csv". There are 13 columns in the csv
% file. They are
% 1 Driver
% 2 Trip	
% 3 Time	
% 4 Ax	
% 5 Ay
% 6 Az	
% 7 PitchRate	
% 8 RollRate	
% 9 YawRate	
% 10 Pitch	
% 11 Roll
% 12 Yaw
% 13 ImuTime
clear all
close all
clc
%% Read Data
data=csvread('driver_1_trip_1_regular.csv',1);
%% Put variables of interest in column vectors
% Longitudinal Acceleration
ax=data(:,4);
% Lateral Acceleration
ay=data(:,5);
% Yaw Rate
yaw=data(:,12)*(pi/180);
%% Get Test Data
test_data=csvread('driver_1_trip_9_rainy.csv',1);
% Longitudinal Acceleration
ax_test=test_data(:,4);
% Lateral Acceleration
ay_test=test_data(:,5);
% Yaw Rate
yaw_test=test_data(:,12)*(pi/180);


%% Clustering on yaw and predicting on yaw
% It is possible to cluster on ax and ay and predict on yaw. BUt, for
% demonstration purposes we a will cluster only on yaw and then predict on
% yaw. What exactly we are doing is that we take 10 points of yaw at a time
% and then see what next 5 points are doing. For example we have 1:10,
% 2:11, 3:12 etc as 10 dimensional vectors. We stack them upon each other
% to cluster. For example if 1:10 and 5:14 belong in same clusters then we
% assume that points 11:15 and 15:19 belong together.
% Number of clusters
num=20;
%Prediction Horizon
horizon=5;
% Taking how many points for clustering
windowsize=10;
% CreateRollingWindow function is useful for stacking vectors of windowsize
% together.
X=createRollingWindow(yaw,windowsize);
% Time Vector
time=data(:,13);
% Setting Clustering Parameters
opts = statset('Display','final');
%% Performing Clustering
% idx is the cluster assignment, for example if set 1 is in cluster 4, then
% idx for that set is 4.
% C is centroids of the cluster. There will be num (number of clusters) centroids
[idx,C] = kmeans(X,num,'Distance','cityblock',...
    'Replicates',10,'Options',opts);

%% Probability distribution On every time instant calculation
% Using ecdf functionality of MATLAB, F is a vector of values of the cdf
% calculated at points metioned in x. Remember Cumulative Distribution
% function is num X 5 cell with each cell containing cumulative
% distribution function over the total number of points belonging to that
% cluster.
F={};
x={};
for i=1:num
    P=[];
    for j=1:length(find(idx==i))
        c1_coord=find(idx==i);
        P(j,:)=yaw(c1_coord(j)+1:c1_coord(j)+horizon);
    end
    size(P)
for p=1:horizon
     [F{i,p},x{i,p}]=ecdf(P(:,p));
end
end


%% Validation part
% Create Rollingwindow for validation part
X_test=createRollingWindow(yaw_test,windowsize);
time=data(:,13);
% Setting clusteing parameters
opts = statset('Display','final');
% idx_test is the cluster assignment, for example if set 1 is in cluster 4, then
% idx_test for that set is 4.
% C_test is centroids of the cluster. There will be num (number of clusters) centroids
[idx_test,C_test] = kmeans(X_test,num,'Distance','cityblock',...
    'Replicates',10,'Options',opts);
%% Comparison of clusters
% We divided trainign and test data into 100 clusters. But clustering is
% random, and we dont know what cluster in test data closely matches to
% cluster in train data. We can cmopare centroids for that. The centroids
% which are closest will be similar. FInd each centroids edistance with
% other centroids and find the minimum distances from 100 X 100 matrices.
for i=1:num
    for j=1:num
        compliance(i,j)=norm(C(i,:)-C_test(j,:));
    end
end
[x_min,id]=min(compliance)

%% Performance Evaluation
% Comparing whether for each of the next 5 samples the prediction remains
% within the range. M_test is a logical of 5 columns.
M_test=[];
M_Z=[];
for i=1:num
    P=[];
    Z=[];
    for j=1:length(find(idx_test==i))
        c1_coord=find(idx_test==i);
        Z(j,:)=yaw_test(c1_coord(j)+1:c1_coord(j)+horizon);
        % P will be a matrix of logicals. 1 for within the range and 0 for
        % out of the range. 
        P(j,:)=[(yaw_test(c1_coord(j)+1)<max(x{id(i),1})) & (yaw_test(c1_coord(j)+1)>min(x{id(i),1})), ...
            (yaw_test(c1_coord(j)+2)<max(x{id(i),2})) & (yaw_test(c1_coord(j)+2)>min(x{id(i),2})),...
            (yaw_test(c1_coord(j)+3)<max(x{id(i),3})) & (yaw_test(c1_coord(j)+3)>min(x{id(i),3})), ...
            (yaw_test(c1_coord(j)+4)<max(x{id(i),4})) & (yaw_test(c1_coord(j)+4)>min(x{id(i),4})),...
            (yaw_test(c1_coord(j)+5)>min(x{id(i),5})) & (yaw_test(c1_coord(j)+5)>min(x{id(i),5}))];
        
    end
    % Computing Recall Function. Recall Function demonstrates that how wide
    % the predictions are. 
    % 6.26 is the range of yaw_rate by the driver (max - min)
    for r=1:horizon
    N(i,r)=((max(x{id(i),r})-min(x{id(i),r}))/(6.26))*((length(find(idx_test==i)))/(length(yaw_test)));
    end
    M_test=[M_test;P];
    M_Z=[M_Z;Z];
end
%% Computing accuracy
% If throughout the row, M_test is 1 (True), this 
% implies that the prediction is good enough for
% next 5 samples. So, we assign a True (1) only when all the columns are
% True in this row. The total percentage of such 'True' will give us the
% accuracy.

for m=1:length(M_test)
    ResMat(m)=all(M_test(m,:)>0);
end
success=sum(ResMat)/length(M_test)
recall=1-sum(sum(abs(N)))
