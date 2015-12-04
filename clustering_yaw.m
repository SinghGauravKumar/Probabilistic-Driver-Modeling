%% Code by Gaurav Kumar Singh
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

clear
close all
clc
load 'vals.mat';
for v=11009:27106
    
    a=vals(v,1);
    b=vals(v,2);
conn=database('tri-IVBSSDB','','');  % here I don't enter the username and password.
ping(conn);

setdbprefs('DataReturnFormat','cellarray');
val=[];
s=sprintf('SELECT ALL * FROM [LvFot].[dbo].[Imu] WHERE Driver=%d AND Trip=%d',a,b);
curs=exec(conn,s);
curs=fetch(curs);
data=curs.Data; % valid data expected in this.
data=cell2mat(data);
if size(data,1)<500
    continue
end
close(curs);
close(conn);
%% Put variables of interest in column vectors
% Longitudinal Acceleration
ax=data(:,4);
% Lateral Acceleration
ay=data(:,5);
% Yaw Rate
yaw=data(:,12)*(pi/180);
yaw_rate=data(:,9)*(pi/180);
% Keeping first 25000 points for training and rest for testing
% To save efforts making new variables, we call training data as ax, ay,
% yaw only, while validation data is called ax_test, ay_test, yaw.
brk=floor(size(data,1)*0.6);
ax_test=ax(brk+1:end);
ay_test=ay(brk+1:end);
yaw_test=yaw(brk+1:end);
yaw_rate_test=yaw_rate(brk+1:end);
ax=ax(1:brk);
ay=ay(1:brk);
yaw=yaw(1:brk);
yaw_rate=yaw_rate(1:brk);
clear data
%% Clustering on yaw and predicting on yaw
% It is possible to cluster on ax and ay and predict on yaw. BUt, for
% demonstration purposes we a will cluster only on yaw and then predict on
% yaw. What exactly we are doing is that we take 10 points of yaw at a time
% and then see what next 5 points are doing. For example we have 1:10,
% 2:11, 3:12 etc as 10 dimensional vectors. We stack them upon each other
% to cluster. For example if 1:10 and 5:14 belong in same clusters then we
% assume that points 11:15 and 15:19 belong together.
% Number of clusters
num=50;
%Prediction Horizon
horizon=5;
% Taking how many points for clustering
windowsize=10;
% CreateRollingWindow function is useful for stacking vectors of windowsize
% together.
X=createRollingWindow(yaw,windowsize);
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
% function is 100 X 5 cell with each cell containing cumulative
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

for i=1:size(X_test,1)
    for j=1:num
        compliance(i,j)=min(norm(C(j,:)-X_test(i,:)));
    end
end
[x_min,id]=min(compliance');

%% Performance Evaluation
% Comparing whether for each of the next 5 samples the prediction remains
% within the range.
M_test=[];
M_Z=[];
for i=1:num
    P=[];
    Z=[];
    for j=1:length(find(id==i))
        c1_coord=find(id==i);
        Z(j,:)=yaw_test(c1_coord(j)+1:c1_coord(j)+horizon);
        % P will be a matrix of logicals. 1 for within the range and 0 for
        % out of the range. 
        P(j,:)=[(yaw_test(c1_coord(j)+1)<max(x{i,1})) & (yaw_test(c1_coord(j)+1)>min(x{i,1})), ...
            (yaw_test(c1_coord(j)+2)<max(x{i,2})) & (yaw_test(c1_coord(j)+2)>min(x{i,2})),...
            (yaw_test(c1_coord(j)+3)<max(x{i,3})) & (yaw_test(c1_coord(j)+3)>min(x{i,3})), ...
            (yaw_test(c1_coord(j)+4)<max(x{i,4})) & (yaw_test(c1_coord(j)+4)>min(x{i,4})),...
            (yaw_test(c1_coord(j)+5)>min(x{i,5})) & (yaw_test(c1_coord(j)+5)>min(x{i,5}))];
        
    end
    % Computing Recall Function. Recall Function demonstrates that how wide
    % the predictions are.
    for r=1:horizon
    N(i,r)=((max(x{i,r})-min(x{i,r}))/(2*pi))*((length(find(id==i))));
    end
    M_test=[M_test;P];
    M_Z=[M_Z;Z];
end
%% Computing accuracy
% From M_test a matrix of logicals for 5 columns. If throughout the row,
% M_test is 1 (True), this implies that the prediction is good enough for
% next 5 samples. So, we assign a True (1) only when all the columns are
% True in this row. The total percentage of such 'True' will give us the
% accuracy.

for m=1:length(M_test)
    ResMat(m)=all(M_test(m,:)>0);
end
success=sum(ResMat)/length(M_test)
recall=1-(sum(sum(abs(N)))/length(yaw_test))
vals(v,8)=success;
vals(v,9)=recall;
clear success
clear recall

%% Clustering on yaw rate and predicting on yaw rate
% It is possible to cluster on ax and ay and predict on yaw rate. BUt, for
% demonstration purposes we a will cluster only on yaw rate and then predict on
% yaw rate. What exactly we are doing is that we take 10 points of yaw rate at a time
% and then see what next 5 points are doing. For example we have 1:10,
% 2:11, 3:12 etc as 10 dimensional vectors. We stack them upon each other
% to cluster. For example if 1:10 and 5:14 belong in same clusters then we
% assume that points 11:15 and 15:19 belong together.
% Number of clusters
num=50;
%Prediction Horizon
horizon=5;
% Taking how many points for clustering
windowsize=10;
% CreateRollingWindow function is useful for stacking vectors of windowsize
% together.
X=createRollingWindow(yaw_rate,windowsize);
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
% function is 100 X 5 cell with each cell containing cumulative
% distribution function over the total number of points belonging to that
% cluster.
F={};
x={};
for i=1:num
    P=[];
    for j=1:length(find(idx==i))
        c1_coord=find(idx==i);
        P(j,:)=yaw_rate(c1_coord(j)+1:c1_coord(j)+horizon);
    end
    size(P)
for p=1:horizon
     [F{i,p},x{i,p}]=ecdf(P(:,p));
end
end


%% Validation part
% Create Rollingwindow for validation part
X_test=createRollingWindow(yaw_rate_test,windowsize);

for i=1:size(X_test,1)
    for j=1:num
        compliance(i,j)=min(norm(C(j,:)-X_test(i,:)));
    end
end
[x_min,id]=min(compliance');

%% Performance Evaluation
% Comparing whether for each of the next 5 samples the prediction remains
% within the range.
M_test=[];
M_Z=[];
for i=1:num
    P=[];
    Z=[];
    for j=1:length(find(id==i))
        c1_coord=find(id==i);
        Z(j,:)=yaw_rate_test(c1_coord(j)+1:c1_coord(j)+horizon);
        % P will be a matrix of logicals. 1 for within the range and 0 for
        % out of the range. 
        P(j,:)=[(yaw_rate_test(c1_coord(j)+1)<max(x{i,1})) & (yaw_rate_test(c1_coord(j)+1)>min(x{i,1})), ...
            (yaw_rate_test(c1_coord(j)+2)<max(x{i,2})) & (yaw_rate_test(c1_coord(j)+2)>min(x{i,2})),...
            (yaw_rate_test(c1_coord(j)+3)<max(x{i,3})) & (yaw_rate_test(c1_coord(j)+3)>min(x{i,3})), ...
            (yaw_rate_test(c1_coord(j)+4)<max(x{i,4})) & (yaw_rate_test(c1_coord(j)+4)>min(x{i,4})),...
            (yaw_rate_test(c1_coord(j)+5)>min(x{i,5})) & (yaw_rate_test(c1_coord(j)+5)>min(x{i,5}))];
        
    end
    % Computing Recall Function. Recall Function demonstrates that how wide
    % the predictions are.
    for r=1:horizon
    N(i,r)=((max(x{i,r})-min(x{i,r}))/(2*pi))*((length(find(id==i))));
    end
    M_test=[M_test;P];
    M_Z=[M_Z;Z];
end
%% Computing accuracy
% From M_test a matrix of logicals for 5 columns. If throughout the row,
% M_test is 1 (True), this implies that the prediction is good enough for
% next 5 samples. So, we assign a True (1) only when all the columns are
% True in this row. The total percentage of such 'True' will give us the
% accuracy.

for m=1:length(M_test)
    ResMat(m)=all(M_test(m,:)>0);
end
success=sum(ResMat)/length(M_test)
recall=1-(sum(sum(abs(N)))/length(yaw_rate_test))
vals(v,10)=success;
vals(v,11)=recall;
if mod(v,200)==0
    save('vals.mat','vals')
end
clearvars -except vals

end
