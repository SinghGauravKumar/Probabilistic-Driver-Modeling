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
% Yaw
yaw=data(:,12)*(pi/180);
% Yaw Rate 
yaw_rate=data(:,9)*(pi/180);
t=0.5;
ax=zeros(size(ax));
ay=s(size(ay));
yaw=ones(size(yaw));

vx=zeros(size(ax));
vy=zeros(size(ay));

for i=2:length(ax)
    vx(i)=vx(i-1)+ax(i)*t;
    vy(i)=vy(i-1)+ay(i)*t;
end

xdot=vx.*cos(yaw);
ydot=vy.*sin(yaw);
figure
plot(xdot,ydot)
x=zeros(size(xdot));
y=zeros(size(ydot));
for i=2:length(xdot)
    x(i)=x(i-1)+xdot(i)*t;
    y(i)=y(i-1)+ydot(i)*t;
end

figure
plot(x(1:1000),y(1:1000),'linewidth',2)
xlabel('x')
ylabel('y')
figure
subplot(311)
plot(ax)
title('ax')
subplot(312)
plot(ay)
title('ay')
subplot(313)
plot(yaw)
title('yaw')