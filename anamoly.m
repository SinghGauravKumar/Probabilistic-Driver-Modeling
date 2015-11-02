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

clear all
close all
clc

% Create the globe with graticule
axesm('globe');
gridm('GLineStyle','-','Gcolor',[.8 .7 .6],'Galtitude', .02);    
load coast
plot3m(lat,long,.01,'k');
%% Read Data
data=csvread('Fcy_table.csv',1);
%% Put variables of interest in column vectors
lon=data(:,13);
lat=data(:,12);

% Plot trajectory
h = plot3m(lat,lon,'r.-', 'MarkerSize', 20);
view(3);

