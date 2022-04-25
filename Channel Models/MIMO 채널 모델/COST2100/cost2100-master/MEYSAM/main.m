%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem: given snapNum CSI, we want to say if an object is moving from left to right or from right to left? 
% NumSamples is the number of datapoints we have (for right to left and also fro left to right so in general 2 X NumSamples)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

parameter.freq = [2.58e9 2.62e9]; % starting parameter.freq. - ending freq. [Hz]

parameter.snapNum = 5; % number of snapshots (number of plots of the nvironment per sample index) (given parameter.MSVelo, cover about .25 m)
parameter.snapRate = 4;
disp('NUmber of snaps per second:')
disp(parameter.snapNum / parameter.snapRate) 

parameter.elipsAxisX = 0.25;
parameter.elipsAxisY = 0.1;

parameter.MSPos = [0,10,0];  % AT THE MOMENT I HAVE JUST ONE ILLUMINATOR - BUT I NEED TO EXTEND THIS TO MULTIPLE ILUMINATOR ==> THEN I NEED TO ADD ONE MORE DIMENSSION FOR AntNotBlocked
parameter.MSVelo = [0,0,0];

parameter.NumObs = 1;

parameter.BSPosCenter  = [0 0 0]; % center position of BS array [x, y, z] (m)
parameter.BSPosSpacing = [0.05 0 0]; % inter-position spacing (m), for large arrays.
parameter.BSPosNum = 100; % number of positions at each BS site, for large arrays.

parameter.c_lightSpeed = 3e8;

parameter.NumSamples = 1 %8192; % is the number of samples from letf to right and right to left = it is like the number of datapoints 
%%
RightToLeft = zeros(parameter.NumSamples, parameter.snapNum, parameter.BSPosNum, 2); % num_sample x num_time_steps x num_BS_ant x 2 (2 is for real and imaginary parts of channel)
RightToLeft_phase_amp = zeros(parameter.NumSamples, parameter.snapNum, parameter.BSPosNum, 2);
plot_Env=true; % if parameter.NumSamples is a big number (>3) do not plot environment,because it generates many plots that kills you memory and fill the page
for sample_index =1:parameter.NumSamples % write a loop to create channels for RIGHT to LEFT moving with random speed
    if mod(sample_index,100) == 0
        disp(sample_index)
    end
    x_velocity = (1 + rand(parameter.NumObs,1)); %    1 < v_x < 2
    y_velocity = rand(parameter.NumObs,1) - 0.5 ; %  -0.5 < v_y < 0.5
    parameter.ObsVelo = [x_velocity,y_velocity,zeros(parameter.NumObs,1)];
    
    x_pos = -4 * rand(parameter.NumObs,1) + 1 ;  %  -3 < x < 1
    y_pos = 3 + 2 *  rand(parameter.NumObs,1); %     3 < y < 5
    parameter.ObsPos = [x_pos,y_pos,zeros(parameter.NumObs,1)];
     
    
    [AntNotBlocked,LOS_channels,LOS_phase_ampl,LOS_channels_ReIm] = GenerateChannel(parameter,plot_Env);
    RightToLeft(sample_index,:,:,:) = LOS_channels_ReIm;
    RightToLeft_phase_amp(sample_index,:,:,:) = LOS_phase_ampl;
end

%% 
LeftToRight = zeros(parameter.NumSamples, parameter.snapNum, parameter.BSPosNum, 2); % num_sample x num_time_steps x num_BS_ant x 2 (2 is for real and imaginary parts of channel)
LeftToRight_phase_amp = zeros(parameter.NumSamples, parameter.snapNum, parameter.BSPosNum, 2);
for sample_index =1:parameter.NumSamples % write a loop to create channels for LEFT to RIGHT moving with random speed
    if mod(sample_index,100) == 0
        disp(sample_index)
    end
    x_velocity = -(1 + rand(parameter.NumObs,1)); %      -2 < v_x < -1
    y_velocity = rand(parameter.NumObs,1) - 0.5 ; %      -0.5 < v_y < 0.5
    parameter.ObsVelo = [x_velocity,y_velocity,zeros(parameter.NumObs,1)];
    
    
    x_pos = 3 * rand(parameter.NumObs,1) - 1;            % -1 < x < 2
    y_pos = 3.3 + 2 *  rand(parameter.NumObs,1);
    parameter.ObsPos = [x_pos,y_pos,zeros(parameter.NumObs,1)];
    
    
    [AntNotBlocked,LOS_channels,LOS_phase_ampl,LOS_channels_ReIm] = GenerateChannel(parameter,plot_Env);
    LeftToRight(sample_index,:,:,:) = LOS_channels_ReIm;
    LeftToRight_phase_amp(sample_index,:,:,:) = LOS_phase_ampl;
end



































