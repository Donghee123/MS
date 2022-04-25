function [AntNotBlocked,LOS_channels,LOS_phase_ampl, LOS_channels_ReIm] = GenerateChannel(parameter,plot_Env)


%% Position of the antennas within one BS, for very large arrays
startPos_AntArray = parameter.BSPosCenter - ((parameter.BSPosNum-1)/2)*parameter.BSPosSpacing;
BS_antpos_vec=zeros(parameter.BSPosNum,3);
for i=1:parameter.BSPosNum
    BS_antpos_vec(i,:) = startPos_AntArray + (i-1) * parameter.BSPosSpacing;
    
end
%% Obstacke Initial Positions
numObs = length(parameter.ObsPos(:,1));                                    
%% Check which antennas are blocked at each snapshot for one MS (transmitter or iluminator) and multiple obstacles --> need to be revised for multiple MS (iluminator) - AntNotBlocked should have one more dimenssion for different MSs, which I have not done yet.
AntNotBlocked = ones(parameter.BSPosNum,parameter.snapNum); % if AntNotBlocked(i,j) becomes 0, it means the antenna i is blocked by the Obstackels at snapshot j
for snap_index=1:parameter.snapNum
    envt.ObsPos(snap_index,:,:) = parameter.ObsPos + ( (snap_index-1) * parameter.ObsVelo * (1/parameter.snapRate) );
    for Obs_index=1:numObs
        for BSantena_index=1:parameter.BSPosNum
            for MSindex=1:length(parameter.MSPos(:,1)) % if we have multiple iluminating MS. BUt it is not complete as AntNotBlocked should have one more dimenssion for different MSs, which I have not done yet.
                isBlocked = isLOSblocked(parameter.MSPos(MSindex,:),BS_antpos_vec(BSantena_index,:),squeeze(envt.ObsPos(snap_index,Obs_index,:)),parameter.elipsAxisX,parameter.elipsAxisY);
                if isBlocked
                    AntNotBlocked(BSantena_index,snap_index) = 0;
                end
            end
        end
    end
end
%% Now we have the attacked models! Lets plot the environment and create the delays and amplitudes
LOS_phase_ampl = zeros(parameter.snapNum, parameter.BSPosNum, 2); % the last dimenssion is phase and absolute value (amplitude)
LOS_channels = zeros(parameter.snapNum, parameter.BSPosNum); % it is important to initialize it with 0, as those antenna which are blocked need to be zero!
LOS_channels_ReIm = zeros(parameter.snapNum, parameter.BSPosNum, 2); % last dimenssion contains real and imaginary pasrt of the channel

if plot_Env
    %% Create the channel and Plot the environmet
    for snap_index=1:parameter.snapNum
        % LETS PLOT THE ENVIRONMENT AT THIS SNAPSHOT
        figure
        % First the MS antenna
        plot(parameter.MSPos(1),parameter.MSPos(2),'o'); hold on
        % Second the BS antennas
        for i=1:parameter.BSPosNum                                                      
            plot(BS_antpos_vec(i,1),BS_antpos_vec(i,2),'b*'); hold on
        end
        % Third: the Obstacles ellipse
        for Obs_index=1:numObs
            [x,y] = plotElipse(parameter.elipsAxisX,parameter.elipsAxisY,envt.ObsPos(snap_index,Obs_index,:));
        end
        % Fourth: plot LOS lines between MS and BS antennas
        number_points = 2;
        for BSantena_index=1:parameter.BSPosNum                                                      
            if AntNotBlocked(BSantena_index,snap_index) == 1
                PlotConnectLine(parameter.MSPos(MSindex,:),BS_antpos_vec(BSantena_index,:), number_points);
                dist_LOS = sqrt( (parameter.MSPos(MSindex,1)-BS_antpos_vec(BSantena_index,1))^2 + (parameter.MSPos(MSindex,2)-BS_antpos_vec(BSantena_index,2))^2 );
                delayLOS = dist_LOS / parameter.c_lightSpeed;
                pathloss_LOS_dB = 20*log10(dist_LOS) + 20*log10(mean(parameter.freq)) - 147.6;
                pathloss_LOS = 10^(-pathloss_LOS_dB/10);
                LOS_phase_ampl(snap_index , BSantena_index, :) = [ mod( 2 * pi * mean(parameter.freq) * delayLOS,2*pi ) , sqrt(pathloss_LOS)];
                LOS_channels(snap_index,BSantena_index) = sqrt(pathloss_LOS) * exp(1i * 2 * pi * mean(parameter.freq) * delayLOS);  % NOTE THAT SQRT(pathloss) should be applied here! From D. Tse book, page 24, assuming a Block Fading (Flat and slow fading).
            end
        end
        LOS_channels_ReIm(snap_index , :, 1) = real(LOS_channels(snap_index, :,1));
        LOS_channels_ReIm(snap_index , :, 2) = imag(LOS_channels(snap_index, :,1));
    end
else
    %% Create the channel but do not plot environment
    for snap_index=1:parameter.snapNum
        for BSantena_index=1:parameter.BSPosNum                                                      
            if AntNotBlocked(BSantena_index,snap_index) == 1
                %PlotConnectLine(parameter.MSPos(MSindex,:),BS_antpos_vec(BSantena_index,:), number_points);
                dist_LOS = sqrt( (parameter.MSPos(MSindex,1)-BS_antpos_vec(BSantena_index,1))^2 + (parameter.MSPos(MSindex,2)-BS_antpos_vec(BSantena_index,2))^2 );
                delayLOS = dist_LOS / parameter.c_lightSpeed;
                pathloss_LOS_dB = 20*log10(dist_LOS) + 20*log10(mean(parameter.freq)) - 147.6;
                pathloss_LOS = 10^(-pathloss_LOS_dB/10);
                LOS_phase_ampl(snap_index , BSantena_index, :) = [ mod( 2 * pi * mean(parameter.freq) * delayLOS,2*pi ) , sqrt(pathloss_LOS)];
                LOS_channels(snap_index, BSantena_index) = sqrt(pathloss_LOS) * exp(1i * 2 * pi * mean(parameter.freq) * delayLOS);  % NOTE THAT SQRT(pathloss) should be applied here! From D. Tse book, page 24, assuming a Block Fading (Flat and slow fading). 
            end
        end
        LOS_channels_ReIm(snap_index , :, 1) = real(LOS_channels(snap_index, :,1));
        LOS_channels_ReIm(snap_index , :, 2) = imag(LOS_channels(snap_index, :,1));
    end
end






















