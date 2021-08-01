clear all
close all
clc

freq = [2.58e9 2.62e9]; % starting freq. - ending freq. [Hz]
snapNum = 4; % number of snapshots (given MSVelo, cover about .25 m)
snapRate = 1;
elipsAxisX = 0.25;
elipsAxisY = 0.1;
MSPos = [0,10,0];  % AT THE MOMENT I HAVE JUST ONE ILLUMINATOR - BUT I NEED TO EXTEND THIS TO MULTIPLE ILUMINATOR ==> THEN I NEED TO ADD ONE MORE DIMENSSION FOR AntNotBlocked
MSVelo = [0,0,0];
ObsPos = [-2,5,0;...           % two users
    -2,7,0];
ObsVelo = [1,0,0;...
    2,0,0];
BSPosCenter  = [0 0 0]; % center position of BS array [x, y, z] (m)
BSPosSpacing = [0.05 0 0]; % inter-position spacing (m), for large arrays.
BSPosNum = 100; % number of positions at each BS site, for large arrays.

c_lightSpeed = 3e8;
%% Position of the antennas within one BS, for very large arrays
startPos_AntArray = BSPosCenter - ((BSPosNum-1)/2)*BSPosSpacing;
BS_antpos_vec=zeros(BSPosNum,3);
for i=1:BSPosNum
    BS_antpos_vec(i,:) = startPos_AntArray + (i-1) * BSPosSpacing;
    
end
%% Obstacke Initial Positions
numObs = length(ObsPos(:,1));                                    
%% Check which antennas are blocked at each snapshot for one MS (iluminator) and multiple obstacles --> need to be revised for multiple MS (iluminator) - AntNotBlocked should have one more dimenssion for different MSs, which I have not done yet.
AntNotBlocked = ones(BSPosNum,snapNum); % if AntNotBlocked(i,j) becomes 0, it means the antenna i is blocked by the Obstackels at snapshot j
for snap_index=1:snapNum
    envt.Obspos(snap_index,:,:) = ObsPos + ( (snap_index-1) * ObsVelo * (1/snapRate) );
    for Obs_index=1:numObs
        for BSantena_index=1:BSPosNum
            for MSindex=1:length(MSPos(:,1)) % if we have multiple iluminating MS. BUt it is not complete as AntNotBlocked should have one more dimenssion for different MSs, which I have not done yet.
                isBlocked = isLOSblocked(MSPos(MSindex,:),BS_antpos_vec(BSantena_index,:),squeeze(envt.Obspos(snap_index,Obs_index,:)),elipsAxisX,elipsAxisY);
                if isBlocked
                    AntNotBlocked(BSantena_index,snap_index) = 0;
                end
            end
        end
    end
end
%% Now we have the attacked models! Lets plot the environment and create the delays and amplitudes
LOS_delay_ampl = zeros(2, BSPosNum, snapNum); % the first dimenssion is delay and absolute value
LOS_channels = zeros(BSPosNum, snapNum); % it is important to initialize it with 0, as those antenna which are blocked need to be zero!
for snap_index=1:snapNum
    % LETS PLOT THE ENVIRONMENT AT THIS SNAPSHOT
    figure
    % First the MS antenna
    plot(MSPos(1),MSPos(2),'o'); hold on
    % Second the BS antennas
    for i=1:BSPosNum                                                      
        plot(BS_antpos_vec(i,1),BS_antpos_vec(i,2),'b*'); hold on
    end
    % Third: the Obstacles ellipse
    for Obs_index=1:numObs
        [x,y] = plotElipse(elipsAxisX,elipsAxisY,envt.Obspos(snap_index,Obs_index,:));
    end
    % Fourth: plot LOS lines between MS and BS antennas
    number_points = 2;
    for BSantena_index=1:BSPosNum                                                      
        if AntNotBlocked(BSantena_index,snap_index) == 1
            PlotConnectLine(MSPos(MSindex,:),BS_antpos_vec(BSantena_index,:), number_points);
            dist_LOS = sqrt( (MSPos(MSindex,1)-BS_antpos_vec(BSantena_index,1))^2 + (MSPos(MSindex,2)-BS_antpos_vec(BSantena_index,2))^2 );
            delayLOS = dist_LOS / c_lightSpeed;
            pathloss_LOS_dB = 20*log10(dist_LOS) + 20*log10(mean(freq)) - 147.6;
            pathloss_LOS = 10^(-pathloss_LOS_dB/10);
            LOS_delay_ampl(:,BSantena_index,snap_index) = [delayLOS,pathloss_LOS];
            LOS_channels(BSantena_index,snap_index) = sqrt(pathloss_LOS) * exp(1i * 2 * pi * mean(freq) * delayLOS);  % NOTE THAT SQRT(pathloss) should be applied here! From D. Tse book, page 24, assuming a Block Fading (Flat and slow fading).
        end
    end
    
end























































% 
% % For every MS
% for m = 1:numMS
%     MS(m).idx = m; % Label of MS(m)       
%     MS(m).pos = MSPos(m,:); % Position of MS
%     MS(m).velo = MSVelo(m,:); % Velocity of MS
%     
%     MS(m).cluster_local = get_cluster_local('MS', m, paraEx, paraSt); % MS local cluster
%     MS(m).mpc_local = get_mpc( MS(m).cluster_local, paraSt); % MPC in MS local cluster
%     MS(m).dmc_local = get_dmc( MS(m).cluster_local, paraSt); % DMC in MS local cluster          
% end
% 
% % Get the channel of each link
% for nSS = 1:snapNum  
%     for nB = 1:numBS % Loop for every BS
%         for nB_pos = 1:BSPosNum(nB) % Loop for every position at the same BS
%             for nM = 1:numMS % Loop for every MS  
%                 link(nB, nM).channel{nB_pos, nSS} = get_channel(BS(nB), BS.BS_pos(nB_pos, :), MS(nM), VRtable, MS_VR, BS_VR, BS_VR_len, BS_VR_slope,...
%                                                                 cluster, mpc, dmc, paraEx, paraSt);
%                 link(nB, nM).MS(nB_pos, nSS) = MS(nM); % Record of MS at snapshot m
%                 link(nB, nM).BS(nB_pos, nSS) = BS(nB); % Record of BS at snapshot m
%             end
%         end    
%     end
%    
%     % Update MS information according to movement
%     for nM = 1:numMS
%         MS(nM) = update_chan(MS(nM), paraEx, paraSt);
%     end
% end