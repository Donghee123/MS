function plotEnv(MSPos,BS_antpos_vec,ObsPos,ObsVelo,snapNum,snapRate,elipsAxisX,elipsAxisY,BSPosNum)





%% Obstacke Initial Positions
numObs = length(ObsPos(:,1));
envt.Obspos = zeros(snapNum,numObs,3); % the center of the obstacle. Note the obstacke is an ellipsoid
%envt.Obspos(1,:,:) = ObsPos;
%%
  


                                      
%%
AntNotBlocked = ones(BSPosNum,snapNum); % if AntNotBlocked(i,j) becomes 0, it means the antenna i is blocked by the Obstackels at snapshot j
for snap_index=1:snapNum
    %figure
    envt.Obspos(snap_index,:,:) = ObsPos + ( (snap_index-1) * ObsVelo * (1/snapRate) );
    
    
    
        for Obs_index=1:numObs
            figure
            % LETS PLOT THE ENVIRONMENT AT THIS SNAPSHOT
            % First the MS antenna
            plot(MSPos(1),MSPos(2),'o'); hold on
            % Second the BS antennas
            for i=1:BSPosNum                                                       % BSPosNum = length(BS_antpos_vec)
                plot(BS_antpos_vec(i,1),BS_antpos_vec(i,2),'b*'); hold on
            end
            % Third: the Obstacles ellipse
            [x,y] = plotElipse(elipsAxisX,elipsAxisY,envt.Obspos(snap_index,Obs_index,:));
            for BSantena_index=1:BSPosNum
                for MSindex=1:length(MSPos(:,1)) % if we have multiple iluminating MS.
                    isBlocked = isLOSblocked(MSPos(MSindex,:),BS_antpos_vec(BSantena_index,:),squeeze(envt.Obspos(snap_index,Obs_index,:)),elipsAxisX,elipsAxisY);
                    if isBlocked
                        AntNotBlocked(BSantena_index,snap_index) = 0;
                    end
                end
            end
            %squeeze(envt.Obspos(snap_index,Obs_index,:))
            % check ifthe antenna is blocked
    end
end
%%





























