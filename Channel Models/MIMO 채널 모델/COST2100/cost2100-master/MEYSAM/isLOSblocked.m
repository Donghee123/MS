function [isBlocked] = isLOSblocked(currentMSPos,currentBSpos,currentObspos,elipsAxisX,elipsAxisY)
isBlocked = false;
resolution = 1000; % number of points in the crosss section. IT SHOULD BE VERY HIGH > 1000
num_points = 10; % number of points to be considered in line connecting each antenna of BS to the illuminating MS
slope = ( currentMSPos(2) - currentBSpos(2) ) / ( currentMSPos(1) - currentBSpos(1) );
bias = currentMSPos(2) - slope * currentMSPos(1);
xRange = linspace(min(currentMSPos(1) , currentBSpos(1) ) , max(currentMSPos(1) , currentBSpos(1)), num_points );
%xRange = min(currentMSPos(1) , currentBSpos(1) ) : num_points : max(currentMSPos(1) , currentBSpos(1) );
yRange = slope * xRange + bias;
% hold on
% plot(xRange,yRange,'-d')

xLower = currentObspos(1) - elipsAxisX;
xUpper = currentObspos(1) + elipsAxisX;
x_conect_index = ((xRange>=xLower) .* (xRange<=xUpper));
arg_index = find(x_conect_index);
%%%%%%
if length(arg_index) > 0 % note that if length(arg_index)==0, it means there is no blockage!
    if arg_index(1)>1
        start_index = arg_index(1)-1; % -1 is to be sure we did not miss anything on the left side of the first intersection
    else
        start_index = arg_index(1);
    end
    %
    if arg_index(end) < length(xRange)
        end_index = arg_index(end) + 1; % +1 is to make sure we did not miss anything on the right side of the last intersection
    else
        end_index = arg_index(end);
    end
    %
    intersect_x = linspace(xRange(start_index),xRange(end_index),resolution); %    % xRange(start_index:end_index);
    intersect_y = slope * intersect_x + bias;
%     hold on
%     plot(intersect_x,intersect_y,'+')
%     hold on
    distance = ((intersect_x - currentObspos(1)) / elipsAxisX).^2   +   ((intersect_y - currentObspos(2)) / elipsAxisY).^2 ;
    if sum(distance<=1) > 0
        isBlocked = true;
    end
end
