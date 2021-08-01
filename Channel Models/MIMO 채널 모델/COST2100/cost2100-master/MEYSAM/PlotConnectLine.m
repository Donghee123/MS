function[xRange,yRange,slope,bias] = PlotConnectLine(currentMSPos,currentBSpos, number_points)

slope = ( currentMSPos(2) - currentBSpos(2) ) / ( currentMSPos(1) - currentBSpos(1) );
bias = currentMSPos(2) - slope * currentMSPos(1);

xRange = linspace(min(currentMSPos(1) , currentBSpos(1) ) , max(currentMSPos(1) , currentBSpos(1)), number_points );
%xRange = min(currentMSPos(1) , currentBSpos(1) ) : num_points : max(currentMSPos(1) , currentBSpos(1) );
yRange = slope * xRange + bias;
hold on
plot(xRange,yRange,'--s')