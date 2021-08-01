function [x,y] = plotElipse(elipsAxisX,elipsAxisY,center) 

t=-pi:0.1:pi+0.05;
x = center(1) + elipsAxisX * cos(t);
y = center(2) + elipsAxisY * sin(t);
plot(x,y); hold on