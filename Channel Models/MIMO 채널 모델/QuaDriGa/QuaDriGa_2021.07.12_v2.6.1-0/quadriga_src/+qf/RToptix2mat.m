function out = RToptix2mat( rtopterix_res_dir, max_no_paths  )
%RTOPTIX2MAT Reads the RToptix results and converts them to a MATLAB struct
%
% Description:
%   This function reads the data generated by the RToptix ray tracing ending and writes it to a
%   MATLAB struct variable with the following fields:
%
% Fields:
%   tx_pos
%   Position of the TX in Cartesian coordinates using units of [m]; [3 x 1] vector
%
%   rx_pos
%   Rosition of rhe RX in Cartesian coordinates using units of [m]; [3 x 1] vector
%
%   no_path
%   The number of paths in the output struct
%
%   frequency
%   Center frequency in [Hz]
%
%   pow
%   The normalized path gain (squared average amplitude) for each path in linear scale;
%   [1 x no_path]
%
%   delay
%   The delays for each path in [s]; Output paths are ordered by delay; [1 x no_path]
%
%   aod
%   The azimuth of departure angles for each path in [rad]; [1 x no_path]
%
%   eod
%   The elevation of departure angles for each path in [rad]; [1 x no_path]
%
%   aoa
%   The azimuth of arrival angles for each path in [rad]; [1 x no_path]
%
%   eoa
%   The elevation of departure angles for each path in [rad]; [1 x no_path]
%
%   xprmat
%   The complex-valued polarization transfer matrix describing the polarization change during
%   scattering; [2 x 2 x no_path]
%
% Input:
%   rtopterix_res_dir
%   Name of the directory which contains the RToptix results (string)
%
%   max_no_paths
%   Maximum number of paths to be returned. By default, all paths are returned. The 'max_no_paths'
%   is smaller than the number of paths in the RT results, only the 'max_no_paths' with strongest
%   power are returned.
%
% Output:
%   out
%   The result data structure.
%
%
% QuaDRiGa Copyright (C) 2011-2020
% Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
% Fraunhofer Heinrich Hertz Institute, Einsteinufer 37, 10587 Berlin, Germany
% All rights reserved.
%
% e-mail: quadriga@hhi.fraunhofer.de
%
% This file is part of QuaDRiGa.
%
% The Quadriga software is provided by Fraunhofer on behalf of the copyright holders and
% contributors "AS IS" and WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, including but not limited to
% the implied warranties of merchantability and fitness for a particular purpose.
%
% You can redistribute it and/or modify QuaDRiGa under the terms of the Software License for 
% The QuaDRiGa Channel Model. You should have received a copy of the Software License for The
% QuaDRiGa Channel Model along with QuaDRiGa. If not, see <http://quadriga-channel-model.de/>.

if ~exist( 'rtopterix_res_dir','var' ) || isempty( rtopterix_res_dir )
    error('QuaDRiGa:RToptix2mat','Name of results folder is not given.');
elseif exist( rtopterix_res_dir, 'dir' ) == 0
    error('QuaDRiGa:RToptix2mat',['Results folder "',rtopterix_res_dir,'" does not exist.']);
end

if ~exist( 'max_no_paths', 'var' ) || isempty( max_no_paths )
    max_no_paths = Inf;
end

% Get list of file containing path data
path_files = dir( fullfile( rtopterix_res_dir,'*.txt' ) );

% Read data from path files
no_path_files   = numel(path_files);
fl_freq         = nan( 1,no_path_files );
fl_no_path      = zeros( 1,no_path_files );
fl_tx_pos       = nan( 3,no_path_files );
fl_no_rx        = nan( 1,no_path_files );
fl_rx_pos       = nan( 3,no_path_files );
fl_path_data    = cell(no_path_files,1);

for n = 1 : no_path_files
    fid = fopen( fullfile( rtopterix_res_dir, path_files(n).name ) , 'r');
    tline = fgetl(fid);
    
    % Read Center Frequency
    i_line = 1;
    while ( ~feof(fid) && i_line < 20 && numel(tline) < 20 ) || ...
            ( i_line < 20 && ~strcmp(tline(1:20), 'simulation frequency') )
        i_line = i_line + 1;
        tline = fgetl(fid);
    end
    if i_line ~= 20
        fl_freq(:,n) = sscanf(tline,'simulation frequency : %f Hz');
    else
        warning('QuaDRiGa:RToptix2mat',['"',path_files(n).name,'" invalid!']);
        continue
    end
    
    % Read TX position
    while ( ~feof(fid) && i_line < 20 && numel(tline) < 21 ) ||...
            ( i_line < 20 && ~strcmp(tline(1:21), 'point source position') )
        i_line = i_line + 1;
        tline = fgetl(fid);
    end
    if i_line ~= 20
        fl_tx_pos(:,n) = sscanf(tline,'point source position : %f %f %f');
    else
        warning('QuaDRiGa:RToptix2mat',['"',path_files(n).name,'" invalid!']);
        continue
    end
    
    % Read number of receivers:
    while ( ~feof(fid) && i_line < 20 && numel(tline) < 19 ) ||...
            ( i_line < 20 && ~strcmp(tline(1:19), 'number of receivers') )
        i_line = i_line + 1;
        tline = fgetl(fid);
    end
    if i_line ~= 20
        fl_no_rx(:,n) = sscanf(tline,'number of receivers : %f');
    else
        warning('QuaDRiGa:RToptix2mat',['"',path_files(n).name,'" invalid!']);
        continue
    end
    
    % Read number of paths
    while ( ~feof(fid) && i_line < 20 && numel(tline) < 14 ) || ~strcmp(tline(1:14), 'receiver 1 of ')
        i_line = i_line + 1;
        tline = fgetl(fid);
    end
    if i_line ~= 20
        tmp = sscanf(tline,'receiver 1 of %d : %d propagation paths');
        fl_no_path(n) = tmp(2);
        if tmp(1) ~= 1
            warning('QuaDRiGa:RToptix2mat',...
                ['"',path_files(n).name,'" contains more the one receiver - only one is allowed!']);
        end
    else
        warning('QuaDRiGa:RToptix2mat',['"',path_files(n).name,'" invalid!']);
        continue
    end
    
    % Read RX position
    while ( ~feof(fid) && i_line < 20 && numel(tline) < 17 ) || ~strcmp(tline(1:17), 'receiver position')
        i_line = i_line + 1;
        tline = fgetl(fid);
    end
    if i_line ~= 20
        fl_rx_pos(:,n) = sscanf(tline,'receiver position : %f %f %f');
    else
        warning('QuaDRiGa:RToptix2mat',['"',path_files(n).name,'" invalid!']);
        continue
    end
    
    % Read path data
    tmp = fscanf(fid, ['\n\tpath %*d of %*d%*[:()RTDS0-9- ]\n',...
        '\t\tlaunch angle = %f %f (%*f %*f)\n',...
        '\t\tarrival angle = %f %f (%*f %*f)\n',...
        '\t\ttime delay = %f ns (%*f m free space distance)\n',...
        '\t\thorizontal polarization data:\n',...
        '\t\t\tattenuation : %f dB\n',...
        '\t\t\tcopolar transfer factor : %f %f\n',...
        '\t\t\tcrosspolar transfer factor: %f %f\n',...
        '\t\tvertical polarization data:\n',...
        '\t\t\tattenuation : %f dB\n',...
        '\t\t\tcopolar transfer factor : %f %f\n',...
        '\t\t\tcrosspolar transfer factor: %f %f\n']);
    
    fl_path_data{n} = reshape( tmp,[],fl_no_path(n));
    
    fclose(fid);
end

% Remove invalid entries
ind = ~isnan(fl_no_rx);
fl_freq         = fl_freq(ind);
fl_no_path      = fl_no_path(ind);
fl_tx_pos       = fl_tx_pos(:,ind);
fl_rx_pos       = fl_rx_pos(:,ind);
fl_path_data    = fl_path_data(ind);

% Find matching positions
tmp = [fl_tx_pos;fl_rx_pos];
nT  = size(tmp,2);
ind = false( nT );
iT = 1; iI = 1;
while iT <= nT
    ind( iI,: ) = sum( (repmat( tmp(:,iT), 1,nT) - tmp).^2,1 ) < 1e-4;
    iT = find( ind( iI,: ),1,'last')+1;
    iI = iI + 1;
end
ind = ind(1:iI-1,:);
nI  = size( ind,1 );

% Prepare output variable
out = struct;

% Process RT data
for iI = 1 : nI
    iRT = find( ind(iI,:) );
    
    rx_pos = fl_rx_pos(:,iRT(1));
    tx_pos = fl_tx_pos(:,iRT(1));
    no_path = sum( fl_no_path(iRT) );
    
    iP = 0;
    DL  = zeros( 1,no_path );
    M   = zeros( 2,2,no_path );
    POW = zeros( 1,no_path );
    AOD = zeros( 1,no_path );
    EOD = zeros( 1,no_path );
    AOA = zeros( 1,no_path );
    EOA = zeros( 1,no_path );
    for n = 1 : numel( iRT )
        if fl_no_path(iRT(n)) > 0
            iP = iP(end) + (1:fl_no_path(iRT(n)));
            
            tmp = sqrt( 10.^(-0.1*fl_path_data{iRT(n)}(11,:)) );        % vertical polarization
            M( 1,1,iP ) = fl_path_data{iRT(n)}(12,:).*tmp + 1j*fl_path_data{iRT(n)}(13,:).*tmp;
            M( 1,2,iP ) = fl_path_data{iRT(n)}(14,:).*tmp + 1j*fl_path_data{iRT(n)}(15,:).*tmp;
            
            tmp = sqrt( 10.^(-0.1*fl_path_data{iRT(n)}(6,:)) );        % horizontal polarization
            M( 2,2,iP ) = fl_path_data{iRT(n)}(7,:).*tmp + 1j*fl_path_data{iRT(n)}(8,:).*tmp;
            M( 2,1,iP ) = fl_path_data{iRT(n)}(9,:).*tmp + 1j*fl_path_data{iRT(n)}(10,:).*tmp;
            
            POW( iP ) = reshape( max(sum(abs(M(:,:,iP)).^2,2),[],1), 1, [] );
            DL( iP )  = fl_path_data{iRT(n)}(5,:)/1e9;
            EOD( iP ) = pi/2 - fl_path_data{iRT(n)}(1,:);
            AOD( iP ) = fl_path_data{iRT(n)}(2,:);
            EOA( iP ) = pi/2 - fl_path_data{iRT(n)}(3,:);
            AOA( iP ) = fl_path_data{iRT(n)}(4,:);
        end
    end
    
    % Reduce number of paths if required (only keep the strongest paths)
    if no_path > max_no_paths
        [~,ii] = sort(POW,'descend');
        ii = sort(ii(1:max_no_paths));
        DL = DL(ii);
        POW = POW(ii);
        EOD = EOD(ii);
        EOA = EOA(ii);
        AOD = AOD(ii);
        AOA = AOA(ii);
        M = M(:,:,ii);
        no_path = max_no_paths;
    end
    M = M./repmat( permute( sqrt(POW), [1,3,2] ), [2,2,1] );
    
    % Calculate LOS-delay
    dTR = sqrt(sum((tx_pos - rx_pos).^2));
    LOS_delay = dTR ./ 299792458;
    
    % Make sure that the LOS component is included in the path list, even if it has zero-power
    iLOS = abs(DL - LOS_delay)./LOS_delay < 0.001;
    if sum( iLOS ) ~= 0 % Sort paths by delay
        [DL,ii] = sort(DL);
        POW = POW(ii);
        EOD = EOD(ii);
        EOA = EOA(ii);
        AOD = AOD(ii);
        AOA = AOA(ii);
        M = M(:,:,ii);
    else % Add zero-power LOS component
        [DL,ii] = sort(DL);
        
        LOS_angles = zeros( 1,4 );
        LOS_angles(1) = atan2( rx_pos(2) - tx_pos(2) , rx_pos(1) - tx_pos(1) ); % ThetaBs 
        LOS_angles(2) = LOS_angles(1) + pi;
        LOS_angles(3) = atan( ( rx_pos(3) - tx_pos(3) ) ./...
            sqrt( (tx_pos(1) - rx_pos(1)).^2 + (tx_pos(2) - rx_pos(2)).^2 ) );
        LOS_angles(4) = -LOS_angles(3);
        
        DL  = [ LOS_delay, DL ]; %#ok
        POW = [ double(no_path == 0)*1e-20, POW(ii) ]; 
        AOD = [ LOS_angles(1), AOD(ii) ];
        AOA = [ LOS_angles(2), AOA(ii) ];
        EOD = [ LOS_angles(3), EOD(ii) ];
        EOA = [ LOS_angles(4), EOA(ii) ];
        M   = cat(3, [1,0;0,-1], M(:,:,ii) );
        no_path = no_path + 1;
    end
    
    % Write output data structure
    out(iI).tx_pos = tx_pos;
    out(iI).rx_pos = rx_pos;
    out(iI).no_path = no_path;
    out(iI).frequency = fl_freq(iRT(1));
    out(iI).pow =  POW;
    out(iI).delay = DL;
    out(iI).aod = AOD;
    out(iI).eod = EOD;
    out(iI).aoa = AOA;
    out(iI).eoa = EOA;
    out(iI).xprmat = M;
  
end

end
