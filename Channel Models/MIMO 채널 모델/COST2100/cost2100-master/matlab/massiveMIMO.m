clear all
close all
clc
%% Demo model to run the COST 2100 channel model
%------
%Input:
%------
% Network : 'IndoorHall_5GHz','SemiUrban_300MHz','Indoor_CloselySpacedUser_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz', or 'SemiUrban_VLA_2_6GHz'
% Band : 'Wideband' or 'Narrowband'
% Link: 'Multiple' or 'Single'
% Antenna: 'SISO_omni', 'MIMO_omni', 'MIMO_dipole', 'MIMO_measured', 'MIMO_Cyl_patch', 'MIMO_VLA_omni'
% scenario: 'LOS' or 'NLOS'        
% freq: Frequency band [Hz]
% snapRate: Number of snapshots per s
% snapNum: Number of simulated snapshots         
% BSPosCenter: Center position of BS array [x, y, z] [m]
% BSPosSpacing: Inter-position spacing [m], for large arrays
% BSPosNum: Number of positions at each BS site, for large arrays
% MSPos: Position of MSs [m]
% MSVelo: Velocity of MSs [m/s]
%------
%Output:
%------ 
%MIMO_VLA_omni: Transfer function for a physically large array with 128 omni-directional 
% antennas, with lambda/2 inter-element separation, and MS with omni-directional antenna.
% create_IR_omni_MIMO_VLA: users have to set up the frequency separation, delta_f
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose a Network type out of 
% {'IndoorHall_5GHz','SemiUrban_300MHz','Indoor_CloselySpacedUser_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz','SemiUrban_VLA_2_6GHz'}
% to parameterize the COST2100 model
Network = 'Indoor_CloselySpacedUser_2_6GHz';                                               % IndoorHall_5GHz does not support Multiple link (multiple MSs)
% In COST2100, # links = # BSs x # MSs
% Set Link type to `Multiple' if you work with more than one link              ==>> means more than one user (each linke is a user)
% Set Link type to `Single' otherwise
Link = 'Single';
% Choose an Antenna type out of
% {'SISO_omni', 'MIMO_omni', 'MIMO_dipole', 'MIMO_measured', 'MIMO_Cyl_patch', 'MIMO_VLA_omni'}
Antenna = 'MIMO_VLA_omni';
% ...and type of channel: {'Wideband','Narrowband'}.
Band = 'Narrowband';

% Here are some tested combinations of the above variables:
% 'IndoorHall_5GHz', 'Single', 'SISO_omni', 'Wideband'
% 'SemiUrban_300MHz', 'Single', 'SISO_omni', 'Wideband'
% 'SemiUrban_300MHz', 'Multiple', 'MIMO_omni', 'Wideband'
% 'Indoor_CloselySpacedUser_2_6GHz', 'Multiple', 'MIMO_Cyl_patch', 'Wideband'
% 'SemiUrban_CloselySpacedUser_2_6GHz', 'Multiple', 'MIMO_Cyl_patch', 'Wideband'
% 'SemiUrban_VLA_2_6GHz', 'Single', 'MIMO_VLA_omni', 'Wideband'
% 'SemiUrban_VLA_2_6GHz', 'Multiple', 'MIMO_VLA_omni', 'Wideband'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch Network
    %%%%%%%%%%%%%%%%%%%%%%
    case 'IndoorHall_5GHz'
    %%%%%%%%%%%%%%%%%%%%%%
        switch Link
            case 'Single'
                scenario = 'LOS'; % {'LOS'} only LOS is available
                freq = [-10e6 10e6]+5.3e9; % [Hz}
                snapRate = 1; % Number of snapshots per s
                snapNum = 10; % Number of snapshots
                MSPos  = [-5 10  0]; % [m]
                MSVelo = [1 0 0]; % [m/s]
                BSPosCenter  = [0 0 0]; % Center position of BS array [x, y, z] [m]
                BSPosSpacing = [0.05 0 0]; % Inter-position spacing (m), for large arrays
                BSPosNum = 17; % Number of positions at each BS site, for large arrays
            case 'Multiple'
                error('IndoorHall_5GHz does not support multiple links.');
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'Indoor_CloselySpacedUser_2_6GHz'
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        scenario = 'LOS';
        freq = [2.58e9 2.62e9]; % starting freq. - ending freq. [Hz]
        snapNum = 10; % number of snapshots (given MSVelo, cover about .25 m)
        snapRate = 1; % number of snapshots per second (sample at 0.05 m/snapshot)

        % closely-spaced users
        MSPos = [-5,5,0];
        MSVelo = [1,0,0];
%         MSPos = [5,5,0;...           % two users
%             7,7,0];
%         MSVelo = [0,0,0;...
%             0,0,0];
%         MSPos  = [   -2.5600    1.7300    2.2300;...
%                      -3.0800    1.7300    2.2300;...
%                      -2.5600    2.6200    2.5800;...
%                      -4.6400    1.7300    2.2300;...
%                      -2.5600    4.4000    3.3000;...
%                      -3.0800    3.5100    2.9400;...
%                      -3.6000    4.4000    3.3000;...
%                      -4.1200    4.4000    3.3000;...
%                      -4.1200    2.6200    2.5800]; % [x, y, z] (m)
% 
%         MSVelo = repmat([-.25,0,0],9,1); % [x, y, z] (m/s)

        BSPosCenter  = [0 0 0]; % center position of BS array [x, y, z] (m)
        BSPosSpacing = [0.05 0 0]; % inter-position spacing (m), for large arrays.
        BSPosNum = 10; % number of positions at each BS site, for large arrays.
        
        %BSPosCenter = BSPosCenter - mean(MSPos); % center users a origo                 I COMMENTED TIS LINE AND NEXT! THEY MESS WITH THE POSITION INFORMATION AND THEIR APPLICATION IS NOT CLEAR FOR ME!                                      %I%
        %MSPos = MSPos - repmat(mean(MSPos),size(MSPos,1),1); % center users a origo     I COMMENTED TIS LINE AND NEXT! THEY MESS WITH THE POSITION INFORMATION AND THEIR APPLICATION IS NOT CLEAR FOR ME! 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get the MPCs from the COST 2100 channel model
[...
    paraEx,...       % External parameters
    paraSt,...       % Stochastic parameters
    link,...         % Simulated propagation data for all links [nBs,nMs]
    env...           % Simulated environment (clusters, clusters' VRs, etc.)
] = cost2100...
(...
    Network,...      % Model environment
    scenario,...     % LOS or NLOS
    freq,...         % [starting freq., ending freq.]
    snapRate,...     % Number of snapshots per second
    snapNum,...      % Total # of snapshots
    BSPosCenter,...  % Center position of each BS
    BSPosSpacing,... % Position spacing for each BS (parameter for physically very-large arrays)
    BSPosNum,...     % Number of positions on each BS (parameter for physically very-large arrays)
    MSPos,...        % Position of each MS
    MSVelo...        % Velocity of MS movements
    );         
toc

%% Visualize the generated environment
if 1  
    switch Network
        case {'IndoorHall_5GHz','SemiUrban_300MHz'}   
             visual_channel(paraEx, paraSt, link, env);
        case {'SemiUrban_VLA_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz','Indoor_CloselySpacedUser_2_6GHz'}   
             visualize_channel_env(paraEx, paraSt, link, env); axis equal; view(2);
    end   
end
%% Ccombine propagation data with antenna patterns
% Construct the channel data
% The following is example code
% End users can write their own code
switch Link
    %%%%%%%%%%%%%%
    case 'Single'
        %%%%%%%%%%%%%%
        % Channel transfer function with 128 omni-directional at the
        % BS, with lambda/2 inter-element separation, and one MS
        % with one omni-directional antenna
        delta_f = (freq(2)-freq(1))/256;
        h_omni_MIMO = create_IR_omni_MIMO_VLA(link,freq,delta_f,Band);
        switch Band
            case 'Wideband'
                H_omni_MIMO = fft(h_omni_MIMO,[],2);
                figure,mesh((freq(1):delta_f:freq(2))*1e-6,1:size(H_omni_MIMO(:,:,1,2),1),10*log10(abs(H_omni_MIMO(:,:,1,2))))
                xlabel('Frequency [MHz]')
                ylabel('Snapshots')
                zlabel('Power [dB]')
                title('Frequency response for the channel between antenna 1 at Rx side and antenna 2 at Tx side ')
            case 'Narrowband'
                figure,plot(1:size(h_omni_MIMO,1),10*log10(abs(h_omni_MIMO(:,1,1)))) %figure,plot(1:size(h_omni_MIMO,1),10*log10(abs(h_omni_MIMO(:,1,2))))
                xlabel('Snapshots')
                ylabel('Power [dB]')
                title('Impulse response for the SISO channel')
        end
        %%%%%%%%%%%%%%%%
    case 'Multiple'
        %%%%%%%%%%%%%%%%
        % Channel transfer function with 128 omni-directional at the
        % BS, with lambda/2 inter-element separation, and two MSs
        % with one omni-directional antenna each
                                                                                % MEYSAM: Here we can extend it to more than just 2 MSs by manually defining new users. For example, if we have 3 users then define
                                                                                % h_omni_MIMO_Link3 = create_IR_omni_MIMO_VLA(link(3),freq,delta_f,Band); and also
                                                                                % H_omni_MIMO_Link3 = fft(h_omni_MIMO_Link3,[],2); The same applies when we have more users
        delta_f = (freq(2)-freq(1))/256;
        h_omni_MIMO_Link1 = create_IR_omni_MIMO_VLA(link(1),freq,delta_f,Band);
        h_omni_MIMO_Link2 = create_IR_omni_MIMO_VLA(link(2),freq,delta_f,Band);
        switch Band
            case 'Wideband'
                H_omni_MIMO_Link1 = fft(h_omni_MIMO_Link1,[],2);
                H_omni_MIMO_Link2 = fft(h_omni_MIMO_Link2,[],2);
                
                figure;
                subplot(1,2,1);
                mesh(1:BSPosNum, (freq(1):delta_f:freq(2))*1e-6, log10(abs(squeeze(H_omni_MIMO_Link1(1, :, 1, :))).^2));
                xlabel('Base station antennas')
                ylabel('Frequency [MHz]')
                zlabel('Power [dB]')
                title('User #1, Snapshot #1');
                axis square;
                subplot(1,2,2);
                mesh(1:BSPosNum, (freq(1):delta_f:freq(2))*1e-6, log10(abs(squeeze(H_omni_MIMO_Link2(1, :, 1, :))).^2));
                xlabel('Base station antennas')
                ylabel('Frequency [MHz]')
                zlabel('Power [dB]')
                title('User #2, Snapshot #1');
                axis square;
                
                figure;
                subplot(1,2,1);
                y1 = pow2db(squeeze(mean(mean(abs(H_omni_MIMO_Link1).^2, 1), 2)));
                plot(1:BSPosNum, y1);
                xlabel('Base station antennas')
                ylabel('Power [dB]')
                title('User #1, Snapshot #1');
                axis tight;
                subplot(1,2,2);
                y2 = pow2db(squeeze(mean(mean(abs(H_omni_MIMO_Link2).^2, 1), 2)));
                plot(1:BSPosNum, y2);
                xlabel('Base station antennas')
                ylabel('Power [dB]')
                title('User #2, Snapshot #1');
                axis tight;
                
                figure;
                subplot(1,2,1);
                y1 = pow2db(squeeze(mean(mean(abs(H_omni_MIMO_Link1).^2, 1), 4)));
                plot((freq(1):delta_f:freq(2))*1e-6, y1);
                xlabel('Frequency [MHz]')
                ylabel('Power [dB]')
                title('User #1, Snapshot #1');
                axis tight;
                subplot(1,2,2);
                y2 = pow2db(squeeze(mean(mean(abs(H_omni_MIMO_Link2).^2, 1), 4)));
                plot((freq(1):delta_f:freq(2))*1e-6, y2);
                xlabel('Frequency [MHz]')
                ylabel('Power [dB]')
                title('User #2, Snapshot #1');
                axis tight;
            case 'Narrowband'
                H_omni_MIMO_Link1 = fft(h_omni_MIMO_Link1,[],2);
                H_omni_MIMO_Link2 = fft(h_omni_MIMO_Link2,[],2);
                
                figure;
                subplot(1,2,1);
                y1 = pow2db(squeeze(mean(mean(abs(H_omni_MIMO_Link1).^2, 1), 2)));
                plot(1:BSPosNum, y1);
                xlabel('Base station antennas')
                ylabel('Power [dB]')
                title('User #1, Snapshot #1');
                axis tight;
                subplot(1,2,2);
                y2 = pow2db(squeeze(mean(mean(abs(H_omni_MIMO_Link2).^2, 1), 2)));
                plot(1:BSPosNum, y2);
                xlabel('Base station antennas')
                ylabel('Power [dB]')
                title('User #2, Snapshot #1');
                axis tight;
        end
end






























































