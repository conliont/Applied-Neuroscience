clear all; close all; clc

%% Load data, plots and reshape  

load("EEGdata.mat")

% Reshape my data into a 2d matrix 
% where column:trial and row:timepoint
reshaped_data = reshape(EEGdata, [60,700*80]);

% transpose my data
reshaped_data=reshaped_data.'; 

%% Α1: Perform PCA and plot the output  
 
% Perform PCA
[coeff, score, latent] = pca(reshaped_data); 

% Plot first 2 PCs
figure; 
plot(-coeff(:,1)); 
hold on; 
plot(-coeff(:,2),'r'); 
legend('PC1','PC2'); 
xlabel('Timepoints')
ylabel('PC value')
title('First 2 PCs (coeffs)')

% Plot data projected on first 2 PCs
figure; 
plot(score(:,1), score(:,2),'.','MarkerSize',1); 
axis equal;
xlabel('PC1'); 
ylabel('PC2');

% Plot eigenvalue magnitude
figure; 
bar(latent/sum(latent))
xlabel('Dimension'); 
ylabel('Eigenvalue magnitude');

% Plot variance explained as a function of number of PCs
% Plot proportion of variance explained by each PC
figure; 
plot(cumsum(latent)/sum(latent)*100); 
xlabel('Number of PCs'); % xlabel('Number of PCs')
ylabel('Proportion of variance explained'); % ylabel('Proportion of variance explained')

% Find the the number of PCs that are needed by using a threshold
PC_number = find(cumsum(latent)/sum(latent) > 0.9, 1);
fprintf("The number of PCs by a threshold of 0.9 is %d\n", PC_number);


%% Α2: Plot the first 2 PCs (coeffs) and their trial-averaged activations (scores) for each one of the 2 stimuli  

% Find the kind of stimulus on each trial (0=face, 1=car)
stim_0 = find(stim == 0);
stim_1 = find(stim == 1);

% Extract scores for each stimulus 
stim_0_score = score(stim_0, 1:2); 
stim_1_score = score(stim_1, 1:2);

% Find trial-averaged activations (scores) for each one of the 2 stimuli
stim_0_score_mean = mean(stim_0_score,2);
stim_1_score_mean = mean(stim_1_score,2);

% Plot first 2 PCs (same as before but with reversed coeffs)
plot(coeff(:,1:2))
xlabel('Timepoints')
ylabel('PC value')
legend('PC1', 'PC2')
title('First 2 PCs (coeffs)')

% plot the first 2 PCs (coeffs) for each one of the 2 stimuli
figure
subplot(2,2,1)
scatter(stim_0_score(:,1), stim_0_score(:,2),'b')
title('scores for stim 0: face')
xlabel('PC 1')
ylabel('PC 2')

subplot(2,2,2)
scatter(stim_1_score(:,1), stim_1_score(:,2),'r')
title('scores for stim 1: car')
xlabel('PC 1')
ylabel('PC 2')

% Plot the first 2 PCs for each stimulus in one diagramm
subplot(2,2,[3,4])
scatter(stim_0_score(:, 1), stim_0_score(:, 2), 'b'); 
hold on;
scatter(stim_1_score(:, 1), stim_1_score(:, 2), 'r'); 
title('scores for both stimulus')
xlabel('PC1'); 
ylabel('PC2'); 
legend('stim 0: face', 'stim 1: car');

% Reshape scores into trials
score_trials = reshape(score, [700, 80, 60]);

% Compute trial-averaged signals separately for each stimulus for PC1 and PC2
stimulus_0_scores = score_trials(:, stim_0, :);
stimulus_1_scores = score_trials(:, stim_1, :);
stimulus_0_av_pc1 = squeeze(mean(stimulus_0_scores(:, :, 1), 2));
stimulus_1_av_pc1 = squeeze(mean(stimulus_1_scores(:, :, 1), 2));
stimulus_0_av_pc2 = squeeze(mean(stimulus_0_scores(:, :, 2), 2));
stimulus_1_av_pc2 = squeeze(mean(stimulus_1_scores(:, :, 2), 2));

% Plot trial-averaged signals separately for each stimulus for PC1
figure;
subplot(2, 1, 1);
plot(stimulus_0_av_pc1, 'b');
hold on;
plot(stimulus_1_av_pc1, 'r');
xlabel('Timepoints');
ylabel('Amplitude');
legend('stim0: face', 'stim1: car');
title('Trial-Averaged Scores for PC1');

% Plot trial-averaged signals separately for each stimulus for PC2
subplot(2, 1, 2);
plot(stimulus_0_av_pc2, 'b');
hold on;
plot(stimulus_1_av_pc2, 'r');
xlabel('Timepoints');
ylabel('Amplitude');
legend('stim0: face', 'stim1: car');
title('Trial-Averaged Scores for PC2');


%% B1: classification performance Az in two time windows 

reshaped_data = reshape(EEGdata, [60,700*80]);

% Define time indices for the two time windows of interest
time_window_1 = 201:250;
time_window_2 = 451:500;

% Reshape data to match input format for single_trial_analysis
num_samples = length(time_window_1);

X1 = EEGdata(:, time_window_1, :);
X2 = EEGdata(:, time_window_2, :);

% Compute Az values for each time window
skipLOO = 0; % with leave-one-out cross-validation
duration=num_samples;

[Az1] = single_trial_analysis(X1, stim, duration, skipLOO);
[Az2] = single_trial_analysis(X2, stim, duration, skipLOO);

% Print Az values
fprintf('In time window [201, 250]ms the Az is: %.2f\n', Az1);
fprintf('In time window [451, 500]ms the Az is: %.2f\n', Az2);


%% B2: Az curve across time to illustrate how stimulus classification evolves over the duration of a 700ms trial

% Define some variables
window_size = 50; 
step_size = 1; 
num_windows = size(EEGdata, 2) - window_size;
timepoints = window_size+1:step_size:700;
skipLOO=0;

% Initialize the Az array to store Az values
Az_array=[];

% Loop through each window and compute Az
for k = 1:step_size:num_windows

    % Extract the data within the current window
    data_window = EEGdata(:, k:k+window_size-1,:);

    % Reshape the data for input to single_trial_analysis
    data_window=reshape(data_window,[size(data_window, 1),window_size*size(data_window, 3)]);    
    
    % Compute Az using single_trial_analysis
    Az = single_trial_analysis(data_window, stim, window_size, skipLOO);
    Az_array=[Az_array,Az];
end

% Find when the highest Az value occurs
[pks, locs] = findpeaks(Az_array);
[max_Az, index] = max(pks);
time = timepoints(locs(index));
fprintf('Best stimulus classification is at %dms\n', time);

% Plot the Az curve
figure;
plot(timepoints,Az_array);
xlabel('Timepoints (ms)');
ylabel('Az');
title('Classification performance over time');
text = sprintf('window size: %.f\nstep size: %.f\nbest time: %.fms', window_size, step_size, time);
legend(text,'Location','best');
