% --- fitnet_interaction_plot.m ---
% Generates a 2D contour plot (interaction surface plot)
% to address reviewer comment 3.8.

% --- 1. Load the Final Trained Model ---
if ~exist('final_regression_model.mat', 'file')
    error('final_regression_model.mat not found. Please run fitnet_final_plots.m first.');
end
load('final_regression_model.mat', 'net');
fprintf('Final regression model loaded.\n');

% --- 2. Define Variable Grids and Baseline ---
% Set the optimal baseline for the variable we will hold constant
optimal_ph = 7.5; % From your baseline in the last script

% Define the ranges for the two variables we want to interact
temp_range = 15:0.5:35;  % X-axis
sal_range  = 26:0.2:34;  % Y-axis

% Create a 2D grid of inputs
[T, S] = meshgrid(temp_range, sal_range);
% T is now a grid of Temp values
% S is now a grid of Salinity values

% --- 3. Prepare Input Matrix for the Model ---
% We need to "unroll" the grids into a list of inputs
% The model expects inputs in the order: Temp, Sal, pH
num_points = numel(T);
temp_col = T(:); % Unroll Temp grid to a column
sal_col  = S(:); % Unroll Salinity grid to a column
ph_col   = repmat(optimal_ph, num_points, 1); % Constant pH column

% Combine into the final input matrix (rows=samples, cols=features)
% This must match the original 'Predictor' format
model_inputs = [temp_col, sal_col, ph_col];

% --- 4. Get Model Predictions ---
fprintf('Generating interaction predictions...\n');
% Transpose inputs for the network
predicted_outputs = net(model_inputs');

% Get the first output (EPR)
predicted_epr = predicted_outputs(1, :);

% --- 5. Reshape Data for Plotting ---
% "Re-roll" the output vector back into a 2D grid
Z_epr = reshape(predicted_epr, size(T));

% --- 6. Plot the 2D Contour Plot ---
fprintf('Plotting interaction surface...\n');
figure;

% contourf creates a filled contour plot (a heat map)
[C, h] = contourf(T, S, Z_epr, 15); % 15 contour levels
clabel(C, h, 'FontSize', 10, 'Color', 'black'); % Add labels to the lines
h.LineWidth = 1.5;

% Add a color bar to show the EPR scale
colorbar;
c = colorbar;
c.Label.String = 'Predicted EPR (eggs/female/day)';
c.Label.FontSize = 12;

% Add labels and title
xlabel('Temperature (°C)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Salinity (PSU)', 'FontSize', 12, 'FontWeight', 'bold');
title('ANN Interaction Plot: Temp vs. Salinity (at pH 7.5)', 'FontSize', 14);

fprintf('Done. Save this figure for your manuscript (e.g., as Figure 10).\n');