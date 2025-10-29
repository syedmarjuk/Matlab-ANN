% --- fitnet_sensitivity.m ---
% Performs a sensitivity analysis on the final trained model
% to address reviewer comment 3.5 (Model Interpretability).

% --- 1. Load the Final Trained Model ---
% Make sure you have run 'fitnet_final_plots.m' to create this file
if ~exist('final_regression_model.mat', 'file')
    error('final_regression_model.mat not found. Please run fitnet_final_plots.m first.');
end
load('final_regression_model.mat', 'net');
fprintf('Final regression model loaded.\n');

% --- 2. Define Optimal Baseline and Variable Ranges ---
% !!! PLEASE CHECK THESE VALUES !!!
% Use the optimal values from your experimental results (Figure 1)
baseline.temp = 21;     % Optimal Temp (e.g., 21°C)
baseline.sal = 30;      % Optimal Salinity (e.g., 30 PSU)
baseline.ph = 7.5;    % Optimal pH (e.g., 7.5)

% Define the full range for each variable to be tested
temp_range = 15:0.5:35;  % Test from 15 to 35 °C
sal_range  = 26:0.2:34;  % Test from 26 to 34 PSU
ph_range   = 6.5:0.1:9.0; % Test from 6.5 to 9.0 pH

% --- 3. Run Sensitivity Analysis ---

% A) Vary Temperature
fprintf('Analyzing sensitivity to Temperature...\n');
temp_inputs = [repmat(baseline.sal, length(temp_range), 1), ...
               repmat(baseline.ph, length(temp_range), 1), ...
               temp_range']'; % Note: Need to match original input order
% IMPORTANT: Check if 'Predictor' columns are Temp, Sal, pH. 
% If not, re-order the columns above. Let's assume:
% Col 1 = Temp, Col 2 = Sal, Col 3 = pH
temp_inputs = [temp_range', ...
               repmat(baseline.sal, length(temp_range), 1), ...
               repmat(baseline.ph, length(temp_range), 1)]';

% B) Vary Salinity
fprintf('Analyzing sensitivity to Salinity...\n');
sal_inputs = [repmat(baseline.temp, length(sal_range), 1), ...
              sal_range', ...
              repmat(baseline.ph, length(sal_range), 1)]';

% C) Vary pH
fprintf('Analyzing sensitivity to pH...\n');
ph_inputs = [repmat(baseline.temp, length(ph_range), 1), ...
             repmat(baseline.sal, length(ph_range), 1), ...
             ph_range']';
         
% --- 4. Get Model Predictions ---
% Use the trained network to predict the output for these new inputs
pred_vs_temp = net(temp_inputs);
pred_vs_sal  = net(sal_inputs);
pred_vs_ph   = net(ph_inputs);

% Assuming EPR is the first output, NPR is the second
% If you only have one output, just use pred_vs_temp(1,:)
epr_vs_temp = pred_vs_temp(1, :);
epr_vs_sal  = pred_vs_sal(1, :);
epr_vs_ph   = pred_vs_ph(1, :);

% --- 5. Plot the Results ---
fprintf('Plotting sensitivity analysis...\n');
figure;
sgtitle('ANN Model Sensitivity Analysis (Partial Dependence Plots)', 'FontSize', 14, 'FontWeight', 'bold');

% Plot 1: Temperature
subplot(1, 3, 1);
plot(temp_range, epr_vs_temp, 'r-', 'LineWidth', 2.5);
hold on;
xline(baseline.temp, 'k--', 'Label', 'Optimal Baseline');
title('Effect of Temperature', 'FontSize', 12);
xlabel('Temperature (°C)');
ylabel('Predicted EPR (eggs/female/day)');
grid on;

% Plot 2: Salinity
subplot(1, 3, 2);
plot(sal_range, epr_vs_sal, 'b-', 'LineWidth', 2.5);
hold on;
xline(baseline.sal, 'k--', 'Label', 'Optimal Baseline');
title('Effect of Salinity', 'FontSize', 12);
xlabel('Salinity (PSU)');
ylabel('Predicted EPR (eggs/female/day)');
grid on;

% Plot 3: pH
subplot(1, 3, 3);
plot(ph_range, epr_vs_ph, 'g-', 'LineWidth', 2.5);
hold on;
xline(baseline.ph, 'k--', 'Label', 'Optimal Baseline');
title('Effect of pH', 'FontSize', 12);
xlabel('pH');
ylabel('Predicted EPR (eggs/female/day)');
grid on;

set(gcf, 'Position', [100, 100, 1200, 400]); % Make the figure wide
fprintf('Done. Save this figure for your manuscript (e.g., as Figure 9).\n');