% --- fitnet_cv.m ---
% Solves an Input-Output Fitting problem with 10-fold Cross-Validation
% This script replaces your original 'fitnet_script.m'
% It assumes 'Predictor' (inputs) and 'Response' (targets) are loaded.

% --- Load Data ---
rng(0); % Sets the seed for random number generation
x = Predictor';
t = Response';

% --- Cross-Validation Setup ---
k = 10; % Number of folds
numSamples = size(x, 2);
c = cvpartition(numSamples, 'KFold', k); % Create 10-fold partition

% --- Storage for Test Results ---
% We will store the performance metric from each of the 10 folds
mse_scores = zeros(k, 1);
r2_scores = zeros(k, 1);

fprintf('Starting 10-Fold Cross-Validation for Regression (fitnet)...\n');

for i = 1:k
    fprintf('Fold %d/%d...\n', i, k);
    
    % --- Get Training and Test Indices for this Fold ---
    trainIdx = training(c, i); % Indices for training (90%)
    testIdx = test(c, i);      % Indices for testing (10%)
    
    % --- Select Training and Test Data ---
    xTrain = x(:, trainIdx);
    tTrain = t(:, trainIdx);  % <-- THIS IS THE CORRECTED LINE
    xTest = x(:, testIdx);
    tTest = t(:, testIdx);
    
    % --- Create a Fitting Network (MUST be re-created each loop) ---
    trainFcn = 'trainlm';  % Levenberg-Marquardt
    hiddenLayerSize = 10;
    net = fitnet(hiddenLayerSize, trainFcn);
    
    % --- Set Pre/Post-Processing (as in original script) ---
    net.input.processFcns = {'removeconstantrows', 'mapminmax'};
    net.output.processFcns = {'removeconstantrows', 'mapminmax'};
    
    % --- Setup Data Division (IMPORTANT) ---
    % We will let the 'train' function automatically partition our 90%
    % 'xTrain'/'tTrain' set into its own internal training and validation.
    % We will NOT use the test set for this.
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';
    net.divideParam.trainRatio = 85/100; % 85% of 90%
    net.divideParam.valRatio = 15/100;   % 15% of 90%
    net.divideParam.testRatio = 0/100;     % 0% - we have our own test set
    
    % --- Set Performance and Plot Functions (suppress plots) ---
    net.performFcn = 'mse';  % Mean Squared Error
    net.plotFcns = {};       % Suppress plot windows during loop
    
    % --- Train the Network ---
    % We train ONLY on the xTrain/tTrain data
    [net, tr] = train(net, xTrain, tTrain);
    
    % --- Test the Network ---
    % We test ONLY on the "unseen" xTest/tTest data
    yTest = net(xTest);
    
    % --- Store Performance Metrics for this Fold ---
    mse_scores(i) = perform(net, tTest, yTest); % Get Mean Squared Error
    
    % Calculate Overall R-squared for this fold
    % This correlates all predicted points vs. all actual points
    R = corr(tTest(:), yTest(:));
    r2_scores(i) = R^2;
    
end

fprintf('\n--- Cross-Validation for Regression Complete ---\n');

% --- Calculate Final Statistics ---
mean_mse = mean(mse_scores);
std_mse = std(mse_scores);

mean_rmse = mean(sqrt(mse_scores));
std_rmse = std(sqrt(mse_scores));

mean_r2 = mean(r2_scores);
std_r2 = std(r2_scores);

% --- Display Final Results ---
fprintf('Regression Model (fitnet) - 10-Fold CV Results\n');
fprintf('---------------------------------------------------\n');
fprintf('Mean Squared Error (MSE):   %.4f (± %.4f SD)\n', mean_mse, std_mse);
fprintf('Root Mean Squared (RMSE): %.4f (± %.4f SD)\n', mean_rmse, std_rmse);
fprintf('Overall R-squared (R2):     %.4f (± %.4f SD)\n', mean_r2, std_r2);
fprintf('---------------------------------------------------\n');
fprintf('Update Table 2 with these Mean (± SD) values.\n');