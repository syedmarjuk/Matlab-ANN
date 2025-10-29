% --- fitnet_final_plots.m ---
% Script to generate FINAL plots for the manuscript.
% This model is trained on 100% of the data AFTER cross-validation
% has already proven the model architecture is robust.

rng(0); % Use the same seed for reproducibility
x = Predictor';
t = Response';

% --- Create a Fitting Network ---
trainFcn = 'trainlm';  % Levenberg-Marquardt
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize, trainFcn);

% --- Set Pre/Post-Processing ---
net.input.processFcns = {'removeconstantrows', 'mapminmax'};
net.output.processFcns = {'removeconstantrows', 'mapminmax'};

% --- Setup Division of Data (MODIFIED FOR FINAL MODEL) ---
% We use 100% of the data for training and validation.
% We set TestRatio = 0 because the 10-fold CV script 
% already handled the independent testing.
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 0/100; % <-- CRITICAL CHANGE

% --- Set Performance and Plot Functions ---
net.performFcn = 'mse';  % Mean Squared Error
net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotfit'}; % <-- PLOTS ARE ON

fprintf('Training final regression model on 100%% of data...\n');
% Train the Network
[net, tr] = train(net, x, t);

% Test the Network (on all data)
y = net(x);
e = gsubtract(t, y);
performance = perform(net, t, y)

fprintf('--- Final Model Training Complete ---\n');
fprintf('New figures (Regression, Performance, etc.) have been generated.\n');
fprintf('Save these plots for your manuscript.\n');
fprintf('The R-value from the "All" plot is the new overall R.\n');

% View the Network
view(net)

% --- Save the final network for sensitivity analysis ---
save('final_regression_model.mat', 'net');
fprintf('Final regression model saved as final_regression_model.mat\n');