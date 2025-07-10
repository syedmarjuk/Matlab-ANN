% Solve a Pattern Recognition problem with a Neural Network
% Script adapted for classification from a fitting template.
% This script assumes 'Predictor' (inputs) and 'Response' (original outputs like EPR/NPR)
% variables are loaded in the workspace.

% --- User Data Loading Check ---
if ~exist('Predictor', 'var') || ~exist('Response', 'var')
    error('Please load your "Predictor" (inputs) and "Response" (e.g., EPR, NPR) data into the workspace before running this script.');
end

rng(0); % Sets the seed for random number generation for reproducibility

% --- Input Data (Features) ---
% Predictor should be a matrix where rows are samples and columns are features (Temp, Salinity, pH)
x = Predictor'; % Input data (features as rows, samples as columns)

% --- Target Data (Classes) ---
% This is the CRUCIAL step for pattern recognition.
% You need to convert your continuous 'Response' data (EPR, NPR) into categorical class labels.
% Then, these labels must be converted to one-hot encoded vectors for 'patternnet'.

% Example: Define classes based on the first column of 'Response' (e.g., EPR)
% PLEASE MODIFY THIS SECTION ACCORDING TO YOUR RESEARCH NEEDS
% -------------------------------------------------------------------------
if size(Response, 2) > 0
    output_to_classify = Response(:,1); % Example: using the first output (e.g., EPR)
else
    error('The "Response" variable is empty or has no columns to derive classes from.');
end

% Define your class thresholds. This is just an illustrative example.
% You should define these based on your specific criteria for "Low", "Medium", "High" production, etc.
% Let's say you want 3 classes.
numClasses = 3;
minValue = min(output_to_classify);
maxValue = max(output_to_classify);
threshold1 = minValue + (maxValue - minValue) / 3;
threshold2 = minValue + 2 * (maxValue - minValue) / 3;

class_indices = ones(size(output_to_classify)); % Initialize all to class 1
class_indices(output_to_classify >= threshold1) = 2; % Class 2
class_indices(output_to_classify >= threshold2) = 3; % Class 3

% Ensure class_indices are row vector for ind2vec
if size(class_indices, 1) > 1 && size(class_indices, 2) == 1
    class_indices = class_indices'; % Transpose if it's a column vector
end

if isempty(class_indices)
    error('Class indices are empty. Check your classification logic and Response data.');
end
if min(class_indices) < 1 || max(class_indices) > numClasses
    warning('Class indices might not cover all defined classes or are out of expected range.');
end

% Convert class indices to one-hot encoded target matrix
% t will be a numClasses x numSamples matrix
t = ind2vec(class_indices, numClasses);
% If ind2vec is not available (older toolboxes), or for more control:
% numSamples = size(x, 2);
% t = zeros(numClasses, numSamples);
% for i = 1:numSamples
%     t(class_indices(i), i) = 1;
% end
% -------------------------------------------------------------------------
% End of User-Defined Classification Logic

fprintf('Data prepared: %d samples, %d features, %d classes.\n', size(x,2), size(x,1), numClasses);
if size(t,2) ~= size(x,2)
    error('Mismatch in the number of samples between inputs (x) and targets (t). Check data preparation.');
end


% --- Choose a Training Function ---
% 'trainlm' is Levenberg-Marquardt (can be fast, but memory intensive for very large datasets).
% 'trainscg' (Scaled Conjugate Gradient) is often recommended for patternnet, good for large problems.
% 'trainrp' (Resilient Backpropagation) is another good option.
trainFcn = 'trainlm';  % Using Levenberg-Marquardt as per your initial context
% trainFcn = 'trainscg'; % Alternative good choice for pattern recognition

% --- Create a Pattern Recognition Network ---
hiddenLayerSize = 10; % Number of neurons in the hidden layer. Tune this parameter.
net = patternnet(hiddenLayerSize, trainFcn);

% --- Choose Input Pre/Post-Processing Functions ---
% Default for patternnet typically includes mapminmax for inputs.
% Output processing is handled by the softmax transfer function in the output layer for patternnet.
net.input.processFcns = {'removeconstantrows', 'mapminmax'};

% --- Setup Division of Data for Training, Validation, Testing ---
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% --- Choose a Performance Function ---
net.performFcn = 'crossentropy';  % Cross-Entropy is typical for classification

% --- Choose Plot Functions ---
% These plots will be generated during and after training.
net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
                'plotconfusion', 'plotroc'}; % Added plotconfusion and plotroc

% --- Train the Network ---
fprintf('Starting network training...\n');
% [net,tr] = train(net,x,t, 'useGPU', 'yes'); % Uncomment to use GPU if available
[net,tr] = train(net,x,t); % tr is the training record
fprintf('Network training complete.\n');

% --- Test the Network ---
% Get the network's predictions (outputs are probabilities for each class)
y_probabilities = net(x);

% Convert probabilities to class indices
predicted_classes_idx = vec2ind(y_probabilities);
actual_classes_idx    = vec2ind(t); % Convert one-hot actual targets to indices

% --- Performance Evaluation ---
% Overall Accuracy
accuracy = sum(actual_classes_idx == predicted_classes_idx) / numel(actual_classes_idx);
fprintf('\n----------------------------------------------------\n');
fprintf('      Neural Network Classification Performance   \n');
fprintf('----------------------------------------------------\n');
fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);

% The 'plotconfusion' function, if called during training or manually,
% provides a detailed confusion matrix.
% Example of how to display confusion matrix values for different sets:

% Get indices for each dataset from the training record
trainInd = tr.trainInd;
valInd = tr.valInd;
testInd = tr.testInd;

% Training Data Performance
train_targets_actual_idx = actual_classes_idx(trainInd);
train_predictions_idx = predicted_classes_idx(trainInd);
train_accuracy = sum(train_targets_actual_idx == train_predictions_idx) / numel(train_targets_actual_idx);
fprintf('Training Accuracy: %.2f%%\n', train_accuracy * 100);

% Validation Data Performance
if ~isempty(valInd)
    val_targets_actual_idx = actual_classes_idx(valInd);
    val_predictions_idx = predicted_classes_idx(valInd);
    val_accuracy = sum(val_targets_actual_idx == val_predictions_idx) / numel(val_targets_actual_idx);
    fprintf('Validation Accuracy: %.2f%%\n', val_accuracy * 100);
else
    fprintf('No validation data.\n');
end

% Testing Data Performance
if ~isempty(testInd)
    test_targets_actual_idx = actual_classes_idx(testInd);
    test_predictions_idx = predicted_classes_idx(testInd);
    test_accuracy = sum(test_targets_actual_idx == test_predictions_idx) / numel(test_targets_actual_idx);
    fprintf('Testing Accuracy: %.2f%%\n', test_accuracy * 100);
else
    fprintf('No testing data.\n');
end
fprintf('----------------------------------------------------\n');

% You can generate specific confusion matrices:
% figure;
% plotconfusion(ind2vec(actual_classes_idx(trainInd),numClasses), ind2vec(predicted_classes_idx(trainInd),numClasses), 'Training');
% figure;
% plotconfusion(ind2vec(actual_classes_idx(valInd),numClasses), ind2vec(predicted_classes_idx(valInd),numClasses), 'Validation');
% figure;
% plotconfusion(ind2vec(actual_classes_idx(testInd),numClasses), ind2vec(predicted_classes_idx(testInd),numClasses), 'Testing');
figure;
plotconfusion(t, y_probabilities, 'All Data'); % Overall confusion matrix
title('Confusion Matrix (All Data)');

% View the Network object
% view(net)

% Plots
% The plot functions specified in net.plotFcns will automatically display.
% To regenerate them later or save them:
% figure, plotperform(tr)
% title('Network Training Performance (Cross-Entropy)')
% figure, plottrainstate(tr)
% title('Network Training State')
% errors = gsubtract(t,y_probabilities); % Element-wise errors if needed for errhist
% figure, ploterrhist(errors)
% title('Error Histogram')
% figure, plotroc(t, y_probabilities)
% title('ROC Curves')

% --- Deployment Options (generated by Neural Pattern Recognition app, initially false) ---
if (false)
    % Generate MATLAB function for neural network
    genFunction(net,'myPatternRecognitionFunction');
    y_deployed_probs = myPatternRecognitionFunction(x);
    y_deployed_classes = vec2ind(y_deployed_probs);
end
if (false)
    % Generate a matrix-only MATLAB function for code generation
    genFunction(net,'myPatternRecFunction_matrixOnly','MatrixOnly','yes');
    y_deployed_matrix_probs = myPatternRecFunction_matrixOnly(x);
    y_deployed_matrix_classes = vec2ind(y_deployed_matrix_probs);
end
if (false)
    % Generate a Simulink diagram
    gensim(net);
end

disp(char(10));
disp('Script execution finished.');
disp('Check plots for visual performance assessment (e.g., Confusion Matrix, ROC curves).');
disp('Accuracy metrics are displayed above.');