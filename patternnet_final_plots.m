% --- patternnet_final_plots.m ---
% Script to generate FINAL plots for the manuscript.
% This model is trained on 100% of the data AFTER cross-validation
% has already proven the model architecture is robust.

rng(0); % Use the same seed

% --- 1. Prepare Data (exactly as in original script) ---
fprintf('Preparing data for classification...\n');
x = Predictor'; % Input data
output_to_classify = Response(:,1);
numClasses = 3;
minValue = min(output_to_classify);
maxValue = max(output_to_classify);
threshold1 = minValue + (maxValue - minValue) / 3;
threshold2 = minValue + 2 * (maxValue - minValue) / 3;
class_indices = ones(size(output_to_classify));
class_indices(output_to_classify >= threshold1) = 2;
class_indices(output_to_classify >= threshold2) = 3;
if size(class_indices, 1) > 1 && size(class_indices, 2) == 1
    class_indices = class_indices';
end
t = ind2vec(class_indices, numClasses);
fprintf('Data prepared: %d samples, %d features, %d classes.\n', size(x,2), size(x,1), numClasses);

% --- Create a Pattern Recognition Network ---
trainFcn = 'trainlm';
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% --- Set Pre-Processing ---
net.input.processFcns = {'removeconstantrows', 'mapminmax'};

% --- Setup Division of Data (MODIFIED FOR FINAL MODEL) ---
% We use 100% of the data for training and validation.
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 0/100; % <-- CRITICAL CHANGE

% --- Set Performance and Plot Functions ---
net.performFcn = 'crossentropy';
net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
                'plotconfusion', 'plotroc'}; % <-- PLOTS ARE ON

fprintf('Training final classification model on 100%% of data...\n');
% Train the Network
[net, tr] = train(net, x, t);

% Test the Network (on all data)
y_probabilities = net(x);
predicted_classes_idx = vec2ind(y_probabilities);
actual_classes_idx = vec2ind(t);

% Overall Accuracy
accuracy = sum(actual_classes_idx == predicted_classes_idx) / numel(actual_classes_idx);
fprintf('--- Final Model Training Complete ---\n');
fprintf('New figures (Confusion Matrix, ROC, etc.) have been generated.\n');
fprintf('Save these plots for your manuscript.\n');
fprintf('Overall Accuracy (on all data): %.2f%%\n', accuracy * 100);