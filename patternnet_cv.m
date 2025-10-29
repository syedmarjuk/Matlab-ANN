% --- patternnet_cv.m ---
% Solves a Pattern Recognition problem with 10-fold Cross-Validation
% This script replaces your original 'patternnet_script.m'
% It assumes 'Predictor' (inputs) and 'Response' (targets) are loaded.

rng(0); % Sets the seed

% --- 1. Prepare Data (exactly as in original script) ---
fprintf('Preparing data for classification...\n');
x = Predictor'; % Input data

% --- Target Data (Classes) Conversion ---
output_to_classify = Response(:,1); % Using the first output (e.g., EPR)
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
t = ind2vec(class_indices, numClasses); % One-hot encoded targets
fprintf('Data prepared: %d samples, %d features, %d classes.\n', size(x,2), size(x,1), numClasses);

% --- 2. Cross-Validation Setup ---
k = 10; % Number of folds
numSamples = size(x, 2);
% --- CHANGE 1: We now pass 'class_indices' to stratify the folds ---
c = cvpartition(class_indices, 'KFold', k); % Stratified 10-fold partition

% --- Storage for Test Results ---
test_accuracy_scores = zeros(k, 1);
macro_avg_auc_scores = zeros(k, 1);

fprintf('Starting 10-Fold Stratified Cross-Validation (patternnet)...\n');

for i = 1:k
    fprintf('Fold %d/%d...\n', i, k);
    
    % --- Get Training and Test Indices for this Fold ---
    trainIdx = training(c, i); % Indices for training (90%)
    testIdx = test(c, i);      % Indices for testing (10%)
    
    % --- Select Training and Test Data ---
    xTrain = x(:, trainIdx);
    tTrain = t(:, trainIdx);
    xTest = x(:, testIdx);
    tTest = t(:, testIdx);
    
    % --- Create a Pattern Recognition Network (re-create each loop) ---
    % --- CHANGE 2: Using 'trainscg' - better for classification ---
    trainFcn = 'trainscg'; % Scaled Conjugate Gradient
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % --- Set Pre-Processing (as in original script) ---
    net.input.processFcns = {'removeconstantrows', 'mapminmax'};
    
    % --- Setup Data Division (IMPORTANT) ---
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 0/100; % We use our own test set
    
    % --- Set Performance and Plot Functions (suppress plots) ---
    net.performFcn = 'crossentropy'; % 'trainscg' works with this
    net.plotFcns = {}; % Suppress plot windows during loop
    
    % --- Train the Network ---
    [net, tr] = train(net, xTrain, tTrain);
    
    % --- Test the Network ---
    yTest_probs = net(xTest); % Get probabilities
    yTest_idx = vec2ind(yTest_probs); % Get predicted class indices
    tTest_idx = vec2ind(tTest);       % Get actual class indices
    
    % --- Store Performance Metrics for this Fold ---
    
    % 1. Test Accuracy
    test_accuracy_scores(i) = sum(tTest_idx == yTest_idx) / numel(tTest_idx);
    
    % 2. Macro-Average AUC
    class_auc_scores = zeros(numClasses, 1);
    for c_idx = 1:numClasses
        true_labels_for_class = (tTest_idx == c_idx);
        scores_for_class = yTest_probs(c_idx, :);
        
        % Because we are stratified, this should now work
        [~,~,~,auc] = perfcurve(true_labels_for_class, scores_for_class, 1);
        class_auc_scores(c_idx) = auc;
    end
    macro_avg_auc_scores(i) = mean(class_auc_scores);
    
end

fprintf('\n--- Cross-Validation for Classification Complete ---\n');

% --- Calculate Final Statistics ---
mean_accuracy = mean(test_accuracy_scores);
std_accuracy = std(test_accuracy_scores);

mean_macro_auc = mean(macro_avg_auc_scores);
std_macro_auc = std(macro_avg_auc_scores);

% --- Display Final Results ---
fprintf('Classification Model (patternnet) - 10-Fold CV Results\n');
fprintf('---------------------------------------------------\n');
fprintf('Mean Test Accuracy:     %.4f (± %.4f SD)\n', mean_accuracy, std_accuracy);
fprintf('Mean Macro-Average AUC: %.4f (± %.4f SD)\n', mean_macro_auc, std_macro_auc);
fprintf('---------------------------------------------------\n');
fprintf('Update Table 3 with these Mean (± SD) values.\n');