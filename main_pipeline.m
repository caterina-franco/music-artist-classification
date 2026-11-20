clear; clc

% ––––––––––––––––––––––––––––– AUDIO CLASSIFICATION PIPELINE –––––––––––––––––––––––––––––––

addpath(genpath(pwd)) 

% –––––––––––––––––––––––––––– NORMAL FILES ––––––––––––––––––––––––––––––

disp('––––––––––––––––––––––––––– ANALYSING THE FILES –––––––––––––––––––––––––––')
fprintf('\n')

% set a standard window (used to divide the audio signal into smaller segments) and step length (the interval between consecutive time windows)
stepLength = 0.01;
windowLength = 0.03;

% create two cell arrays, one with the artists folders and one with the test/train folders
folder1 = {'laufey'; 'simpleplan'; 'skz'};
folder2 = {'train'; 'test'};

% features extraction for every test and train folder
[chromaLaufeyTrainFeats, mfccLaufeyTrainFeats, LaufeyTrainFeats] = feature_extraction(folder1{1}, folder2{1}, 'mp3');
[chromaLaufeyTestFeats, mfccLaufeyTestFeats, LaufeyTestFeats] = feature_extraction(folder1{1}, folder2{2}, 'mp3');
[chromaSimplePlanTrainFeats, mfccSimplePlanTrainFeats, SimplePlanTrainFeats] = feature_extraction(folder1{2}, folder2{1}, 'mp3');
[chromaSimplePlanTestFeats, mfccSimplePlanTestFeats, SimplePlanTestFeats] = feature_extraction(folder1{2}, folder2{2}, 'mp3');
[chromaSKZTrainFeats, mfccSKZTrainFeats, SKZTrainFeats] = feature_extraction(folder1{3}, folder2{1}, 'mp3');
[chromaSKZTestFeats, mfccSKZTestFeats, SKZTestFeats] = feature_extraction(folder1{3}, folder2{2}, 'mp3');

% labels extraction for every test and train folder
[all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3] = label_extraction(chromaLaufeyTrainFeats, chromaLaufeyTestFeats, chromaSimplePlanTrainFeats, chromaSimplePlanTestFeats, chromaSKZTrainFeats, chromaSKZTestFeats);
[recognitionRate1, title1] = KNN_confusionMatrix(all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3, 'CHROMA', '-pg');

[all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3] = label_extraction(mfccLaufeyTrainFeats, mfccLaufeyTestFeats, mfccSimplePlanTrainFeats, mfccSimplePlanTestFeats, mfccSKZTrainFeats, mfccSKZTestFeats);
[recognitionRate2, title2] = KNN_confusionMatrix(all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3, 'MFCCs', '-pm');

[all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3] = label_extraction(LaufeyTrainFeats, LaufeyTestFeats, SimplePlanTrainFeats, SimplePlanTestFeats, SKZTrainFeats, SKZTestFeats);
[recognitionRate3, title3] = KNN_confusionMatrix(all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3, 'GENERAL', '-pr');

[recognitionRates, index] = max([recognitionRate1, recognitionRate2, recognitionRate3]);

% find the best K-Nearest Neighbors (KNN)
if index == 1
    bestKNN = 'CHROMA';
elseif index == 2
    bestKNN = 'MFCCs';
else
    bestKNN = 'GENERAL';
end

% –––––––––––––––––––––––– NOISE-DISTURBED FILES –––––––––––––––––––––––––

disp('–––––––––––––––––––––– ANALYSING THE DISTURBED FILES ––––––––––––––––––––––')
fprintf('\n')

% chose an audio file with the noise 
noise = 'babble.wav';

% create two cell arrays, one with the artists folders and one with the test/train folders
folder1 = {'laufey'; 'simpleplan'; 'skz'};
folder2 = {'train'; 'test'};

% add noise to the audio files and put them in two folders created specifically
for i = 1:3
    for j = 1:2
        noise_enhanced(folder1{i}, noise, folder2{j})
    end
end

% create a cell array with the the noise-disturbed files
folder2 = {'noisetrain'; 'noisetest'};

% best KNN extraction for every test and train folder
[LaufeyTrainFeats] = bestKNN_extraction(folder1{1}, folder2{1}, 'wav', index);
[LaufeyTestFeats] = bestKNN_extraction(folder1{1}, folder2{2}, 'wav', index);
[SimplePlanTrainFeats] = bestKNN_extraction(folder1{2}, folder2{1}, 'wav', index);
[SimplePlanTestFeats] = bestKNN_extraction(folder1{2}, folder2{2}, 'wav', index);
[SKZTrainFeats] = bestKNN_extraction(folder1{3}, folder2{1}, 'wav', index);
[SKZTestFeats] = bestKNN_extraction(folder1{3}, folder2{2}, 'wav', index);

% labels extraction, compute the KNN and confusion matrix and plot the results of the matrix on a colored graph 
[all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3] = label_extraction(LaufeyTrainFeats, LaufeyTestFeats, SimplePlanTrainFeats, SimplePlanTestFeats, SKZTrainFeats, SKZTestFeats);
KNN_confusionMatrix(all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3, bestKNN, '-pg');

% ––––––––––––––––––––––––––– ENHANCED FILES –––––––––––––––––––––––––––––

disp('–––––––––––––––––––––– ANALYSING THE ENHANCED FILES –––––––––––––––––––––––')
fprintf('\n')

% create two cell arrays, one with the artists folders and one with the enhanced files
folder1 = {'laufey'; 'simpleplan'; 'skz'};
folder2 = {'enhancedtrain'; 'enhancedtest'};

% best KNN extraction for every test and train folder
[LaufeyTrainFeats] = bestKNN_extraction(folder1{1}, folder2{1}, 'wav', index);
[LaufeyTestFeats] = bestKNN_extraction(folder1{1}, folder2{2}, 'wav', index);
[SimplePlanTrainFeats] = bestKNN_extraction(folder1{2}, folder2{1}, 'wav', index);
[SimplePlanTestFeats] = bestKNN_extraction(folder1{2}, folder2{2}, 'wav', index);
[SKZTrainFeats] = bestKNN_extraction(folder1{3}, folder2{1}, 'wav', index);
[SKZTestFeats] = bestKNN_extraction(folder1{3}, folder2{2}, 'wav', index);

% labels extraction, compute the KNN and confusion matrix and plot the results of the matrix on a colored graph 
[all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3] = label_extraction(chromaLaufeyTrainFeats, chromaLaufeyTestFeats, chromaSimplePlanTrainFeats, chromaSimplePlanTestFeats, chromaSKZTrainFeats, chromaSKZTestFeats);

KNN_confusionMatrix(all_Train, all_Test, all_Labels, ground_truth, testLabel1, testLabel2, testLabel3, bestKNN, '-pg');