% -------------------------------------------------------------------------
% LOAD THE DATA AND PREPARE TRAINING AND TEST DATASETS
% -------------------------------------------------------------------------

load('TrainingData.mat');
load('TestData.mat');
load('best_model.mat');

% split the test data in to input, target and output
test_features = Test{:,1:9}';
test_target = Test{:,10:11}';
classlabel = Test{:,12}';

% BEST MODEL 

best_model = FinalValue(FinalValue.ErrorValue == min_Error, :)
best_learning_rate = best_model{:,2};
best_momentum = best_model{:,1};
best_hidden = best_model{:,3}; 

net = patternnet(best_hidden,'traingdx');
net.trainParam.lr = best_learning_rate
net.trainParam.mc = best_momentum 

% As indices for train and validation, last cv indices
% generated are used and indices for test are the ones in the
% final, corresponding to the intial split values that are
% still unseen by our model

% train the model and observe the result
net = train(net,x',y');
pred = net(test_features);
error_test = perform(net,test_target,pred);
plotconfusion(test_target,pred) % find the indices of maximum probabilities
plotroc(test_target,pred)
[~, pred] = max(pred);  

accuracy=(sum(classlabel == pred) / length(classlabel))*100;

final_accuracy = accuracy
final_error = error_test









 
