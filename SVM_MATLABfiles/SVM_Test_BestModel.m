% read the dataset
data = readtable('normdata2.xls');
summary(data);

% split the main dataset into test and training&validation datasets with the help of
% Randperm
rng(3);
[M,N] = size(data);
P = 0.70;
Rand = randperm(M);
Training_and_validation = data(Rand(1:round(P*M)),:); 
Test = data(Rand(round(P*M)+1:end),:);

% split the training_and_validation dataset into target and features
training_and_validation_target = table2array(Training_and_validation(:,end));
training_and_validation_features = table2array(Training_and_validation(:,1:9));

%  split the test dataset into target and features
test_target = table2array(Test(:,end));
test_features = table2array(Test(:,1:9)); 

% Split training_and_validation data into Training and Validation datasets
[m,n] = size(Training_and_validation);
p = 0.70;
rand = randperm(m);
Train_features = training_and_validation_features(rand(1:round(p*m)),:);
Train_target = training_and_validation_target(rand(1:round(p*m)),:);
Val_features = training_and_validation_features(rand(round(p*m)+1):end,:);
Val_target = training_and_validation_target(rand(round(p*m)+1):end,:);

tic; %measuring time taken to compute
%  Applying SVM model
SVMModel = fitcsvm(Train_features, Train_target,'BoxConstraint' , 6, 'KernelFunction', 'polynomial', 'polynomialorder', 4, 'Standardize',false,'ClassNames',{'1','2'}) ;
predictedGroups1 = predict(SVMModel,test_features)  ;

%Measuring loss of SVM model
loss1 = resubLoss(SVMModel);
disp(['Loss of SVM model on Test data is: ', num2str(loss1)]);

% Measuring accuracy of SVM model
test_target1 = cellstr(num2str(test_target));
Accuracy = classperf(test_target1, predictedGroups1);
disp(['Accuracy of SVM model on Test data is: ', num2str(Accuracy.CorrectRate)]);
toc;

%changing data type of predicted groups to plot confusion matrix
predictedGroups_str = cellstr(predictedGroups1);

%preparing confusion matrix for SVM model
CM_SVM = confusionmat(test_target1, predictedGroups_str)
Confusion_chart_SVM = confusionchart(CM_SVM)

%preparing ROC curve for SVM model and displaying 'area under curve'
[~,score_SVM] = resubPredict(SVMModel);
diffscore = score_SVM(:,1) - score_SVM(:,2);
[X,Y,T,AUCsvm,OPTROCPT,suby,subnames] = perfcurve(Train_target,diffscore,1);
disp([num2str(AUCsvm),' - Area Under Curve for SVM'])

%plotting ROC curve for SVM Model
plot(X,Y) 
legend('SVM','Location','Best')
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC Curve for SVM')