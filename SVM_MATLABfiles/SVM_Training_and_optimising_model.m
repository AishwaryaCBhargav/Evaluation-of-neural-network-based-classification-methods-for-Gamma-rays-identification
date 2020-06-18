% read the dataset
data = readtable('normdata2.xls');
summary(data);

% split the dataset into test and training&validation datasets using
% Randperm
rng(3);
[M,N] = size(data);
P = 0.70;
Rand = randperm(M);
Training_and_validation = data(Rand(1:round(P*M)),:); 
Test = data(Rand(round(P*M)+1:end),:);

% split the training&validation dataset into target and features
training_and_validation_target = table2array(Training_and_validation(:,end));
training_and_validation_features = table2array(Training_and_validation(:,1:9));

%  split the test dataset into target and features
test_target = table2array(Test(:,end));
test_features = table2array(Test(:,1:9)); 

% calculate summary stats
GroupStats_AllData = grpstats(data,'class',{'mean','std',@skewness});
GroupStats_TrainingData = grpstats(Training_and_validation,'class', ...
                          {'mean','std',@skewness});
GroupStats_TestData = grpstats(Test,'class',{'mean','std',@skewness});


% Applying Grid search with different values of hyperparameters and running them over 10 loops for 10-fold cross validation 

%listing different hyperparameters for SVM model
KernelFunction = ["linear", "gaussian"];
BoxConstraint = [4,5,6];
polynomial_order = [2,3,4];

%conducting grid search and 10-fold cross validation to tune
%hyperparameters

tic; %to calculate time elapsed
for i = 1:length(KernelFunction)
    for j = 1:length(BoxConstraint)
        
        classLoss1 = 0;
            
        for k = 1:10
            %dividing the training&validation dataset into 70% training and 30% validation data
            rng(6);
            [o,n] = size(Training_and_validation);
            p = 0.70;
            rand = randperm(o);
            Train_features = training_and_validation_features(rand(1:round(p*o)),:);
            Train_target = training_and_validation_target(rand(1:round(p*o)),:);
            Val_features = training_and_validation_features(rand(round(p*o)+1):end,:);
            Val_target = training_and_validation_target(rand(round(p*o)+1):end,:);
            
            %fitting the SVM model on the training data with different
            %hyperparameters and predicting the performance on validation
            %data
             
            SVMModel1 = fitcsvm(Train_features, Train_target,'BoxConstraint' , BoxConstraint(j), 'KernelFunction',KernelFunction(i),'Standardize',false,'ClassNames',{'1','2'}) ;
            predictedGroups1 = predict(SVMModel1,Val_features)  ;

            %calculating SVM classification loss
            loss1 = resubLoss(SVMModel1);       
            classLoss1 = classLoss1 + loss1;
        end
    
        %calculating mean loss for 10 runs of each combination of
        %hyperparameters
        Mean_loss1 = classLoss1/10;
        disp([num2str(Mean_loss1),' - ' ,  'Loss for BoxConstraint - ', BoxConstraint(j), ' and KernelFunction - ', KernelFunction(i)]);
    end
end
toc; %for time elapsed


%%
tic; %to calculate time elapsed
for m = 1:length(BoxConstraint)
    for l = 1:length(polynomial_order)
        classLoss2 = 0;
        
        for k = 1:10
            %dividing the training&validation dataset into 70% training and 30% validation data
            rng(6);
            [o,n] = size(Training_and_validation);
            p = 0.70;
            rand = randperm(o);
            Train_features = training_and_validation_features(rand(1:round(p*o)),:);
            Train_target = training_and_validation_target(rand(1:round(p*o)),:);
            Val_features = training_and_validation_features(rand(round(p*o)+1):end,:);
            Val_target = training_and_validation_target(rand(round(p*o)+1):end,:);

            SVMModel2 = fitcsvm(Train_features, Train_target,'BoxConstraint' , BoxConstraint(m), 'KernelFunction', 'polynomial', 'PolynomialOrder', polynomial_order(l),'Standardize',false,'ClassNames',{'1','2'}) ;           
            predictedGroups2 = predict(SVMModel2,Val_features)  ;
           
            %calculating SVM classification loss                
            loss2 = resubLoss(SVMModel2);                  
            classLoss2 = classLoss2 + loss2;
        end
    
        %calculating mean loss for 10 runs of each combination of hyperparameters 
        Mean_loss2 = classLoss2/10;
        disp([num2str(Mean_loss2),' - ' ,  'Loss for BoxConstraint - ', num2str(BoxConstraint(m)), ' and polynomial order - ', num2str(polynomial_order(l))]);
    end   
end
toc;

tic;
%  Applying the SVM model with optimised hyperparameters (Polynomial kernel of order 4 and Box constraint of 6) on Validation data set
SVMModel = fitcsvm(Train_features, Train_target,'BoxConstraint' , 6, 'KernelFunction', 'polynomial', 'polynomialorder', 4, 'Standardize',false,'ClassNames',{'1','2'}) ;
predictedGroups1 = predict(SVMModel,Val_features)  ;

%Measuring loss of SVM model on Validation data
loss1 = resubLoss(SVMModel);
disp(['Loss of SVM model on Validation data is: ', num2str(loss1)]);

% Measuring accuracy of SVM model on Validation data
Val_target1 = cellstr(num2str(Val_target));
Accuracy = classperf(Val_target1, predictedGroups1);
disp(['Accuracy of SVM model on Validation data is: ', num2str(Accuracy.CorrectRate)]);
toc;

%preparing ROC curve for SVM model and displaying 'area under curve'
[~,score_SVM] = resubPredict(SVMModel);
diffscore = score_SVM(:,1) - score_SVM(:,2);
[X,Y,T,AUCsvm,OPTROCPT,suby,subnames] = perfcurve(Train_target,diffscore,1);
disp([num2str(AUCsvm),' - Area Under Curve for SVM'])

%plotting ROC curve for SVM Model
plot(X,Y) 
legend('SVM','Location','Best');
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC Curve for SVM');

%changing data types for confusion matrix
predictedGroups_str = cellstr(predictedGroups1);

%preparing confusion matrix for SVM model
CM_SVM = confusionmat(Val_target1, predictedGroups_str)
Confusion_chart_SVM = confusionchart(CM_SVM)