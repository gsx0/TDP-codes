% 
clear;clc;
addpath('liblinear-1.94/matlab');
addpath('../');
% Training Process.
lambda = 1;
gamma =  10;
miu =0.05;
Epoch = 10;
% Calculate V,F
load('./mid_data/vd19_V_F_t1_par3.mat');
Vtr = Vtr1;
Vts = Vts1;
clear vtr1 Vts1;
%[W, b, Fai] = Optimized_Single_Task(Vtr,Ftr,lambda,gamma,miu,Epoch);
%save('./mid_data/T1_Initialize_for_Multi_task_par1.mat','W','b','Fai');
load('./mid_data/T1_Initialize_for_Multi_task_par3.mat');
% testing, single task
num_ts = size(Vts,1);
epoch_ts = 0;
clsnum = length(b);
K = size(Vts{1},1);
N = size(Vts{1},2);
ts_fea = zeros(num_ts,K);
lambda_ts = gamma;
lambda_all = [1 100 200 300 400 500];
%lambda_all = [1];
lamnum = length(lambda_all);
accuracy = zeros(1,lamnum);
for tt = 1:lamnum
lambda = lambda_all(tt);
for i = 1:num_ts
    j = 1;
    fai_ts = inv(Vts{i}*Vts{i}' + lambda_ts*eye(K,K))*Vts{i}*ones(N,1); 
     
    while j <= epoch_ts
      % calculate Fts
      fts = fai_ts'*W + b';
      [~,idx] = max(fts);
      fts = zeros(1,clsnum);
      fts(idx) = 1;
      % calculate fai_ts.
      fai_ts =  inv(W*W'+lambda*Vts{i}*(Vts{i})' +lambda*gamma*eye(K,K))*(W*(fts'-b)+lambda*Vts{i}*ones(N,1));
      j = j + 1;  
    end
    ts_fea(i,:) = fai_ts';
end
tr_fea = Fai;
[~,tr_label] = max(Ftr,[],2);
[~,ts_label] = max(Fts,[],2);
if length(lambda_all) == 1
save('./mid_data/self_tune_par3_task1_tdp_trts_fea_label.mat','tr_fea','ts_fea','tr_label','ts_label','-v7.3');
end
tr_fea = tr_fea./repmat(sqrt(sum(tr_fea.*tr_fea,2)),1,K);
ts_fea = ts_fea./repmat(sqrt(sum(ts_fea.*ts_fea,2)),1,K);
% Linear SVM training
c = 1;
options = ['-c ' num2str(c)];
model = train(double(tr_label), sparse(double(tr_fea)), options);
% SVM testing
[C] = predict(ts_label, sparse(double(ts_fea)), model);


 %------------Normalize the accuracy-------------------
class = unique(tr_label);
nclass = length(class);
acc = zeros(nclass, 1);
for jj = 1 : nclass,
     c = class(jj);
     idx = find(ts_label == c);
     curr_pred_label = C(idx);
     curr_gnd_label = ts_label(idx);
     acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
end
accuracy1 = mean(acc);
fprintf('Arage Class accuracy for Caltech101: lambda = %d, %f\n',lambda_all(tt),accuracy1 );

accuracy2 = length(find(ts_label == C))/length(C);
fprintf('Arage Classification accuracy for Caltech101:lambda = %d, %f\n',lambda_all(tt),accuracy2 );
accuracy(tt) = accuracy1;

end
