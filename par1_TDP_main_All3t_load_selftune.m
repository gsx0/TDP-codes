% 
clear;clc;
addpath('liblinear-1.94/matlab');
addpath('../');
addpath('../data/');
% Training Process.
lambda = 1; 
gamma =[10 10];
miu = [0.2 0.1];
beta = 0.1;
Epoch = 10;
load('../data/V_F_2t_par1.mat');
T = size(Vtr,2);

% Here Fai,W,b: cell(1,3) e.g. Fai{1},num_sampxK W{1}, KxC and b{1} Cx1
%[W, b, Fai] = Optimized_T_Task(Vtr,Ftr,lambda,gamma,miu,beta,Epoch);
%save('TDP_M.mat','W','b','Fai','-v7.3');
load('./par1_Vd19_mTDP_WbFai5.mat');
epoch_ts = 10;
clsnum = length(b{1});
lambda_ts = gamma;
clear Vtr;
tr_fea = Fai;
clear Fai;
lambda = [300 100];
for t = 1: T
% testing, single task
num_ts = size(Vts(:,t),1);
K = size(Vts{1,t},1);
N = size(Vts{1,t},2);
ts_fea{t} = zeros(num_ts,K);
for i = 1:num_ts
    j = 1;
    fai_ts = inv(Vts{i,t}*Vts{i,t}' + lambda_ts(t)*eye(K,K))*Vts{i,t}*ones(N,1); 
    while j <= epoch_ts
      % calculate Fts
      fts = fai_ts'*W{t} + b{t}';
      [~,idx] = max(fts);
      fts =zeros(1,clsnum);
      fts(idx) = 1;
      % calculate fai_ts.
      fai_ts =  inv(W{t}*(W{t})'+lambda(t)*Vts{i,t}*(Vts{i,t})' +lambda(t)*gamma(t)*eye(K,K))*(W{t}*(fts'-b{t})+lambda(t)*Vts{i,t}*ones(N,1));
      j = j + 1;  
    end
    ts_fea{t}(i,:) = fai_ts';
end
tr_fea{t} = tr_fea{t}./repmat(sqrt(sum(tr_fea{t}.*tr_fea{t},2)),1,K);
ts_fea{t} = ts_fea{t}./repmat(sqrt(sum(ts_fea{t}.*ts_fea{t},2)),1,K);
end
[~,tr_label] = max(Ftr,[],2);
[~,ts_label] = max(Fts,[],2);

%save('par1_all3tasks_withL2norm_joint_tdp_trts_fea_label.mat','tr_fea','ts_fea','tr_label','ts_label','-v7.3');

tr_fea = cell2mat(tr_fea);
ts_fea = cell2mat(ts_fea);

%tr_fea = tr_fea./repmat(sqrt(sum(tr_fea.*tr_fea,2)),1,K);
%ts_fea = ts_fea./repmat(sqrt(sum(ts_fea.*ts_fea,2)),1,K);

%[~,tr_label] = max(Ftr,[],2);
%[~,ts_label] = max(Fts,[],2);

% save('par1_task1_tdp_trts_fea_label.mat','tr_fea','ts_fea','tr_label','ts_label','-v7.3');
% Linear SVM training
cc = [1 10 100 1000];
for s = 1:length(cc)
options = ['-c ' num2str(cc(s))];
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
fprintf('Arage Class accuracy for mit67 SVM-C:%d: %f\n',cc(s),accuracy1 );

accuracy2 = length(find(ts_label == C))/length(C);
fprintf('Arage Classification accuracy for mit67 SVM-C:%d: %f\n',cc(s),accuracy2 );
accuracy(1,s) = accuracy1;
accuracy(2,s) = accuracy2;
end
