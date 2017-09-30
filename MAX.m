clear;clc;
addpath('./liblinear-1.94/matlab');

% Training Process.
load('./mid_data/vd19_V_F_t1_par1.mat');
Vtr = Vtr1;
Vts = Vts1;
clear Vtr1 Vts1;


num_tr = size(Vtr,1);
num_ts = size(Vts,1);
K = size(Vts{1},1);
N = size(Vts{1},2);
tr_fea = zeros(num_tr,K);
ts_fea = zeros(num_ts,K);
tr_label = zeros(num_tr,1);
ts_label = zeros(num_ts,1);

for i = 1:num_tr
    tr_fea(i,:) = max(Vtr{i},[],2); 
end
for i = 1:num_ts
    ts_fea(i,:) = max(Vts{i},[],2); 
end

[~,tr_label] = max(Ftr,[],2);
[~,ts_label] = max(Fts,[],2);

%save('./mid_data/par3_task1_max_trts_fea_label.mat','tr_fea','ts_fea','tr_label','ts_label','-v7.3');

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
fprintf('Arage Class accuracy for Caltech101: %f\n',accuracy1 );

accuracy2 = length(find(ts_label == C))/length(C);
fprintf('Arage Classification accuracy for Caltech101: %f\n',accuracy2 );

