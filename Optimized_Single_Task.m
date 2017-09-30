% Input: all training images, stored as V = cell(m,1), wherein,V{i} \in
% \mathbb{R}^{KxN, e.g., 256x36}, here K is the dimension of the maps, and N =
% 6x6,8x8,10x10,and so on.X \in R^{mxK} is the concatenated matrix of all
% the xi,i = 1,2,...,m. Note, V can not be too large, orelse, will be hard
% to be optimized. F \in R^{mxC}, C is category labels. 

function [W, b, Fai] = Optimized_Single_Task(V,F,lambda,gamma,miu,Epoch)
epc = 1;
num_samp =  length(V);
K = size(V{1},1);
N = size(V{1},2);
C = size(F,2);
W = zeros(K,C);b=zeros(C,1);
Fai = zeros(num_samp,K);
H = eye(num_samp)-ones(num_samp,1)*ones(num_samp,1)'/num_samp;

while epc<=Epoch
    % fix W,b solve Fai
    for sam = 1:num_samp
        Fai(sam,:) = (inv(W*W'+lambda*V{sam}*(V{sam})'+lambda*gamma*eye(K,K))*(W*(F(sam,:)'-b)+lambda*V{sam}*ones(N,1)))';
    end
    % fix Fai, solve W,b
    W = inv(Fai'*H*Fai+miu*eye(K,K))*Fai'*H*F;
    b = (F-Fai*W)'*ones(num_samp,1)/num_samp;
    % printf loss
    T1 = Fai*W + ones(num_samp,1)*b'-F;
    loss1 = sum((diag(T1*T1'))) + miu * sum((diag(W*W')));
    loss2 = 0;
    for i = 1:num_samp
        loss2  = loss2 + sum(((V{i})'* Fai(i,:)'-ones(N,1)).^2) + gamma*sum(Fai(i,:).^2);
    end
    loss = loss1 + lambda * loss2;
    
    fprintf('Epoch %d has been done! Loss = %f\n',epc,loss);
    epc = epc + 1;
end

end
