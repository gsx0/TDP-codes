%Input: all training images, stored as Vi = cell(m,T), wherein,V{i,j} \in
% \mathbb{R}^{KixNj, e.g., 256x36}, here Ki is the dimension of the maps, and Nj =
% 6x6,8x8,10x10,and so on. X \in R^{mxK} is the concatenated matrix of all
% the xi,i = 1,2,...,m. 
% Note, V can not be too large, or-else, will be hard
% to be optimized. F \in R^{mxC}, C is category labels. 
% here different tasks t = 1...T share the same parameters of lambda,
% gamma, beta, and miu and the same Epoch in different task
function [TW, Tb, TFai] = Optimized_T_Task(V,F,lambda,gamma,miu,beta,Epoch)

num_samp =  size(V,1);
T = size(V,2);
K = zeros(T,1);
N = zeros(T,1);
C = size(F,2);
for t = 1:T
    K(t) = size(V{1,t},1);
    N(t) = size(V{1,t},2);
    TW{t} = zeros(K(t),C);
    Tb{t} = zeros(C,1);
    TFai{t} = zeros(num_samp,K(t));
end
H = eye(num_samp,num_samp)-ones(num_samp,1)*ones(num_samp,1)'/num_samp;
% Initialization
% while t <= T
%     epch = 1;
%     while epch<=10
%         % fix W{t},b{t} solve Fai{t}
%         for sam = 1:num_samp
%             Fai{t}(sam,:) = (inv(W{t}*W{t}'+lambda*V{sam,t}*V{sam,t}'+lambda*gamma*eye(K,K))*(W{t}*(F(sam,:)'-b)+lambda*V{sam,t}*ones(N(t),1)))';
%         end
%         % fix Fai{t}, solve W{t},b{t}
%         W{t} = inv(Fai{t}'*H*Fai{t}+miu(t)*eye(K,K))*Fai{t}'*H*F;
%         b{t} = (F-Fai{t}*W{t})'*ones(num_samp,1)/num_samp;
%     end
% end
% instead by the single task classifiers. and Fai.
for ii = 1:T
    load(sprintf('./mid_data/T%d_Initialize_for_Multi_task_par2.mat',ii));
    TFai{ii} = Fai; 
    TW{ii} = W;
    Tb{ii} = b;
end
% joint iteration
epch =1;
while epch <= Epoch
    W_ktc = [];
    for cc =1:T
        W_ktc = [W_ktc,TW{cc}];
    end
    
    for t =1:T
        % fix W{t},b{t} solve Fai{t}
        for sam = 1:num_samp
            TFai{t}(sam,:) = (inv(TW{t}*(TW{t})'+lambda*V{sam,t}*(V{sam,t})'... 
                        +lambda*gamma(t)*eye(K(t),K(t)))*(TW{t}*(F(sam,:)'-Tb{t})+lambda*V{sam,t}*ones(N(t),1)))';
        end
        % fix Fai{t}, solve W{t},b{t}
        % for cc =1:T
        %    W_ktc = [W_ktc,TW{cc}];
        % end
       TW{t} = inv((TFai{t})'*H*TFai{t}+miu(t)*eye(K(t),K(t))+beta*(inv((W_ktc*W_ktc')^(1/2)))/2)*TFai{t}'*H*F;
       Tb{t} = (F-TFai{t}*TW{t})'*ones(num_samp,1)/num_samp;
    
    end
% printf the loss of Obj.
loss = zeros(1,T);
for ss = 1:T
% printf loss
    T1 = TFai{ss}*TW{ss} + ones(num_samp,1)*Tb{ss}'-F;
    loss1 = sum((diag(T1*T1'))) + miu(ss) * sum((diag(TW{ss}*TW{ss}')));
    loss2 = 0;
    for i = 1:num_samp
        loss2  = loss2 + sum(((V{i,ss})'* TFai{ss}(i,:)'-ones(N(ss),1)).^2) + gamma(ss)*sum(TFai{ss}(i,:).^2);
    end
    loss(ss) = loss1 + lambda * loss2;    

end
loss_o = sum(diag(sqrt(W_ktc'*W_ktc)));
Loss = sum(loss) + beta*loss_o;
fprintf('Epoch %d has been done! Loss = %f\n',epch,Loss);
epch = epch + 1;

end


end
