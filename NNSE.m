function [W,score,indx]=NNSE(X,c,alpha,beta,lambda, NITER, gamma, maxEpochs, learningRate, error_bp, error_diff_thresh, numberOfHiddenNeure)
%Reference to be updated: "Neural Networks Embedded Self-Expression with 
% Adaptive Graph for Unsupervised Feature Selection"
%
%Input:
%      X: d by n matrix, n samples with d dimensions.
%      c: the desired cluster number.
%      alpha, beta, lambda, gamma: parameters refer to paper.
%      NITER: the desired number of iteration.
%      maxEpochs: the maximum epochs for the back propagation neural network
%      learningRate: the initial learning rate (neural network)
%      error_bp: the error (MSE) threshold (neural network)
%      error_diff_thresh: the error (MSE) difference threshold (neural network)
%      numberOfHiddenNeure: the number of neurons in hidden layers
%Output:
%      W: d by c projection matrix.
%      score: d-dimensional vector, preserves the score for each dimensions.
%      indx: the sort of features for selection.

[d,n]=size(X);

Yp = orth(rand(n,c));
X_multi = X*X';
W = rand(d,c);
S = zeros(n,n);
iter=1;
while (iter< NITER)
    fprintf("iter=%d\n",iter);
    bigLambda = diag( 0.5./sqrt(sum(W.*W,2)+eps));
   
    for i=1:n
        for j=1:n
            S(i,j)=exp(-(alpha* norm(Yp(i,:)-Yp(j,:),2)^2 + 0.5*gamma* norm(W'*X(:, i)-W'*X(:, j),2)^2)/(2*beta));
        end
        S(i,:)=S(i,:)./sum(S(i,:));
    end
    S=(S+S')./2;
    D=diag(sum(S,2));
    Ls=D-S;
    
    temp = X_multi+lambda.*bigLambda+gamma.*X*Ls*X';
    W=real(X_multi/temp);
    
    A=alpha.*Ls + eye(size(Ls,1),size(Ls,2));
    
    [B,error_bp]=bpNN(gather(X), gather(Yp), maxEpochs, learningRate, error_bp, error_diff_thresh, numberOfHiddenNeure);
    
    Yp=gpi(A,B,1);
    
    iter = iter+1;
end
score=sum((W.*W),2);
[~,indx]=sort(score,'descend');

end


%To optimize the problem of E(phi(X))=||phi(X)-Yp||_F^2
function [B, error_bp]=bpNN(X, Yp, maxEpochs, learningRate, error_bp, error_diff_thresh, numberOfHiddenNeure)
% X  : d by n matrix, n samples with d dimensions.
% Yp : n by c matrix
% B  : n by c matrix

numberOfSample = size(X,2);
inputDimension = size(X,1);
outputDimension = size(Yp,2);

rand('state', sum(100*clock));

input = X;
output = Yp';

numNeurons = [numberOfHiddenNeure numberOfHiddenNeure];
numOutputDim = outputDimension;
unipolarOrBipolar = 0; %0 for Unipolar, -1 for Bipolar

numEpochsMax = maxEpochs;

resilientGradientDescent_flag = 1; %1 for enable, 0 for disable
learningRate_up = 1.2;
learningRate_down = 0.5;
deltas_start = 0.9;
deltas_min = 10^-6;
deltas_max = 50;

decreaseLearningRate_flag = 1; %1 for enable decreasing, 0 for disable
learningRateDecreaseValue = 0.0001;
minLearningRate = 0.0005;

learningRate_momentum_flag = 1; %1 for enable, 0 for disable
momentum_alpha = 0.05;

DataSamples = input';
TargetLabels = output;
ActualLabels = zeros(size(output,1), size(output,2));

numInputNodes = length(DataSamples(1,:));

numLayers = 2 + length(numNeurons);
numNodesEachLayer = [numInputNodes numNeurons numOutputDim];


numNodesEachLayer(1:end-1) = numNodesEachLayer(1:end-1) + 1;
DataSamples = [ones(length(DataSamples(:,1)),1) DataSamples];


TargetOutputs = TargetLabels';

Weights = cell(1, numLayers);
Delta_Weights = cell(1, numLayers);
ResilientDeltas = Delta_Weights;
for i = 1:length(Weights)-1
    Weights{i} = 2*rand(numNodesEachLayer(i), numNodesEachLayer(i+1))-1; 
    Weights{i}(:,1) = 0;
    Delta_Weights{i} = zeros(numNodesEachLayer(i), numNodesEachLayer(i+1));
    ResilientDeltas{i} = deltas_start*ones(numNodesEachLayer(i), numNodesEachLayer(i+1));
end
Weights{end} = ones(numNodesEachLayer(end), 1);
Old_Delta_Weights_for_Momentum = Delta_Weights;
Old_Delta_Weights_for_Resilient = Delta_Weights;

NodesActivations = cell(1, numLayers);
for i = 1:length(NodesActivations)
    NodesActivations{i} = zeros(1, numNodesEachLayer(i));
end
NodesBackPropagatedErrors = NodesActivations;

zeroRMSReached = 0;
nbrOfEpochs_done = 0;

MSE = -1 * ones(1,numEpochsMax);
for Epoch = 1:numEpochsMax
    
    for sample_i = 1:length(DataSamples(:,1))
       
        NodesActivations{1} = DataSamples(sample_i,:);
        for Layer = 2:numLayers
            NodesActivations{Layer} = NodesActivations{Layer-1}*Weights{Layer-1};
            NodesActivations{Layer} = ActivationFunction(NodesActivations{Layer}, unipolarOrBipolar);
            if (Layer ~= numLayers)
                NodesActivations{Layer}(1) = 1;
            end
        end
        
        NodesBackPropagatedErrors{numLayers} =  TargetOutputs(sample_i,:)-NodesActivations{numLayers};
        for Layer = numLayers-1:-1:1
            gradient = ActivationDrevFunction(NodesActivations{Layer+1}, unipolarOrBipolar);
            for node=1:length(NodesBackPropagatedErrors{Layer}) % For all the Nodes in current Layer
                NodesBackPropagatedErrors{Layer}(node) =  sum( NodesBackPropagatedErrors{Layer+1} .* gradient .* Weights{Layer}(node,:) );
            end
        end
        
        for Layer = numLayers:-1:2
            derivative = ActivationDrevFunction(NodesActivations{Layer}, unipolarOrBipolar);
            Delta_Weights{Layer-1} = Delta_Weights{Layer-1} + NodesActivations{Layer-1}' * (NodesBackPropagatedErrors{Layer} .* derivative);
        end
    end
    
    if (resilientGradientDescent_flag)
        if (mod(Epoch,200)==0)
            for Layer = 1:numLayers
                ResilientDeltas{Layer} = learningRate*Delta_Weights{Layer};
            end
        end
        for Layer = 1:numLayers-1
            mult = Old_Delta_Weights_for_Resilient{Layer} .* Delta_Weights{Layer};
            ResilientDeltas{Layer}(mult > 0) = ResilientDeltas{Layer}(mult > 0) * learningRate_up; % Sign didn't change
            ResilientDeltas{Layer}(mult < 0) = ResilientDeltas{Layer}(mult < 0) * learningRate_down; % Sign changed
            ResilientDeltas{Layer} = max(deltas_min, ResilientDeltas{Layer});
            ResilientDeltas{Layer} = min(deltas_max, ResilientDeltas{Layer});
            
            Old_Delta_Weights_for_Resilient{Layer} = Delta_Weights{Layer};
            
            Delta_Weights{Layer} = sign(Delta_Weights{Layer}) .* ResilientDeltas{Layer};
        end
    end
    if (learningRate_momentum_flag)
        for Layer = 1:numLayers
            Delta_Weights{Layer} = learningRate*Delta_Weights{Layer} + momentum_alpha*Old_Delta_Weights_for_Momentum{Layer};
        end
        Old_Delta_Weights_for_Momentum = Delta_Weights;
    end
    if (~learningRate_momentum_flag && ~resilientGradientDescent_flag)
        for Layer = 1:numLayers
            Delta_Weights{Layer} = learningRate * Delta_Weights{Layer};
        end
    end
    
    for Layer = 1:numLayers-1
        Weights{Layer} = Weights{Layer} + Delta_Weights{Layer};
    end
    
    for Layer = 1:length(Delta_Weights)
        Delta_Weights{Layer} = 0 * Delta_Weights{Layer};
    end
    
    if (decreaseLearningRate_flag)
        new_learningRate = learningRate - learningRateDecreaseValue;
        learningRate = max(minLearningRate, new_learningRate);
    end
    
    for sample_i = 1:length(DataSamples(:,1))
        output_per_sample = EvaluateNetworkFunction(DataSamples(sample_i,:), NodesActivations, Weights, unipolarOrBipolar);%c by 1 or 1 by c
        if size(output_per_sample,1)==1
            output_per_sample = output_per_sample';
        end
        ActualLabels(:,sample_i)=output_per_sample;
    end
    MSE(Epoch) = sum(sum((ActualLabels-TargetLabels).^2))/(length(DataSamples(:,1)));
    if (MSE(Epoch) == 0)
        zeroRMSReached = 1;
    end
    
    nbrOfEpochs_done = Epoch;
    if (zeroRMSReached)
        B = ActualLabels';
        error_bp = MSE(Epoch);
        break;
    end
    
end

B = ActualLabels';
error_bp = MSE(nbrOfEpochs_done);

inf_ind = isinf(B);
[inf_r, inf_c] = find(inf_ind==1);
B(inf_r,inf_c)= 0;
inf_ind = isnan(B);
[inf_r, inf_c] = find(inf_ind==1);
B(inf_r,inf_c)= 0;
end

function fx = ActivationFunction(x, unipolarBipolarSelector)
if (unipolarBipolarSelector == 0)
    fx = 1./(1 + exp(-x));
else
    fx = -1 + 2./(1 + exp(-x));
end
end

function fx_drev = ActivationDrevFunction(fx, unipolarBipolarSelector)
if (unipolarBipolarSelector == 0)
    fx_drev = fx .* (1 - fx);
else
    fx_drev = 0.5 .* (1 + fx) .* (1 - fx);
end
end

function outputs = EvaluateNetworkFunction(Sample, NodesActivations, Weights, unipolarBipolarSelector)

numLayers = length(NodesActivations);

NodesActivations{1} = Sample;
for Layer = 2:numLayers
    NodesActivations{Layer} = NodesActivations{Layer-1}*Weights{Layer-1};
    NodesActivations{Layer} = ActivationFunction(NodesActivations{Layer}, unipolarBipolarSelector);
    if (Layer ~= numLayers)
        NodesActivations{Layer}(1) = 1;
    end
end

outputs = NodesActivations{end};

end
