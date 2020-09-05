function [net, stats] = cnn_train_v2_cifar_BPGrad(net, imdb, getBatch, varargin)
%CNN_TRAIN  An example implementation of SGD for training CNNs
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option).

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
addpath(fullfile(vl_rootnn, 'examples'));

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;  
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.epochSize = inf;
opts.prefetch = false ;
opts.numEpochs = 100 ;   
opts.learningRate = 0.001 ;  
opts.weightDecay = 5e-4 ;  

solver_idx = 6; % To run BPGrad
if solver_idx == 1
    opts.solver = @SGD;   
    opts.filesavename = 'SGD';
elseif solver_idx == 2
    opts.solver = @adam;  
    opts.filesavename = 'adam';
elseif solver_idx == 3
    opts.solver = @adagrad;  
    opts.filesavename = 'adagrad';
elseif solver_idx == 4
    opts.solver = @adadelta;  
    opts.filesavename = 'adadelta';
elseif solver_idx == 5
    opts.solver = @rmsprop;  
    opts.filesavename = 'rmsprop';
elseif solver_idx == 6
    opts.solver = [] ;
    opts.filesavename = 'BPGrad';
end
     
[opts, varargin] = vl_argparse(opts, varargin) ;
if ~isempty(opts.solver)
  assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
    'Invalid solver; expected a function handle with two outputs.') ;
  % Call without input arguments, to get default options
  opts.solverOpts = opts.solver() ;
end

opts.momentum = 0.9;
opts.saveSolverState = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;

opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts.postEpochFn = [] ;  
opts = vl_argparse(opts, varargin) ;

% Update the opts instead of using cnn_cifar_init.m for different solvers
opts.batchSize = 100;
opts.numEpochs = 100;
opts.weightDecay = 5e-4;
% opts.learningRate = [0.01*ones(1,30) 0.005*ones(1,30) 0.001*ones(1,40)];

opts.Ls = [50*ones(1,100)];  

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isscalar(opts.train) && isnumeric(opts.train) && isnan(opts.train)
  opts.train = [] ;
end
if isscalar(opts.val) && isnumeric(opts.val) && isnan(opts.val)
  opts.val = [] ;
end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------
net = vl_simplenn_tidy(net); % fill in some eventually missing values
net.layers{end-1}.precious = 1; % do not remove predictions, used for error

load('../weights_folder/Weights_init_cifar10.mat');
for i=1:numel(net.layers)
    net.layers{i}.weights=Weights_init{i};
end
 
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  for i=1:numel(net.layers)
    J = numel(net.layers{i}.weights) ;
    if ~isfield(net.layers{i}, 'learningRate')
      net.layers{i}.learningRate = ones(1, J) ;
    end
    if ~isfield(net.layers{i}, 'weightDecay')
      net.layers{i}.weightDecay = ones(1, J) ;
    end
  end
end

% setup error calculation function
hasError = true ;
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
      hasError = false ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
    otherwise
      error('Unknown error function ''%s''.', opts.errorFunction) ;
  end
end

state.getBatch = getBatch ;
stats = [] ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% ------------------------------------------------------------------------- 
folder = sprintf('results_%s_080120_mu_%d', opts.filesavename, num); 
opts.My_expDir = fullfile(opts.expDir, folder); % update the folder path
fprintf('opts.My_expDir=%s\n', opts.My_expDir);

if ~exist(opts.My_expDir, 'dir'), mkdir(opts.My_expDir) ; end
modelPath = @(ep) fullfile(opts.My_expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.My_expDir, sprintf('net-train-%s.pdf', opts.filesavename));
 
start = opts.continue * findLastCheckpoint(opts.My_expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, state, stats] = loadState(modelPath(start)) ;
else
  state = [] ;
end

params = opts ;
subset_train = params.('train') ;
params.train_Loss_value_in_batch = zeros(3,ceil(numel(subset_train)/params.batchSize) * params.numEpochs);

subset_val = params.('val') ;
params.val_Loss_value_in_batch = zeros(3,ceil(numel(subset_val)/params.batchSize) * params.numEpochs);
params.L_flag = 0; 
params.train_batch_order = [];
params.val_batch_order = [];
params.fw0 = 0;  
params.rho = 0.1;  
params.yita_t = zeros(1,ceil(numel(subset_train)/params.batchSize) * params.numEpochs); % yita_t equ (12)
params.I_t = []; 

for epoch=start+1:opts.numEpochs
  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, epoch == start+1) ;

  params.epoch = epoch ;
  params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  params.train = params.train(1:min(opts.epochSize, numel(opts.train)));
  params.val = opts.val(randperm(numel(opts.val))) ;
  params.imdb = imdb ;
  params.getBatch = getBatch ;

  params.L = params.Ls(params.epoch);  
  fprintf('params.rho=%.2f, params.L=%d\n', params.rho, params.L); 

  params.train_batch_order(epoch,:) = params.train;
  params.val_batch_order(epoch,:) = params.val;  

  if numel(params.gpus) <= 1
    [net, state, params] = processEpoch(net, state, params, 'train') ;
    [net, state, params] = processEpoch(net, state, params, 'val') ;
    if ~evaluateMode
      saveState(modelPath(epoch), net, state) ;
    end
    lastStats = state.stats ;
  else
    spmd
      [net, state, params] = processEpoch(net, state, params, 'train') ;
      [net, state, params] = processEpoch(net, state, params, 'val') ;
      if labindex == 1 && ~evaluateMode
        saveState(modelPath(epoch), net, state) ;
      end
      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end

  stats.train(epoch) = lastStats.train ;
  stats.val(epoch) = lastStats.val ;
  clear lastStats ;
  if ~evaluateMode
    saveStats(modelPath(epoch), stats) ;
  end

  if params.plotStatistics
    switchFigure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end

  if ~isempty(opts.postEpochFn)
    if nargout(opts.postEpochFn) == 0
      opts.postEpochFn(net, params, state) ;
    else
      lr = opts.postEpochFn(net, params, state) ;
      if ~isempty(lr), opts.learningRate = lr; end
      if opts.learningRate == 0, break; end
    end
  end

  train_Loss_value_in_batch = params.train_Loss_value_in_batch;
  save(fullfile(opts.My_expDir, 'net-train-params-train_Loss_value_in_batch.mat'), 'train_Loss_value_in_batch');
  val_Loss_value_in_batch = params.val_Loss_value_in_batch;
  save(fullfile(opts.My_expDir, 'net-train-params-val_Loss_value_in_batch.mat'), 'val_Loss_value_in_batch');
  train_batch_order = params.train_batch_order;
  save(fullfile(opts.My_expDir, 'net-train-params-train_batch_order.mat'), 'train_batch_order');
  val_batch_order = params.val_batch_order;
  save(fullfile(opts.My_expDir, 'net-train-params-val_batch_order.mat'), 'val_batch_order');
end 
% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function err = error_multiclass(params, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binary(params, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_none(params, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function [net, state, params] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
  for i = 1:numel(net.layers)
    state.solverState{i} = cell(1, numel(net.layers{i}.weights)) ;
    state.solverState{i}(:) = {0} ;
  end
end


% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  net = vl_simplenn_move(net, 'gpu') ;
  for i = 1:numel(state.solverState)
    for j = 1:numel(state.solverState{i})
      s = state.solverState{i}{j} ;
      if isnumeric(s)
        state.solverState{i}{j} = gpuArray(s) ;
      elseif isstruct(s)
        state.solverState{i}{j} = structfun(@gpuArray, s, 'UniformOutput', false) ;
      end
    end
  end
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  vl_simplenn_start_parserv(net, parserv) ;
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

subset = params.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
res = [] ;
error = [] ;
error_obj = 0;
error_toperr = []; 

start = tic ;

batch_num_per_Epoch = ceil(numel(subset)/params.batchSize);

for t=1:params.batchSize:numel(subset)
  batch_idx = fix((t-1)/params.batchSize)+1;
  iteration_t = batch_idx + (params.epoch - 1) * batch_num_per_Epoch;
  fprintf('%s: epoch %02d: %3d/%3d, iteration_t=%d:', mode, params.epoch, ...
          batch_idx, batch_num_per_Epoch, iteration_t) ;
  params.iteration_t = iteration_t;      
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;
  
  

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    [im, labels] = params.getBatch(params.imdb, batch) ;

    if params.prefetch
      if s == params.numSubBatches
        batchStart = t + (labindex-1) + params.batchSize ;
        batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
      params.getBatch(params.imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    if strcmp(mode, 'train')
      dzdy = 1 ;
      evalMode = 'normal' ;
    else
      dzdy = [] ;
      evalMode = 'test' ;
    end
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', evalMode, ...
                      'conserveMemory', params.conserveMemory, ...
                      'backPropDepth', params.backPropDepth, ...
                      'sync', params.sync, ...
                      'cudnn', params.cudnn, ...
                      'parameterServer', parserv, ...
                      'holdOn', s < params.numSubBatches) ;

    % accumulate errors
    error_obj = double(gather(res(end).x));
    error_toperr = reshape(params.errorFunction(params, labels, res),[],1);
    error = sum([error, [sum(error_obj);error_toperr ; ]],2) ;
  
  end

  if strcmp(mode, 'train')
      params.train_Loss_value_in_batch(1, iteration_t) = error(1) / num; 
      params.train_Loss_value_in_batch(2, iteration_t) = error(2) / num; 
      params.train_Loss_value_in_batch(3, iteration_t) = error(3) / num; 
  end
  if strcmp(mode, 'val') 
      params.val_Loss_value_in_batch(1, iteration_t) = error(1) / num;
      params.val_Loss_value_in_batch(2, iteration_t) = error(2) / num; 
      params.val_Loss_value_in_batch(3, iteration_t) = error(3) / num; 
  end
  
  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(parserv), parserv.sync() ; end
    [net, res, state, params] = accumulateGradients(net, res, state, params, batchSize, parserv) ;
  end
  
  % get statistics
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats = extractStats(net, params, error / num) ;
  stats.num = num ;
  stats.time = time ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == 3*params.batchSize + 1 
    adjustTime = 4*batchTime - time ;
    stats.time = time + adjustTime ;
  end
 
  fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s: %.3f', f, stats.(f)) ;
  end
  fprintf('\n') ; 

  % collect diagnostic statistics
  if strcmp(mode, 'train') && params.plotDiagnostics
    switchFigure(2) ; clf ;
    diagn = [res.stats] ;
    diagnvar = horzcat(diagn.variation) ;
    diagnpow = horzcat(diagn.power) ;
    subplot(2,2,1) ; barh(diagnvar) ;
    set(gca,'TickLabelInterpreter', 'none', ...
      'YTick', 1:numel(diagnvar), ...
      'YTickLabel',horzcat(diagn.label), ...
      'YDir', 'reverse', ...
      'XScale', 'log', ...
      'XLim', [1e-5 1], ...
      'XTick', 10.^(-5:1)) ;
    grid on ; title('Variation');
    subplot(2,2,2) ; barh(sqrt(diagnpow)) ;
    set(gca,'TickLabelInterpreter', 'none', ...
      'YTick', 1:numel(diagnpow), ...
      'YTickLabel',{diagn.powerLabel}, ...
      'YDir', 'reverse', ...
      'XScale', 'log', ...
      'XLim', [1e-5 1e5], ...
      'XTick', 10.^(-5:5)) ;
    grid on ; title('Power');
    subplot(2,2,3); plot(squeeze(res(end-1).x)) ;
    drawnow ;
  end
end

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
  if numGpus <= 1
    state.prof.(mode) = profile('info') ;
    profile off ;
  else
    state.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveSolverState
  state.solverState = [] ;
else
  for i = 1:numel(state.solverState)
    for j = 1:numel(state.solverState{i})
      s = state.solverState{i}{j} ;
      if isnumeric(s)
        state.solverState{i}{j} = gather(s) ;
      elseif isstruct(s)
        state.solverState{i}{j} = structfun(@gather, s, 'UniformOutput', false) ;
      end
    end
  end
end

net = vl_simplenn_move(net, 'cpu') ;

% -------------------------------------------------------------------------
function [net, res, state, params] = accumulateGradients(net, res, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;
 
if isempty(params.solver)
    dzdw_vector = [];
    for l=numel(net.layers):-1:1
      for j=numel(res(l).dzdw):-1:1
        if ~isempty(parserv)
          tag = sprintf('l%d_%d',l,j) ;
          parDer = parserv.pull(tag) ;
        else
          parDer = res(l).dzdw{j} ;
        end
        
        % Normalize gradient, and then vectorize
        parDer =  parDer /params.batchSize;      
        parDer_reshape = reshape(parDer, [1, numel(parDer)]);
        dzdw_vector = [dzdw_vector parDer_reshape];        
      end
    end     
    dzdw_norm = norm(dzdw_vector, 2);   
    
    %%% calculate the ft*, yita_t
    params.fw0 = params.rho * min(params.train_Loss_value_in_batch(1, 1:params.iteration_t));
    params.yita_t(params.iteration_t) = (params.train_Loss_value_in_batch(1, params.iteration_t) - params.fw0) / params.L;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for l=numel(net.layers):-1:1
  for j=numel(res(l).dzdw):-1:1

    if ~isempty(parserv)
      tag = sprintf('l%d_%d',l,j) ;
      parDer = parserv.pull(tag) ;
    else
      parDer = res(l).dzdw{j}  ;
    end
    
    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = vl_taccum(...
        1 - thisLR, ...
        net.layers{l}.weights{j}, ...
        thisLR / batchSize, ...
        parDer) ;
    else

      % Standard gradient training.      
      thisDecay = params.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = params.learningRate * net.layers{l}.learningRate(j) ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.layers{l}.weights{j}) ;
        %%% modified for BPGrad 
        if isempty(params.solver) 
            thisLR = params.yita_t(params.iteration_t); 
            parDer = parDer / dzdw_norm;  
       
          % Default solver is the optimised SGD.
          % Update momentum.
            state.solverState{l}{j} = vl_taccum(...
              params.momentum, state.solverState{l}{j}, ...
              -1, parDer) ;

          % Nesterov update (aka one step ahead).
            if params.nesterovUpdate
              delta = params.momentum * state.solverState{l}{j} - parDer ;
            else
              delta = state.solverState{l}{j} ;
            end
          
          %%% Update parameters
            net.layers{l}.weights{j} = vl_taccum(...
              1, net.layers{l}.weights{j}, ...
              thisLR, delta) ;
        else
          % call solver function to update weights
          thisLR = params.learningRate * net.layers{l}.learningRate(j) ;
          [net.layers{l}.weights{j}, state.solverState{l}{j}] = ...
            params.solver(net.layers{l}.weights{j}, state.solverState{l}{j}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
    end

    % if requested, collect some useful stats for debugging
    if params.plotDiagnostics
      variation = [] ;
      label = '' ;
      switch net.layers{l}.type
        case {'conv','convt'}
          if isnumeric(state.solverState{l}{j})
            variation = thisLR * mean(abs(state.solverState{l}{j}(:))) ;
          end
          power = mean(res(l+1).x(:).^2) ;
          if j == 1 % fiters
            base = mean(net.layers{l}.weights{j}(:).^2) ;
            label = 'filters' ;
          else % biases
            base = sqrt(power) ;%mean(abs(res(l+1).x(:))) ;
            label = 'biases' ;
          end
          variation = variation / base ;
          label = sprintf('%s_%s', net.layers{l}.name, label) ;
      end
      res(l).stats.variation(j) = variation ;
      res(l).stats.power = power ;
      res(l).stats.powerLabel = net.layers{l}.name ;
      res(l).stats.label{j} = label ;
    end
  end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net, params, errors)
% -------------------------------------------------------------------------
stats.objective = errors(1) ;
for i = 1:numel(params.errorLabels)
  stats.(params.errorLabels{i}) = errors(i+1) ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, state)
% -------------------------------------------------------------------------
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', '-append') ;
else
  save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = vl_simplenn_tidy(net) ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
%clear vl_tmove vl_imreadjpeg ;
disp('Clearing mex files') ;
clear mex ;
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(params, cold)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end
end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename) ;
  clearMex() ;
  if numGpus == 1
    disp(gpuDevice(params.gpus)) ;
  else
    spmd
      clearMex() ;
      disp(gpuDevice(params.gpus(labindex))) ;
    end
  end
end

function [L] = Estimate_L(params)
iter = params.L_calculate_flag;
Loss_value = params.train_Loss_value_in_batch(1, 1:iter);
Weights_value = params.Weights_value_in_batch(1:iter, :);
dzdw_value  = params.dzdw_value_in_batch(1:iter, :);
Loss_diff_matrix = ones(iter);
W_diff_matrix = ones(iter);
dzdw_diff_matrix = ones(iter);
%%% calculate Loss_diff_matrix
for i = 1:iter
    L1 = Loss_value(i);
    for j = 1:iter
        if i ~= j            
            L2 = Loss_value(j); 
            Loss_diff_matrix(i,j) = abs(L2 - L1);
        end
    end
end
%%% calculate W_diff_matrix
for i = 1:iter
    W1 = Weights_value(i,:);
    for j = 1:iter
        if i ~= j            
            W2 = Weights_value(j,:); 
            dzdw = 0;
            for k = 1:numel(W1)
                if ~isempty(W1{k})
                    for kk = 1:2
                        p1 = reshape(W1{k}{kk}, [numel(W1{k}{kk}), 1]);
                        p2 = reshape(W2{k}{kk}, [numel(W2{k}{kk}), 1]);
                        dzdw = dzdw + norm(p1-p2,2)^2;
                    end
                end
            end
            W_diff_matrix(i,j) = sqrt(dzdw);
        end
    end
end

%%% calculate dzdw_diff_matrix 
for i = 1:iter
    dzdw1 = dzdw_value(i,:);
    for j = 1:iter
        if i ~= j            
            dzdw2 = dzdw_value(j,:); 
            dzdw = 0;
            for k = 1:numel(dzdw1)
                if ~isempty(dzdw1{k})
                    for kk = 1:2
                        p1 = reshape(dzdw1{k}{kk}, [numel(dzdw1{k}{kk}), 1]);
                        p2 = reshape(dzdw2{k}{kk}, [numel(dzdw2{k}{kk}), 1]);
                        dzdw = dzdw + norm(p1-p2,2)^2;
                    end
                end
            end
            dzdw_diff_matrix(i,j) = sqrt(dzdw);
        end
    end
end

%%% calculate L distribution
L_matrix = Loss_diff_matrix ./ W_diff_matrix;
L_vector = reshape(L_matrix, [numel(L_matrix), 1]);
L_max = max(L_vector);
L_mean = mean(L_vector);
L_median = median(L_vector);
L = L_mean;
histogram(L_vector);
title(sprintf('SGD: L max=%.2f, mean=%.2f, midean=%.2f, batches=%d',L_max,L_mean,L_median, iter));