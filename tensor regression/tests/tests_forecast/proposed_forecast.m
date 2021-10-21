function [Xsol_TR, infos_TR] = proposed_forecast(dataset, fraction, myseed, K, rank_dims, lambda)
    
    %% load data
    if strcmp(dataset, 'meteo')
        filename='meteo_tensor_StationTimeVariables.mat'; % loads X and Y.
    elseif strcmp(dataset, 'CCDS')
        filename='CCDS_tensor_2.mat'; % loads X and Y.
    else
        filename = [];
    end
    
    
    %% Load data
    mydatastats = create_forecasting_data(filename, myseed, fraction, K);
    
    
    data_train.subs = mydatastats.train.subs;
    data_train.entries = mydatastats.train.entries;
    
    data_test.subs = mydatastats.test.subs;
    data_test.entries = mydatastats.test.entries;
    
    
    
    %% Side information
    A.A1 = mydatastats.sideinfo.A1;
    A.A2 = mydatastats.sideinfo.A2;
    A.A3 = mydatastats.sideinfo.A3;
    
    Atest.A1 = mydatastats.sideinfo.A1test;
    Atest.A2 = mydatastats.sideinfo.A2test;
    Atest.A3 = mydatastats.sideinfo.A3test;
    
    keyboard;   
    
    
    % Tolerance
    % lambda = 1e5;
    maxiter = 500;
    
    
    %% Call to our algorithm
    
    %  Required problem description
    problem.data_train = data_train;
    problem.data_test = data_test;
    problem.tensor_size = mydatastats.tensor_dims;
    problem.tensor_rank = rank_dims;
    problem.A = A;
    problem.Atest = Atest;
    problem.weights = ones(length(data_train.entries),1);
    
    % Some options, but not mandatory
    reltolgradnorm = 1e-3;
    options.reltolgradnorm = reltolgradnorm;
    options.tolmse = tolmse;
    options.computenmse = true;
    options.lambda = lambda; % Regularization.
    
    
    %% TR Alogorithm
    options.solver =  @trustregions;% Trust regions
    options.maxiter = maxiter;
    
    [Xsol_TR, infos_TR] = RMLMTL(problem, [], options);
    
    
end












