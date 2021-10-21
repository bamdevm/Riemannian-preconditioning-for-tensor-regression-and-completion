function[Xsol_TR, infos_TR] = proposed_multitask(dataset, fraction, myseed, rank_dims, lambda)
    
    %% load data
    if strcmp(dataset, 'restaurant')
        mydatastats = create_restaurant_data(myseed, fraction);
    elseif strcmp(dataset, 'school')
        mydatastats = create_school_data(myseed, fraction);
    else
        mydatastats = [];
    end
    
    
    
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
    
    
    %% Rank imposed
    % rank_dims = [2 2 2]; %
    rank_dims
    
    % Tolerance
    fraction
    % lambda = 1e-2
    lambda
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
    options.computenmse = true;
    options.lambda = lambda; % Regularization.
    
    
    %% TR Alogorithm
    options.solver =  @trustregions;% Trust regions
    options.maxiter = maxiter;
    
    
    [Xsol_TR, infos_TR] = RMLMTL(problem, [], options);
    
    msearray = ([infos_TR.test_rmse]).^2;
    finalmse = msearray(end);
    fprintf('Final MSE  %e\n', finalmse)
    rank_dims
    lambda
    fraction
    myseed
    
    
end