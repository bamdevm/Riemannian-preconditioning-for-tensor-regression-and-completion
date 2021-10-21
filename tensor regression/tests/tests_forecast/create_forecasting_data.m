function mydatastats = create_forecasting_data(filename, seed, train_ratio, K)
    
    rng('default')
    rng(seed);
    
    %% Load data
    
    [P, M, TrainSet, TestSet] = my_generate_data_horll(filename, K, train_ratio);
    
    
    
    n1 = size(TrainSet.X,1);
    n2 = size(TrainSet.Y,2);
    n3 = size(TrainSet.Y,3);
    
    tensor_dims = [n1 n2 n3];
    
    %% Train
    
    nr = prod(size(TrainSet.Y));
    nrtrain = nr;
    
    linindices = find(TrainSet.Y);
    [I1, I2, I3] = ind2sub(tensor_dims, linindices);
    
    subs_train = [I1, I2, I3];
    entries_train = TrainSet.Y(:);
    
    
    % % fraction of training data
    % myfrac = 1;
    % indxP = randperm(nrtrain,round(myfrac*nrtrain));
    % subs_train = subs_train(indxP,:);
    % entries_train = entries_train(indxP);
    
    
    
    %% Test
    nrtest = prod(size(TestSet.Y));
    
    linindicestest = find(TestSet.Y);
    
    n1test = size(TestSet.X,1);
    n2test = size(TestSet.Y,2);
    n3test = size(TestSet.Y,3);
    tensor_dims_test = [n1test n2test n3test];
    [I1test, I2test, I3test] = ind2sub(tensor_dims_test, linindicestest);
    subs_test = [I1test, I2test, I3test];
    entries_test = TestSet.Y(:);
    
    
    
    A1 = full(TrainSet.X);
    A2 = speye(n2);
    A3 = speye(n3);
    
    A1test = full(TestSet.X);
    A2test = speye(n2test);
    A3test = speye(n3test);
    
    mydatastats.sideinfo.A1 = A1;
    mydatastats.sideinfo.A2 = A2;
    mydatastats.sideinfo.A3 = A3;
    
    mydatastats.sideinfo.A1test = A1test;
    mydatastats.sideinfo.A2test = A2test;
    mydatastats.sideinfo.A3test = A3test;
    
    mydatastats.train.subs = subs_train ;
    mydatastats.train.entries = entries_train ;
    
    mydatastats.test.subs = subs_test ;
    mydatastats.test.entries = entries_test ;
    
    mydatastats.tensor_dims = tensor_dims;
    
    mydatastats.TrainSet = TrainSet;
    mydatastats.TestSet = TestSet;
    
end

