function mydatastats = create_yelp_data(seed,fraction)
    
    rng('default')
    rng(seed);
    
    
    %% Load Data
    mydata = load('YELP_1000x992x93');
    
    
    Yorg = mydata.data.tensor;
    Borg = mydata.data.B_city; % [i j val] format
    Borgmat = sparse(Borg(:,1), Borg(:,2), Borg(:,3));

    
    %% Interchange mode 2 and mode 1 to make tensor 992x1000x93
    Y = [Yorg(:,2), Yorg(:,1), Yorg(:,3), Yorg(:,4)];
    

   
    %%
    n1 = max(Y(:,1));
    n2 = max(Y(:,2));
    n3 = max(Y(:,3));
   
    A1 = Borgmat;
    A2 = speye(n2);
    A3 = speye(n3);
    
    A1test = A1;
    A2test = A2;
    A3test = A3;
    
    
    % Train-Test split
    frac = fraction; % Training fraction
    n_entries = size(Y,1);
    ntrain = round(frac * n_entries);
    rand_order = randperm(n_entries);
    Y_train = Y(rand_order(1:ntrain), :);
    Y_test = Y(rand_order(ntrain+1:end),:);
    
    subs_train = Y_train(:,1:3);
    entries_train = Y_train(:,4);
    
    subs_test = Y_test(:,1:3);
    entries_test = Y_test(:,4) ;
    
    
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
    
    mydatastats.tensor_dims = [n1 n2 n3];
    
    
    
end