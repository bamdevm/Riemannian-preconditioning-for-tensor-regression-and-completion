function [P, M, TrainSet, TestSet] = my_generate_data_horll(dataset, K, train_ratio)

    data = importdata(dataset);

    

    P   = size(data.X ,1);  % # of positions
    T   = size(data.X ,2);  % # of time length
    M   = size(data.X ,3);  % dimension of variables

    train_set_length    = floor(T * train_ratio);
    test_set_length     = T - train_set_length;

    % Calculate real lengths because we need K-lag historical data for one prediction.
    train_real_length   = train_set_length - K; 
    test_real_length    = test_set_length;


    %% train
    % generate X_train
    X_train = zeros(P*K*M, train_real_length);

    for t = 1 : train_real_length
        for p = 1 : P
            p_offset = K*M*(p-1);
            for m = 1 : M
                m_offset = K*(m-1);
                for k = 1 : K
                    %X_train(p_offset + m_offset+k, t) = TrainSet(p,t+(k-1),m);
                    X_train(p_offset + m_offset+k, t) = data.X(p,t+(k-1),m);
                end
            end
        end
    end
    X_train = X_train';

    % generate Y_train
    Y_train = zeros(train_real_length, M, P);

    for t = 1 : train_real_length
        for p = 1 : P
            for m = 1 : M
                %Y_train(t, m, p) = TrainSet(p, t+K, m);
                Y_train(t, m, p) = data.X(p, t+K, m);
            end
        end
    end


    %% test
    % generate X_test
    X_test = zeros(P*K*M, test_real_length);

    t_offset = train_set_length - K;
    for t = 1 : test_real_length
        for p = 1 : P
            p_offset = K*M*(p-1);
            for m = 1 : M
                m_offset = K*(m-1);
                for k = 1 : K
                    %fprintf('%d, %d -- %d, %d, %d\n', p_offset + m_offset+k, t, p,t+(k-1),m);
                    %X_test(p_offset + m_offset+k, t) = TestSet(p,t+(k-1),m);
                    X_test(p_offset + m_offset+k, t) = data.X(p,t_offset+t+(k-1),m);
                end
            end
        end
    end
    X_test = X_test';

    % generate Y_test
    Y_test = zeros(test_real_length, M, P);

    for t = 1 : test_real_length
        for p = 1 : P
            for m = 1 : M
                %fprintf('%d, %d, %d -- %d, %d, %d\n', t, m, p, p, t+K, m);
                %Y_test(t, m, p) = TestSet(p, t+K, m);
                Y_test(t, m, p) = data.X(p, t_offset+t+K, m);
            end
        end
    end


    %X_train = tensor(X_train);
    %Y_train = tensor(Y_train);
    
    TrainSet.X  = X_train;
    TrainSet.Y  = Y_train; 
    train_num   = train_real_length;
    
    TestSet.X   = X_test;
    TestSet.Y   = Y_test;  
    test_num    = test_real_length;    

end

