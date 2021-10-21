function run_proposed_forecast()
    clear
    
        mydataset = 'CCDS'
    % mydataset = 'meteo'
    
    
    myfractions = 0.8;
    myseeds = 101 %[101, 815, 906, 127, 633, 914, 98, 279, 547, 958] ;
    
    Karray = 5; %[3 5 7]
    rarray = 5; %[3 4 5]
    
    msemat = nan(length(myfractions), length(myseeds));
    expvarmat = nan(length(myfractions), length(myseeds));
    
    for K = Karray
        K
        
        for myrank = rarray
            
            rank_dims = [myrank myrank myrank]
            
            lambda = 3e2
            
            for ii = 1: length(myfractions)
                for jj = 1: length(myseeds)
                    
                    [~, myinfos] = proposed_forecast(mydataset, myfractions(ii), myseeds(jj), K, rank_dims, lambda);
                    
                    msearray = ([myinfos.test_rmse]).^2;
                    expvararray = 1 -  ([myinfos.test_nmse]);
                    
                    msemat(ii, jj) = msearray(end);
                    expvarmat(ii, jj) = expvararray(end);
                    
                end
            end
            
            msemat
            K
            lambda
            rank_dims
            myseeds
            
            myresults.myseeds = myseeds;
            myresults.myfractions = myfractions;
            myresults.msemat = msemat;
            myresults.expvarmat = expvarmat;
            myresults.rank_dims = rank_dims;
            myresults.K = K;
            myresults.lambda = lambda;
            
            savefile = strcat('results_forecast/proposed_',mydataset,'_K_',num2str(K),'_rank_',num2str(rank_dims(1)),'_results.mat');
            
            save(savefile,'myresults','-v7.3')
            
            fileattrib(savefile,'+w','a');
            
        end
    end
    
    
end