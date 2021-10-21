function run_proposed_multitask()
    clear
    
    mydataset = 'restaurant'
    % mydataset = 'school'
    
    
    myfractions = 0.5; %[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
    myseeds =  101; % [101, 815, 906, 127, 633, 914, 98, 279, 547, 958] ;
    
    msemat = nan(length(myfractions), length(myseeds));
    expvarmat = nan(length(myfractions), length(myseeds));
    
    
    rank_dims = [3 3 3]
    lambda = 1e4

    for ii = 1: length(myfractions)
        for jj = 1: length(myseeds)
            
            [~, myinfos] = proposed_multitask(mydataset, myfractions(ii), myseeds(jj), rank_dims, lambda);
            
            msearray = ([myinfos.test_rmse]).^2;
            expvararray = 1 -  ([myinfos.test_nmse]);
            
            msemat(ii, jj) = msearray(end);
            expvarmat(ii, jj) = expvararray(end);

            
        end
    end


    
    myresults.myseeds = myseeds;
    myresults.myfractions = myfractions;
    myresults.msemat = msemat;
    myresults.expvarmat = expvarmat;
    myresults.rank_dims = rank_dims;
    myresults.lambda = lambda;

    
    savefile = strcat('results_multitask/proposed_',mydataset,'_rank_',num2str(rank_dims(1)),'_results.mat');
    
    save(savefile,'myresults','-v7.3')
    
    fileattrib(savefile,'+w','a');
end