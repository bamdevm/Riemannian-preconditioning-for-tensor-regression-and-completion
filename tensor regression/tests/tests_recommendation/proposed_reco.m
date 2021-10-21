clear;


dataset = 'restaurant';

myseed = 101;

fraction = 0.8;



if strcmp(dataset, 'restaurant')
    mydatastats = create_restaurant_data(myseed, fraction);
elseif strcmp(dataset, 'school')
    mydatastats = create_school_data(myseed, fraction);
elseif strcmp(dataset, 'yelp')
    mydatastats = create_yelp_data(myseed, fraction);
else
    mydatastats = [];
end


%%

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
rank_dims = [1 1 1];

% Tolerance
lambda = 1e-2
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
reltolgradnorm = 1e-5;

options.reltolgradnorm = reltolgradnorm;
options.computenmse = true;
options.lambda = lambda; % Regularization.


%% TR Alogorithm
options.solver =  @trustregions;% Trust regions
options.maxiter = maxiter;

rng('default');
rng(myseed)


[Xsol_TR, infos_TR] = RMLMTL(problem, [], options);





%% Plot
figure;
plot( [infos_TR.test_rmse], 'color', 'blue');
hold on;
hold off;
xlabel('Iterations')
ylabel('Test RMSE')
legend('Proposed')


%%
numinner = [infos_TR.numinner];
numinnergeomTR = [infos_geomTR.numinner];
numinner(1) = [];
numinnergeomTR(1) = [];

figure;
plot( cumsum(numinner), 'color', 'blue');
hold on;
hold off;
xlabel('Iterations')
ylabel('Number inne iterations')
legend('Proposed')
