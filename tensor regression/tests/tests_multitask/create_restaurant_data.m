function mydatastats = create_restaurant_data(seed,fraction)
    
    rng('default')
	rng(seed);

	%% Load data
	mydata = load('RestaurantDataset.mat'); % loads X and Y.


	X = mydata.X;
	Y = mydata.Y;
	aspectIndices = mydata.aspectIndices;
	subjectIndices = mydata.subjectIndices;


	n1 = 3483;
	n2 = 3;
	n3 = 138;

	d1 = 45;

	tensor_dims = [n1 n2 n3];

	%%
	Y = [(1:n1)', aspectIndices', subjectIndices', Y'];

	%% permute
	idx = randperm(size(Y, 1));
	Ypermuted = Y(idx, :);

	%% Train-test split
	fraction = min(0.8, fraction);

	nr = size(Ypermuted, 1);

	nrtrain = round(fraction*nr);
	nrtest = min(nr - nrtrain, round(0.2*nr));

	% first up to 80% data as training data
	subs_train = Ypermuted(1:nrtrain, 1:3);
	entries_train = Ypermuted(1:nrtrain, 4);

	% last 20% data as test data
	subs_test = Ypermuted(nr - nrtest + 1 : nr, 1:3);
	entries_test = Ypermuted(nr - nrtest + 1 : nr, 4);

	A1 = X';
	A2 = speye(n2);
	A3 = speye(n3);

	A1test = A1;
	A2test = A2;
	A3test = A3;


	%% cell

	nips14cell_Xtrain_cell = cell(n2,n3);
	nips14cell_Ytrain_cell = cell(n2,n3);
	nips14cell_Xtest_cell = cell(n2,n3);
	nips14cell_Ytest_cell = cell(n2,n3);

	icml13cell_Xtrain_cell = cell(n2*n3,1);
	icml13cell_Ytrain_cell = cell(n2*n3,1);
	icml13cell_Xtest_cell = cell(n2*n3,1);
	icml13cell_Ytest_cell = cell(n2*n3,1);
	icml13cell_iter = 0;

	subs_train_n1 = subs_train(:,1);
	subs_train_n2 = subs_train(:,2);
	subs_train_n3 = subs_train(:,3);

	subs_test_n1 = subs_test(:,1);
	subs_test_n2 = subs_test(:,2);
	subs_test_n3 = subs_test(:,3);
	train_task_matrix = ones(n2,n3);
	test_task_matrix = ones(n2,n3);

	for j=1:n3
		for i=1:n2
			icml13cell_iter = icml13cell_iter + 1;
			pseudo_idx_train = (subs_train_n2==i) & (subs_train_n3==j);
			if sum(pseudo_idx_train)==0
				fprintf('No train data for task: (%d,%d)\n',i,j);
				train_task_matrix(i,j) = 0;
			end
			idx_train = subs_train_n1(pseudo_idx_train);
			nips14cell_Xtrain_cell{i,j} = A1(idx_train,:);
			nips14cell_Ytrain_cell{i,j} = (entries_train(pseudo_idx_train))';

			icml13cell_Xtrain_cell{icml13cell_iter} = A1(idx_train,:)';
			icml13cell_Ytrain_cell{icml13cell_iter} = entries_train(pseudo_idx_train);

			pseudo_idx_test = (subs_test_n2==i) & (subs_test_n3==j);
			if sum(pseudo_idx_test)==0
				fprintf('No test data for task: (%d,%d)\n',i,j);
				test_task_matrix(i,j) = 0;
			end
			idx_test = subs_test_n1(pseudo_idx_test);
			nips14cell_Xtest_cell{i,j} = A1(idx_test,:);
			nips14cell_Ytest_cell{i,j} = (entries_test(pseudo_idx_test))';

			icml13cell_Xtest_cell{icml13cell_iter} = A1(idx_test,:)';
			icml13cell_Ytest_cell{icml13cell_iter} = entries_test(pseudo_idx_test);
		end
	end

	if any(sum(train_task_matrix,1)==0) && any(sum(train_task_matrix,2)==0)
		fprintf('Error! Some tasks are not having any training data! Learned tensor will have some zero rows or columns\n');
		keyboard;
	end

	if any(sum(test_task_matrix,1)==0) && any(sum(test_task_matrix,2)==0)
		fprintf('Warning! Some tasks are not having any testing data!\n')
	end

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

	mydatastats.nips14cell.Xtrain_cell = nips14cell_Xtrain_cell;
	mydatastats.nips14cell.Ytrain_cell = nips14cell_Ytrain_cell;
	mydatastats.nips14cell.Xtest_cell = nips14cell_Xtest_cell;
	mydatastats.nips14cell.Ytest_cell = nips14cell_Ytest_cell;
	mydatastats.nips14cell.tindex = [d1 n2 n3];

	mydatastats.icml13cell.Xtrain_cell = icml13cell_Xtrain_cell;
	mydatastats.icml13cell.Ytrain_cell = icml13cell_Ytrain_cell;
	mydatastats.icml13cell.Xtest_cell = icml13cell_Xtest_cell;
	mydatastats.icml13cell.Ytest_cell = icml13cell_Ytest_cell;
	mydatastats.icml13cell.indicators = [d1 n2 n3];
end