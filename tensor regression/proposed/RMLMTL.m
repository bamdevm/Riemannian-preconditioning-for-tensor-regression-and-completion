function [Xsol, infos, options] = RMLMTL(problem, X_initial,options)
% Perform Riemannian multilinear multitask learning.
%
% function Xsol = RMLMTL(problem)
% function [Xsol, infos, options] = RMLMTL(problem)
%
% Input:
% -------
%
% problem: The problem structure with the description of the problem.
%
%
% - problem.A: side information or features stored in fields 'A1' (mode 1), 
%				'A2' (mode 2), and 'A3' (mode 3).
%
% - problem.data_train: Data structure for known entries that are used to learn a low-rank model.
%                                   It contains the 2 fields that are shown
%                                   below. An empty "data_train" structure
%                                   will generate a random instance.
%
%       -- data_train.entries:      A column vector consisting of known
%                                   distances. An empty "data_train.entries"
%                                   field will generate a random instance.
%
%
%       -- data_train.subs:         The [i j k] subscripts of corresponding
%                                   entries. An empty "data_train.subs"
%                                   field will generate a random instance.%
%
%
%
% - problem.Atest: test side information or features stored in fields 'A1' (mode 1),
% 					'A2' (mode 2), and 'A3' (mode 3).
%
% - problem.data_test:  Data structure to the "unknown" (to the algorithm) entries.
%                                   It contains the 2 fields that are shown
%                                   below. An empty "data_test" structure
%                                   will not compute the test error.
%
%       -- data_test.entries:       A column vector consisting of "unknown" (to the algorithm)
%                                   entries. An empty "data_test.entries"
%                                   field will not compute the test error.
%       -- data_test.subs:          The [i j k] subscripts of the corresponding
%                                   entries. An empty "data_test.subs"
%                                   field will not compute the test error.
%
%
% - problem.tensor_size: The size of the tensor. An empty
%                                    "tensor_size", but complete "data_train" structure
%                                    will lead to an error, to avoid
%                                    potential data inconsistency.
%
%
% - problem.tensor_rank: Rank. By default, it is [3 4 5].
%
%
% - problem.weights: a column vector of positive numbers that
%                                is used to weigh different entries.
%                                Default is all ones.
%
%
% - problem.options:  Structure array containing algorithm
%                                parameters for stopping criteria.
%
%       -- options.tolgradnorm:   Tolerance on the norm of the gradient.
%                                By default, it is 1e-5.
%       -- options.tolmse:        Tolerance on the mlse.
%                                By default, it is 1e-5.
%       -- options.maxiter:       Maximum number of fixe-rank iterations.
%
%       -- options.solver:        Fixed-rank algorithm. Options are
%                                '@trustregions' for trust-regions,
%                                '@conjugategradient' for conjugate gradients,
%                                '@steepestdescent' for steepest descent.
%                                 By default, it is '@conjugategradient'.
%
%       -- options.linesearch:    Stepsize computation. Options are
%                                '@linesearchguess2' for guessed stepsize
%                                 with degree 2 polynomial guess,
%                                '@linesearchdefault' by Manopt
%                                 By default, it is '@linesearchguess2'.
%
%      -- options.show_plots:     Show basic plots.
%                                 By default, it is false.
%
%      -- options.verbosity:      Show output with 0, 1, 2
%                                 By default, 2.
%
%      -- options.lambda:         Regularization parameter for || X||_F^2
%                                 regularization.
%
%
%
% Output:
% --------
%
%   X_sol:                Solution.
%   infos:                Structure array with computed statistics.
%	options:			  Options used in optimization.
%

    % Define the problem data
    data_train = problem.data_train;
    data_train.nentries = length(data_train.entries); % Useful
    
    data_test =  problem.data_test; % Test data
    
    % Weights
    w = problem.weights; % Weighted tensor completion W.*(W - W^start).
    
    
    
    tensor_size = problem.tensor_size;
    tensor_rank = problem.tensor_rank;
    
    lambda = options.lambda; % Regularizer.
    
    n1 = tensor_size(1);
    n2 = tensor_size(2);
    n3 = tensor_size(3);
    
    r1 = tensor_rank(1);
    r2 = tensor_rank(2);
    r3 = tensor_rank(3);
   
    
    A = problem.A; % Side information
    A1 = A.A1;
    A2 = A.A2;
    A3 = A.A3;
    
    learned_tensor_size = [size(A1, 2), size(A2, 2), size(A3, 2)]; % We learn a tensor of this size.
    
    % Local defaults
    localdefaults.reltolgradnorm = 1e-5; % Relative gradnorm tolerance.
    localdefaults.maxiter = 250;
    localdefaults.maxinner = 30;
    localdefaults.solver = @conjugategradient; % Conjugate gradients
    localdefaults.linesearch = @linesearchguess2; % Stepsize guess with 2 polynomial
    localdefaults.show_plots = false;
    localdefaults.verbosity = 2;
    localdefaults.lambda = 0; % Regularization.
    localdefaults.computenmse = false;
    localdefaults.computeauc = false; 
    
    options = mergeOptions(localdefaults, options);
    
    if ~isempty(data_test)
        Atest = problem.Atest; % Side information for the test set.
        A1test = Atest.A1;
        A2test = Atest.A2;
        A3test = Atest.A3;
    end
    
    % The fixed-rank Tucker factory
    M = fixedrankfactory_tucker_preconditioned(learned_tensor_size, tensor_rank);
    problem.M = M;
    
    
    % Problem description: cost
    problem.cost = @cost;
    function [f, store] = cost(X, store)
        if ~isfield(store, 'residual_vec') % Re use if it already computed
            W.U1 = full(A1*X.U1);
            W.U2 = full(A2*X.U2);
            W.U3 = full(A3*X.U3);
            W.G = full(X.G);
            store.W = W;
            
            store.residual_vec = compute_residual(W, data_train);
            residual_vec = w.*store.residual_vec;
            store.cost = 0.5*(residual_vec'*residual_vec); % BM
        end
        W = store.W;
        f = store.cost;
        f = f + 0.5*lambda*norm(W.G(:), 2)^2; % Add regularization.
    end
    
    
    
    % Problem description: gradient
    % We need to give only the Euclidean gradient, Manopt converts it to
    % to the Riemannian gradient internally.
    problem.egrad =  @egrad;
    function [g, store] = egrad(X, store)
        if ~isfield(store, 'residual_vec') % Re use if it already computed
            [~, store] = cost(X, store);
        end
        W = store.W;
        
        [temp,temp1,temp2,temp3] = calcProjection_mex(data_train.subs', (w.^2).*store.residual_vec, W.U1', W.U2', W.U3' );
        
        
        Y23 = reshape(temp1, [n1 r2 r3]); % tensor(reshape(temp1, [n1 r2 r3]));
        Y13 = reshape(temp2, [r1 n2 r3]); % tensor(reshape(temp2, [r1 n2 r3]));
        Y12 = reshape(temp3, [r1 r2 n3]); % tensor(reshape(temp3, [r1 r2 n3]));
        
        T1 = reshape(Y23, n1, r2*r3) * reshape(W.G, r1, r2*r3)';
        T2 = reshape(permute(Y13, [2 1 3]), n2, r1*r3) * reshape(permute(W.G, [2 1 3]), r2, r1*r3)';
        T3 = reshape(permute(Y12, [3 1 2]), n3, r1*r2)* reshape(permute(W.G, [3 1 2]), r3, r1*r2)';
        
        g.U1 = full(A1'*T1);
        g.U2 = full(A2'*T2);
        g.U3 = full(A3'*T3);
        g.G = full(reshape(temp, [r1 r2 r3]));% tensor(reshape(temp, [r1 r2 r3]), [r1, r2, r3]);
        
        g.G = g.G + lambda*W.G;
    end
    
    % Problem description: Hesssian
    % We need to give only the Euclidean Hessian, Manopt converts it to
    % to the Riemannian gradient internally.
    problem.ehess =  @egraddot;
    function [gdot, store] = egraddot(X, Xdot, store)
        if ~isfield(store, 'residual_vec') % Re use if it already computed
            [~, store] = cost(X, store);
        end
        W = store.W;
        
        Wdot.U1 = full(A1*Xdot.U1);
        Wdot.U2 = full(A2*Xdot.U2);
        Wdot.U3 = full(A3*Xdot.U3);
        Wdot.G = full(Xdot.G);
        
        [temp,temp1,temp2,temp3] = calcProjection_mex(data_train.subs', (w.^2).*store.residual_vec, W.U1', W.U2', W.U3' );
        
        % Dots
        residual_vec_dot = compute_residual_dot(W, data_train, Wdot);
        
        [tempdot1,temp1dot1,temp2dot1,temp3dot1] = calcProjection_mex(data_train.subs', (w.^2).*residual_vec_dot, W.U1', W.U2', W.U3' ) ;
        [tempdot2,temp1dot2,temp2dot2,temp3dot2] = calcProjection_mex(data_train.subs', (w.^2).*store.residual_vec, Wdot.U1', W.U2', W.U3' ) ;
        [tempdot3,temp1dot3,temp2dot3,temp3dot3] = calcProjection_mex(data_train.subs', (w.^2).*store.residual_vec, W.U1', Wdot.U2', W.U3' ) ;
        [tempdot4,temp1dot4,temp2dot4,temp3dot4] =  calcProjection_mex(data_train.subs', (w.^2).*store.residual_vec, W.U1', W.U2', Wdot.U3' );
        
        
        tempdot = tempdot1 + tempdot2 + tempdot3 + tempdot4;
        temp1dot = temp1dot1 + temp1dot3 + temp1dot4;
        temp2dot = temp2dot1 + temp2dot2 +  temp2dot4;
        temp3dot = temp3dot1 + temp3dot2 + temp3dot3 ;
        
        
        % Ydots
        Y23 = reshape(temp1, [n1 r2 r3]); % tensor(reshape(temp1, [n1 r2 r3]));
        Y13 = reshape(temp2, [r1 n2 r3]); % tensor(reshape(temp2, [r1 n2 r3]));
        Y12 = reshape(temp3, [r1 r2 n3]); % tensor(reshape(temp3, [r1 r2 n3]));
        
        Y23dot = reshape(temp1dot, [n1 r2 r3]); % tensor(reshape(temp1, [n1 r2 r3]));
        Y13dot = reshape(temp2dot, [r1 n2 r3]); % tensor(reshape(temp2, [r1 n2 r3]));
        Y12dot = reshape(temp3dot, [r1 r2 n3]); % tensor(reshape(temp3, [r1 r2 n3]));
        
        
        % Tdots
        %         T1 = reshape(Y23, n1, r2*r3) * reshape(W.G, r1, r2*r3)';
        %         T2 = reshape(permute(Y13, [2 1 3]), n2, r1*r3) * reshape(permute(W.G, [2 1 3]), r2, r1*r3)';
        %         T3 = reshape(permute(Y12, [3 1 2]), n3, r1*r2)* reshape(permute(W.G, [3 1 2]), r3, r1*r2)';
        
        T1dot = reshape(Y23dot, n1, r2*r3) * reshape(W.G, r1, r2*r3)'...
            + reshape(Y23, n1, r2*r3) * reshape(Wdot.G, r1, r2*r3)';
        T2dot = reshape(permute(Y13dot, [2 1 3]), n2, r1*r3) * reshape(permute(W.G, [2 1 3]), r2, r1*r3)' ...
            + reshape(permute(Y13, [2 1 3]), n2, r1*r3) * reshape(permute(Wdot.G, [2 1 3]), r2, r1*r3)';
        T3dot = reshape(permute(Y12dot, [3 1 2]), n3, r1*r2)* reshape(permute(W.G, [3 1 2]), r3, r1*r2)'...
            + reshape(permute(Y12, [3 1 2]), n3, r1*r2)* reshape(permute(Wdot.G, [3 1 2]), r3, r1*r2)';
        
        gdot.U1 = full(A1'*T1dot);
        gdot.U2 = full(A2'*T2dot);
        gdot.U3 = full(A3'*T3dot);
        gdot.G = full(reshape(tempdot, [r1 r2 r3]));% tensor(reshape(tempdot, [r1 r2 r3]), [r1, r2, r3]);
        
        gdot.G = gdot.G + lambda*Wdot.G;
    end
    
    
    
    
    % Problem description: stepsize computation with degree 2 polynomial
    % approximation
    function [t, store] = linesearchguess2(X, Xdot, store)
        if ~isfield(store, 'residual_vec') % Re use if it already computed
            [~, store] = cost(X, store);
        end
        W = store.W;
        Wdot.U1 = full(A1*Xdot.U1);
        Wdot.U2 = full(A2*Xdot.U2);
        Wdot.U3 = full(A3*Xdot.U3);
        Wdot.G = full(Xdot.G);
        
        t  = stepsize_guess_degree2(W, Wdot, store); % Better for large-scale instances.
    end
    
    function tmin  = stepsize_guess_degree2(W, Wdot, store)
        if ~isfield(store, 'residual_vec') % Re use if it already computed
            store.residual_vec = compute_residual(W, data_train);
        end
        residual_vec = store.residual_vec;
        
        % new compute
        tmin = compute_stepsize_initial(W, Wdot, residual_vec, data_train.subs);
    end
    
    
    
    %     % Check numerically whether gradient and Hessian are correct
    %     checkgradient(problem); %# ok
    %     drawnow;
    %     pause;
    %     checkhessian(problem); %# ok
    %     drawnow;
    %     pause;
    
    
    
    % Options
    % Ask Manopt to compute the error at every iteration when a
    % test dataset is provided.
    if ~isempty(data_test)
        optionsmanopt.statsfun = @compute_test_error;
    end
    function [stats, store] = compute_test_error(problem, X, stats, store)
        Wtest.U1 = full(A1test*X.U1);
        Wtest.U2 = full(A2test*X.U2);
        Wtest.U3 = full(A3test*X.U3);
        Wtest.G = full(X.G);
    
        residual_vec_test = compute_residual(Wtest, data_test);
        pred_test = residual_vec_test + data_test.entries;
        
        test_rmse = calcError(pred_test, data_test.entries, 'rmse'); % Reusing PJ code.;
        stats.test_rmse = test_rmse;
        
        
        if options.computenmse
            test_nmse = calcError(pred_test, data_test.entries, 'nmse'); % Reusing PJ code.
            stats.test_nmse = test_nmse;
            fprintf('Test NMSE %e\n',test_nmse);
        end

        if options.computeauc
            test_auc = calcError(pred_test, data_test.entries, 'auc'); % Reusing PJ code.
            stats.test_auc = test_auc;
            fprintf('Test AUC %e\n',test_auc);
        end
        
        Wtrain.U1 = full(A1*X.U1);
        Wtrain.U2 = full(A2*X.U2);
        Wtrain.U3 = full(A3*X.U3);
        Wtrain.G = full(X.G);

        residual_vec_train = compute_residual(Wtrain, data_train);
        pred_train = residual_vec_train + data_train.entries;
        
        train_rmse = calcError(pred_train, data_train.entries, 'rmse'); % Reusing PJ code.;
        stats.train_rmse = train_rmse;
        

        fprintf('Train RMSE %e      Test RMSE %e\n', train_rmse, test_rmse);

    end
    
    
    % Call appropriate algorithm
    solver = options.solver; % Any algorithm that Manopt supports
    optionsmanopt.maxiter = options.maxiter;
    optionsmanopt.verbosity =  options.verbosity;
    optionsmanopt.maxinner = options.maxinner;
    optionsmanopt.storedepth = 5;
    
    
    % Stopping criteria options
    optionsmanopt.stopfun = @mystopfun;
    function stopnow = mystopfun(problem, Y, info, last) %#ok<INUSL>
        gradnormlist = [info.gradnorm];
        norm0 = gradnormlist(1);
        normcurrent = gradnormlist(end);
        stopnow = (last >= 3 && normcurrent < norm0*options.reltolgradnorm);
    end
    
    
    if strcmp(func2str(options.linesearch), 'linesearchguess2')
        problem.linesearch = @linesearchguess2;
    elseif strcmp(func2str(options.linesearch), 'linesearchdefault')
        optionsmanopt.linesearch = @linesearch;
    else
        warning('RMLMTL:linesearch', ...
            'Linesearch is not properly defined. We work with default Manopt option.\n3');
        optionsmanopt.linesearch = @linesearch;
    end
    
    
    % Call the solver
    [Xsol, unused, infos] = solver(problem, X_initial, optionsmanopt);
    
    
    
    % Third party Mex files by Michael Steinlechner <michael.steinlechner@epfl.ch> (MS)
    
    % Use of MS Mex file on computing entries of a tensor stored in Tucker format
    function vals = compute_residual(W, A_Omega)
        %CALCGRADIENT Calculate the euclid. gradient of the obj. function
        %   Wrapper function for calcGradient_mex.c
        %
        %   Computes the euclid. gradient of the objective function
        %
        %         W_Omega - A_Omega
        %
        %   between a sparse tensor A_Omega and a Tucker tensor W.
        
        %   GeomCG Tensor Completion. Copyright 2013 by
        %   Michael Steinlechner
        %   Questions and contact: michael.steinlechner@epfl.ch
        %   BSD 2-clause license, see LICENSE.txt
        
        vals = calcGradient_mex(A_Omega.subs', A_Omega.entries, ...
            W.G, W.U1', W.U2', W.U3');
        
    end
    
    % Use of MS Mex file on computing entries of a tensor stored in Tucker format
    function valsdot = compute_residual_dot(W, A_Omega, Wdot)
        %CALCGRADIENT Calculate the euclid. gradient of the obj. function
        %   Wrapper function for calcGradient_mex.c
        %
        %   Computes the euclid. gradient of the objective function
        %
        %         W_Omega - A_Omega
        %
        %   between a sparse tensor A_Omega and a Tucker tensor W.
        
        %   GeomCG Tensor Completion. Copyright 2013 by
        %   Michael Steinlechner
        %   Questions and contact: michael.steinlechner@epfl.ch
        %   BSD 2-clause license, see LICENSE.txt
        
        vals = calcGradient_mex(A_Omega.subs', A_Omega.entries, Wdot.G, W.U1', W.U2', W.U3') ...
            + calcGradient_mex(A_Omega.subs', A_Omega.entries, W.G, Wdot.U1', W.U2', W.U3') ...
            + calcGradient_mex(A_Omega.subs', A_Omega.entries, W.G, W.U1', Wdot.U2', W.U3') ...
            + calcGradient_mex(A_Omega.subs', A_Omega.entries, W.G, W.U1', W.U2', Wdot.U3');
        
        valsdot = vals + 4*A_Omega.entries;
    end
    
    function cost = compute_cost(W, A_Omega)
        %CALCFUNCTION Calculate the value of the objective function
        %   Wrapper function for calcFunction_mex.c
        %
        %   Computes the value of the objective Function
        %
        %       0.5 * || W_Omega - A_Omega ||^2
        %
        %   See also calcGradient
        
        %   GeomCG Tensor Completion. Copyright 2013 by
        %   Michael Steinlechner
        %   Questions and contact: michael.steinlechner@epfl.ch
        %   BSD 2-clause license, see LICENSE.txt
        
        cost = 0.5 * calcFunction_mex(A_Omega.subs', A_Omega.entries, ...
            W.G, W.U1', W.U2', W.U3');
    end
    
    
    
    function t = compute_stepsize_initial(W, eta, residual_vec, subs)
        %CALCINITIAL Calculate the initial guess for the line search.
        %
        %   Wrapper function for calcInitial_mex.c
        %
        %   See also calcGradient, calcFunction
        %
        
        %   GeomCG Tensor Completion. Copyright 2013 by
        %   Michael Steinlechner
        %   Questions and contact: michael.steinlechner@epfl.ch
        %   BSD 2-clause license, see LICENSE.txt
        
        t = -calcInitial_mex(subs', residual_vec, ...
            W.G, W.U1', W.U2', W.U3',...
            eta.G, eta.U1', eta.U2', eta.U3');
    end
    
    
    
end











