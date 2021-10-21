function [err] = calcError(YPred, YAct, errType)
    
    % 1-D vectors of same length expected
    if (sum(size(YPred)) == length(YPred)+1) && (sum(size(YAct)) == length(YAct)+1) && (length(YPred) == length(YAct))
        if ~all(size(YPred)==size(YAct))
            YAct = YAct';
        end
    else
        fprintf('[calculateErr]Dimensionality mismmatch of YPred and YAct\n')
        keyboard;
    end
    
    % if ~all(size(YPred) == size(YAct))
    %     if all(size(YPred) == size(YAct'))
    %         YAct = YAct';
    %     else
    %         fprintf('[calculateErr]Dimensionality mismmatch of YPred and YAct\n')
    %         keyboard;
    %     end
    % end
    
    switch lower(errType)
        
        case {'class_error'}
            err = sum(sign(YPred) ~= sign(YAct))/length(YPred);
        case {'class_accuracy'}
            err = sum(sign(YPred) == sign(YAct))/length(YPred);
        case {'accuracy'}
            err = sum(YPred == YAct)/length(YPred);
        case {'rmse'}
            err = sqrt(norm(YPred-YAct)^2/length(YPred));
        case {'mse'}
            err = norm(YPred-YAct)^2/length(YPred);
        case {'nmse'}
            err = (norm(YPred-YAct)^2/length(YPred))/var(YAct); % mse/var
        case {'amse'}
            err = (norm(YPred-YAct)^2/length(YPred))/norm(YAct)^2; % mse/norm_sq
        case {'auc'}
            
            err = mycomputeAUC(YPred, YAct);
            
%             % Formula take paper - Confidence Intervals for the Area under the ROC Curve
%             % See Small-sample precision of ROC-related estimates - says AUC is bad
%             % not optimal implementation - n^2 complexity
%             npos = sum(YAct == 1);
%             nneg = sum(YAct == -1);
%             
%             if (npos~=0 && nneg~=0)
%                 auc = 0;
%                 posIndex = find(YAct == 1);
%                 negIndex = find(YAct == -1);
%                 
%                 for i=1:npos
%                     for j=1:nneg
%                         auc = auc + double(YPred(posIndex(i)) > YPred(negIndex(j)));
%                     end
%                 end
%                 err = auc/npos/nneg;
%             else
%                 fprintf('calculateAUC.m : zero positive or negative instances\n');
%                 keyboard;
%             end
            
        otherwise
            fprintf('\n[calculateErr] Unknown error type\n');
            keyboard;
    end
