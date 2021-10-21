function X = mode32multiarray(X3,tensordims)
    % Converts a mode 3 unfolding matrix to its multiarray.
    n1 = tensordims(1);
    n2 = tensordims(2);
    n3 = tensordims(3);
    
    X = permute(reshape(X3, n3, n1, n2), [2 3 1]);
    
end