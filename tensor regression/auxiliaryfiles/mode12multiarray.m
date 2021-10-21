function X = mode12multiarray(X1,tensordims)
    % Converts a mode 1 unfolding matrix to its multiarray.
    n1 = tensordims(1);
    n2 = tensordims(2);
    n3 = tensordims(3);
    
    X = reshape(X1, n1, n2, n3);
    
end