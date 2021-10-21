function X = mode22multiarray(X2,tensordims)
    % Converts a mode 2 unfolding matrix to its multiarray.
    n1 = tensordims(1);
    n2 = tensordims(2);
    n3 = tensordims(3);
    
    X = permute(reshape(X2, n2, n1, n3), [2 1 3]);
    
end