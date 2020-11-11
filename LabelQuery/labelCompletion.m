function f = labelCompletion(G, f, feat)
% Given a graph G (n*n symmetric matrix: 1=edge 0=no edge),
% and a partial node label vector (n vector: 1=positive, 0=unlabeled, -1=negative),
% return the complete node label vector.
% This implementation uses the harmonic function solution.

% Dist = pdist2(feat, feat);
% Dist(G == 0) = 0;
% G = Dist;
n = size(G,1);
L = find(f);
l = length(L);
U = setdiff((1:n)', L);
W = G([L; U], [L; U]); % shuffle indices so labeled data come first
Laplacian = diag(sum(W))-W;
fu = - inv(Laplacian(l+1:n, l+1:n) + 0.01 * eye(length(U)))*Laplacian(l+1:n,1:l)*f(L); % add a small multiple of identity matrix as regularization
f(U) = (fu>=0)*2-1;



