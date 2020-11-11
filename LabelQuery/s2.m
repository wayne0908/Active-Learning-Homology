function [x, Dist, Pred, G] = s2(G, f, L, Dist, Pred)
% Select the median node of the shortest shortest path connecting
% opposite labeled nodes in G, if any, for querying during the focus phase
% of S2 algorithm.

n = size(G,1);

x = L(end); % last query
xlabel = f(x); % label of last query

opp = find(f==-xlabel); % opposite labeled nodes
cuts = opp(G(x, opp)>0); % obvious cuts

% remove obvious cuts and update shortest paths that use them
if ~isempty(cuts)
    G(x,cuts)=0; G(cuts,x)=0;
    
    for i=1:size(L,2)-1     % check paths from all labeled nodes, except last one
        l = L(i);
        opp_nodes = find(f==-f(l));
        if ~isempty(opp_nodes)
            
            for j=1:size(cuts,1)
                c = cuts(j);
                if Pred(l,c)==x || Pred(l,x)==c  % check if cut edge is used in shortest paths
                    indexed_opp = zeros(1,n);
                    indexed_opp(opp_nodes(Dist(opp_nodes, x)==Inf)) = 1; % oppositely labeled with path not computed already
                    [Dist(l,:), Pred(l,:)] = all_sp(G, l, indexed_opp);
                    break;
                end
            end
        end
        
    end
end

% compute shortest paths from new node to all other oppositely labeled
% nodes that do not already have a path to this node
if ~isempty(opp)
    indexed_opp = zeros(1,n);
    indexed_opp(opp(Dist(opp, x)==Inf))=1; %  oppositely labeled with path not computed already
    [Dist(x,:), Pred(x,:)] = all_sp(G, x, indexed_opp);
else
    x=0;
    return;
end

Dist_opp = Inf(size(Dist));

P=find(f==1); % positive queries
N=find(f==-1); % negative queries
Dist_opp(P, N) = Dist(P, N); Dist_opp(N,P) = Dist(N,P);

[val, idx] = min(Dist_opp(:));

if val == Inf,
    x=0;
    return
end

[row, col] = ind2sub(size(Dist_opp), idx);

sp=[]; while col~=0, sp(end+1)=col; col=Pred(row,col); end; sp=fliplr(sp); % shortest shortest path

x = sp(floor(median(1:length(sp))));  % median of s2 path

end
