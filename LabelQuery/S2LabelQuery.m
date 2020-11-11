function [QueryIndex, Label, Flags] = S2LabelQuery(G, Data)

Oracle = Data(:, end);
Oracle(Oracle==0) = -1;
Budget = length(Data);
Priority = ones(length(Data),1)./length(Data);

[QueryIndex, Label, Flags] = s2_al(G, Oracle, Priority, Budget);                             





