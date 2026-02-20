function [ ReachBoundaries ] = MergeShortReachesVarMin(ReachBoundaries,FlowDist,Concavity,Width)
%This function merges short reaches by identifying which of the immediate
%neighbors is more similar to a short reach. 
%Renato Frasson December 30, 2016
%
%List of inputs
%ReachBoundaries  : Location of the reach boundaries expressed in terms of
%                   the indices of the vector x
%FlowDist         : Flow distance calculated at the nodes in m
%Concavity        : Estimate of water surface concavity at the nodes
%Widths           : River widths in meters

%List of outputs
%ReachBoundaries  : Location of the reach boundaries expressed in terms of
%                   the indices of the vector x

    ReachLength=zeros(length(ReachBoundaries)-1,1);
    for count=1:length(ReachBoundaries)-1;
        ReachLength(count)=FlowDist(ReachBoundaries(count+1))-FlowDist(ReachBoundaries(count));
    end
    [MinLength,ReachID ]=min(ReachLength);
    MinReachLen=zeros(length(ReachBoundaries)-1,1);
    for count=1:length(ReachBoundaries)-1
        MinReachLen(count)=min(mean(Width(ReachBoundaries(count):ReachBoundaries(count+1))),(FlowDist(end)-FlowDist(1))/2);
        MinReachLen(count)=max(MinReachLen(count),100);
    end 
    while MinLength(1)<MinReachLen(ReachID)&&length(ReachBoundaries)>2
        %merge reaches until all are larger than the minimum length
        %find the smaller reach. merge it with the surroundings until it is
        %larger than the minimum length. Once that reach is large enough, than
        %sort again and proceed with merging    
        if ReachID>1
        %the reach is not the first in the series, so check if its
        %sinuisity is closer to the up or downstream's
            if ReachID+2<=length(ReachBoundaries)
                %then there is an upstream reach, otherwhise, ReachID is
                %the last reach
                AveSinCurr=mean(Concavity(ReachBoundaries(ReachID):ReachBoundaries(ReachID+1)));
                AveSdownstr=mean(Concavity(ReachBoundaries(ReachID+1):ReachBoundaries(ReachID+2)));
                AveSupstr=mean(Concavity(ReachBoundaries(ReachID-1):ReachBoundaries(ReachID)));
                if abs(AveSdownstr-AveSinCurr)<abs(AveSupstr-AveSinCurr)
                    %current is more similar to the downstream
                    ReachBoundaries=[ReachBoundaries(1:ReachID);ReachBoundaries(ReachID+2:length(ReachBoundaries))];
                    ReachLength(ReachID)=FlowDist(ReachBoundaries(ReachID+1))-FlowDist(ReachBoundaries(ReachID));
                    ReachLength=[ReachLength(1:ReachID);ReachLength(ReachID+2:length(ReachLength))];
                    MinReachLen(ReachID)=min(mean(Width(ReachBoundaries(ReachID):ReachBoundaries(ReachID+1))),(FlowDist(end)-FlowDist(1))/2);
                    MinReachLen=[MinReachLen(1:ReachID);MinReachLen(ReachID+2:length(MinReachLen))];
                else
                    %current is more similar to the upstream
                    ReachBoundaries=[ReachBoundaries(1:ReachID-1);ReachBoundaries(ReachID+1:length(ReachBoundaries))];
                    ReachLength=[ReachLength(1:ReachID-1);ReachLength(ReachID+1:length(ReachLength))];
                    MinReachLen=[MinReachLen(1:ReachID-1);MinReachLen(ReachID+1:length(MinReachLen))];
                    ReachID=ReachID-1;
                    ReachLength(ReachID)=FlowDist(ReachBoundaries(ReachID+1))-FlowDist(ReachBoundaries(ReachID));   
                    MinReachLen(ReachID)=min(mean(Width(ReachBoundaries(ReachID):ReachBoundaries(ReachID+1))),(FlowDist(end)-FlowDist(1))/2);         
                end
            else
                %ReachID is the last reach, so merge with upstream
                ReachBoundaries=[ReachBoundaries(1:ReachID-1);ReachBoundaries(ReachID+1)];
                ReachID=ReachID-1;
                MinReachLen=MinReachLen(1:ReachID);
                ReachLength=ReachLength(1:ReachID);
                ReachLength(ReachID)=FlowDist(ReachBoundaries(ReachID+1))-FlowDist(ReachBoundaries(ReachID));
                MinReachLen(ReachID)=min(mean(Width(ReachBoundaries(ReachID):ReachBoundaries(ReachID+1))),(FlowDist(end)-FlowDist(1))/2);
            end
        else
            %Since it is the first reach, merge first and second reaches
            ReachBoundaries=[ReachBoundaries(1);ReachBoundaries(3:length(ReachBoundaries))];
            ReachLength=ReachLength(2:length(ReachLength));
            ReachLength(1)=FlowDist(ReachBoundaries(2))-FlowDist(ReachBoundaries(1));
        end
        [MinLength,ReachID ]=min(ReachLength);
    end


end

