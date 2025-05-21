clear all; close all

load 'Reaches81.mat'

nReach=length(ReachData81.reach_id);

m=0;
for i=1:nReach
    %check upstream
    for j=1:ReachData81.n_rch_up(i)
       k=find(ReachData81.reach_id==ReachData81.rch_id_up(i,j)); 
       if ~any(ReachData81.rch_id_dn(k,:)==ReachData81.reach_id(i))
           m=m+1;
           disp(['Issue #' num2str(m) ': reach ' num2str(ReachData81.reach_id(i)) ' lists reach ' num2str(ReachData81.rch_id_up(i,j)) ' as being upstream, but ' ...
                 'reach ' num2str(ReachData81.rch_id_up(i,j)) ' does not list reach '  num2str(ReachData81.reach_id(i)) ' as being downstream.' ])
       end
    end
    %check downstream
    for j=1:ReachData81.n_rch_down(i)
       k=find(ReachData81.reach_id==ReachData81.rch_id_dn(i,j)); 
       if ~any(ReachData81.rch_id_up(k,:)==ReachData81.reach_id(i))
           m=m+1;
           disp(['Issue #' num2str(m) ': reach ' num2str(ReachData81.reach_id(i)) ' lists reach ' num2str(ReachData81.rch_id_dn(i,j)) ' as being downstream, but ' ...
                 'reach ' num2str(ReachData81.rch_id_dn(i,j)) ' does not list reach '  num2str(ReachData81.reach_id(i)) ' as being upstream.' ])           
       end
    end
    
end