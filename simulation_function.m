function [fullNLL, log, yPoke, yState] = simulation_function(taskparam,agentparam)

% % controllable rule
% % action (left-right), state (A,B,C), future state (A', B', C')
% taskparam.tnoise=0;
% 
% taskparam.T{1}(1,:,:)=[0 1 0; 0 0 1; 1 0 0];
% taskparam.T{1}(2,:,:)=[0 0 1; 1 0 0; 0 1 0];
% 
% % uncontrollable rule
% taskparam.T{2}(1,:,:)=[0 0 1; 1 0 0; 0 1 0];
% taskparam.T{2}(2,:,:)=[0 0 1; 1 0 0; 0 1 0];
% 
% % specify simulation
% taskparam.ruleordering=repmat([1 2],1,100); % the sequence of conditions does not matter
% taskparam.rulelength=repmat([200 200 200 200],1,4); % the length of each streak does not matter (we only need enough to estimate MI and TE accurately)
% 
% % initialize the beliefs of agents
% % the experiment consists of 3 states, each offering a binary action which
% % activates the next state.
% % the agent must therefore predict the upcoming state based on a
% agentparam.baseLR=0.3;
% agentparam.stateslope=5;
% agentparam.pokeslope=5;
% agentparam.slopeOmega=100;
% agentparam.thresholdOmega=-0.2;
% agentparam.omegaLR=0.2;

[beliefs] = agent('initialize',agentparam,[]);

% helper variables
changepoints=cumsum(taskparam.rulelength);
ruleordering=repmat([1 2],1,length(changepoints)+1);

% dynamic variables
a=nan(changepoints(end),1);
s=nan(changepoints(end)+1,1);
sa=nan(changepoints(end)+1,1);
reward=nan(changepoints(end)+1,1);
fullNLL=nan(changepoints(end),2);
yPoke=zeros(1,changepoints(end));
yState=zeros(3,changepoints(end));

% initial values
s(1)=1;
rev=1;
actrule=taskparam.T{1};

for t=1:changepoints(end)
    
    % each trial start with a poke left or right
    [beliefs,actionProb, action, NLLpoke] = agent('pokechoice',agentparam, beliefs,s(t),[],[],[]);
%      a(t,1)=2-double(actionProb(1)>rand);
    a(t,1)=action; %3-action;
    pokeProb=actionProb(a(t,1));
    yPoke(1,t)=double(action==2);

    % is there a change in the rule?
    if ismember(t,changepoints)
        if t==changepoints(end)+1
            break
        end
        rev=rev+1;
        actrule=taskparam.T{ruleordering(rev)};
    end

    % state action
    [beliefs,actionProb,action,NLLstate] = agent('statechoice',agentparam, beliefs,s(t),[],a(t),[]);
    singlerand=rand;
    sa(t+1,1)=action;
    stateProb=actionProb(sa(t+1,1));
    yState(action,t)=1;

    % define next state
    s(t+1,1)=find(squeeze(actrule(a(t,1),s(t,1),:))==max(squeeze(actrule(a(t,1),s(t,1),:))));
    if rand<taskparam.tnoise
        s(t+1,1)=find(~ismember([1 2 3], [s(t+1,1) s(t,1)]));
        tnoisy=1;
    else
        tnoisy=0;
    end
    
    % decide if successful transition
    if sa(t+1,1)==s(t+1,1)
        reward(t+1)=1;
    else
        reward(t+1)=0;
        s(t+1,1)=s(t,1); % the state does not change until a correct poke
    end
        
    % update
    [beliefs] = agent('update',agentparam,beliefs,s(t),sa(t+1),a(t,1),reward(t+1));

    % standard log
    log(t,:)=[s(t) a(t) s(t+1,1) sa(t+1,1) reward(t+1) tnoisy beliefs.Omega beliefs.Arbitrator pokeProb stateProb ruleordering(rev)];
 
    fullNLL(t,:)=[NLLpoke NLLstate];
    
end

end
% % figure
% plot([1:size(log,1)],log(:,5))';
% hold on
% plot([1:size(log,1)],log(:,9))';
% hold on
% plot([1:size(log,1)],log(:,6))';
