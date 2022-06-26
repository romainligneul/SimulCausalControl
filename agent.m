function [beliefs,actionProb, actionCat, NLL] = agent(agentmode,agentparam,beliefs,paststate,state,action,reward, varargin)
% AGENT_RL

if isfield(beliefs, 'in')
    
    beliefs.Actor=beliefs.x(beliefs.in.map.Actor);
    beliefs.Spectator=beliefs.x(beliefs.in.map.Spectator);
    beliefs.Omega=beliefs.x(beliefs.in.map.Omega);
    beliefs.Arbitrator=beliefs.x(beliefs.in.map.Arbitrator);

end
    
switch agentmode
    
    case 'initialize'
        
        beliefs.Actor(1,:,:)=[1 1 1; 1 1 1; 1 1 1]/3;
        beliefs.Actor(2,:,:)=[1 1 1; 1 1 1; 1 1 1]/3;
        beliefs.Spectator=[1 1 1; 1 1 1; 1 1 1]/3;
        beliefs.Omega=0.5;
        beliefs.Arbitrator=1./(1+exp(-agentparam.slopeOmega*(beliefs.Omega + agentparam.thresholdOmega)));

    case 'update'
        
        % replace paststate, state, etc from u structure passed as supplementary
        % argument if available
        if ~isempty(varargin)
            u=varargin{1};
            paststate=u(1);
            state=u(2);
            action=u(3);
            reward=u(4);
        end
        
        if isnan(paststate)
            return
        end
        
        % assign parameters to structure if needed
        if ~isstruct(agentparam)
            P=agentparam;
            clear agentparam;
            agentparam.baseLR=P(1);
            agentparam.slopeOmega=P(2);
            agentparam.thresholdOmega=P(3);
            agentparam.omegaLR=P(4);
        end
        
        % update values
        ActorPE=agentparam.baseLR*(reward-beliefs.Actor(action,paststate,state));
        SpectatorPE=agentparam.baseLR*(reward-beliefs.Spectator(paststate,state));
        
        beliefs.Actor(action,paststate,state)=beliefs.Actor(action,paststate,state)+ActorPE;
        beliefs.Spectator(paststate,state)=beliefs.Spectator(paststate,state)+SpectatorPE;
        otherstates=find(~ismember([1 2 3],state));

        if reward<=0
            beliefs.Actor(action,paststate,otherstates) = beliefs.Actor(action,paststate,otherstates)-(ActorPE/2);
            beliefs.Spectator(paststate,otherstates) = beliefs.Spectator(paststate,otherstates)-(SpectatorPE/2);            
        else
            beliefs.Actor(action,paststate,otherstates) = beliefs.Actor(action,paststate,otherstates)*(1-agentparam.baseLR);
            beliefs.Spectator(paststate,otherstates) = beliefs.Spectator(paststate,otherstates)*(1-agentparam.baseLR); 
        end
        
        % update controllability
%          agentparam.omegaLR
        beliefs.Omega=beliefs.Omega+agentparam.omegaLR*(SpectatorPE-ActorPE-beliefs.Omega);
        beliefs.Arbitrator=1./(1+exp(-agentparam.slopeOmega*beliefs.Omega + agentparam.thresholdOmega));
        
        if isfield(beliefs, 'in')
            beliefs.x(beliefs.in.map.Actor)=beliefs.Actor;
            beliefs.x(beliefs.in.map.Spectator)=beliefs.Spectator;
            beliefs.x(beliefs.in.map.Omega)=beliefs.Omega;
            beliefs.x(beliefs.in.map.Arbitrator)=beliefs.Arbitrator;
        end
        
%         disp(['omega:' num2str(beliefs.Omega)]);
        
    case 'statechoice'
         
        % replace paststate, state, etc from u structure passed as supplementary
        % argument if available
        if ~isempty(varargin)
            u=varargin{1};
            paststate=u(5);
            action=u(6);
        end
        
        % assign parameters to structure if needed
        if ~isstruct(agentparam)
            P=agentparam;
            clear agentparam;
            agentparam.stateslope=P(1);
            agentparam.pokeslope=P(2);
        end
        
        % softmax
        composite=beliefs.Arbitrator*squeeze(beliefs.Actor(action,paststate,:))+(1-beliefs.Arbitrator)*squeeze(beliefs.Spectator(paststate,:)');
        for a=1:length(composite)
            actionProb(a,1)=exp(agentparam.stateslope*composite(a))/sum(exp(agentparam.stateslope*composite));
        end
        
        actionCat=find(rand<cumsum(actionProb),1,'first');
        NLL=-log(actionProb(action));
        
    case 'pokechoice'
        
        % replace paststate, state, etc from u structure passed as supplementary
        % argument if available
        if ~isempty(varargin)
            u=varargin{1};
            paststate=u(5);
        end
        
        % assign parameters to structure if needed
        if ~isstruct(agentparam)
            P=agentparam;
            clear agentparam;
            agentparam.stateslope=P(1);
            agentparam.pokeslope=P(2);
        end
               
        % softmax
        composite=max(squeeze(beliefs.Actor(:,paststate,:)),[],2);
        for a=1:length(composite)
            actionProb(a,1)=exp(agentparam.pokeslope*composite(a))/sum(exp(agentparam.pokeslope*composite));
        end   
        
        actionCat=find(rand<cumsum(actionProb),1,'first');
        NLL=-log(actionProb(action));
        
end


end

