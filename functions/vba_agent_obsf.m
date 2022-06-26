function [ gx ] = vba_agent_obsf(x,P,u,inG)
%VBA_AGENT_EVOF Wrapper around the agent for VBA toolbox observation function

    for p=1:length(P)
        P(p)=inG.paramtransform{p}(P(p));
    end
    
    % agent beliefs
    beliefs.in=inG;
    beliefs.x=x;
    
    % 
    [~,gxPoke] = agent('pokechoice',P,beliefs,[],[],[],[], u);
    
    
    [~,gxState] = agent('statechoice',P,beliefs,[],[],[],[], u);

    gx=[gxPoke(2);gxState];
    
end
