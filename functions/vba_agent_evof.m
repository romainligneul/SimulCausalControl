function fx = vba_agent_evof(x,P,u,inF)
%VBA_AGENT_EVOF Wrapper around the agent for VBA toolbox evolution function

    for p=1:length(P)
        P(p)=inF.paramtransform{p}(P(p));
    end
    
    % agent beliefs
    beliefs.in=inF;
    beliefs.x=x;

    % 
    [beliefs] = agent('update',P,beliefs,[],[],[],[], u);
   
    fx=beliefs.x;
    
end

