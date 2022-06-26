clear all
close all
addpath('functions')

%% specify simulation

% rules
% action (left-right), state (A,B,C), future state (A', B', C')
taskparam.tnoise=0.1;
taskparam.T{1}(1,:,:)=[0 1 0; 0 0 1; 1 0 0];
taskparam.T{1}(2,:,:)=[0 0 1; 1 0 0; 0 1 0];
% uncontrollable rule
taskparam.T{2}(1,:,:)=[0 0 1; 1 0 0; 0 1 0];
taskparam.T{2}(2,:,:)=[0 0 1; 1 0 0; 0 1 0];

% rule reversals
taskparam.rulelength=repmat([500 500 100 100 100 100],1,1); % the length of each streak does not matter (we only need enough to estimate MI and TE accurately)
taskparam.rulelength=taskparam.rulelength;
%% parameter ranges for simulated agents
% the experiment consists of 3 states, each offering a binary action which
% activates the next state.
% the agent must therefore predict the upcoming state based on a
agentparamRange.baseLR=[0:0.01:1];
agentparamRange.stateslope=[0:0.1:10];
agentparamRange.pokeslope=[0:0.1:10];
agentparamRange.slopeOmega=[0:0.1:10];
agentparamRange.thresholdOmega=[-0.5:0.01:0.5];
agentparamRange.omegaLR=[0:0.01:1];

%% VBA fit
% hidden state mappings
in.map.Actor=reshape(1:numel(ones(2,3,3)),2,3,3);
in.map.Spectator=numel(in.map.Actor)+reshape(1:numel(ones(3,3)),3,3);
in.map.Omega=max(max(in.map.Spectator))+1;
in.map.Arbitrator=max(max(in.map.Spectator))+2;
dim.n=in.map.Arbitrator;
options.priors.muX0=(1/3)*ones(dim.n,1);
options.priors.muX0(in.map.Omega)=0.5;
options.priors.muX0(in.map.Arbitrator)=0.5;

% parameter transformations
Traw = @(x) x; Texp = @exp; Tinvexp = @(x) 1./exp(x); Tsig = @(x) 1./(1+exp(-x));
TsigMin1to1 = @(x) -1+(2./(1+exp(-x)));
Tsig0to10 = @(x) 10./(1+exp(-x));

% evolution priors for recovery
dim.n_theta=4;
options.priors.muTheta=[0;0;0;0];
options.priors.SigmaTheta=3*eye(dim.n_theta);
options.priors.SigmaTheta(2,2)=3;
options.inF=in;
options.inF.paramtransform={Tsig,Tsig0to10,TsigMin1to1,Tsig};
% observation priors for recovery
dim.n_phi=2;
options.priors.muPhi=[0;0;];
options.priors.SigmaPhi=10*eye(dim.n_phi);
options.inG=in;
options.inG.paramtransform={Traw, Traw};
% observations sources
options.sources(1).type=1; % poke choice (binomial)
options.sources(1).out=1;
options.sources(2).type=2; % state choice (multinomial)
options.sources(2).out=2:4;
% other VBA options
options.DisplayWin=0;
options.updateX0=0;

nsimulations=500;

for sim=1:nsimulations
    
    pickids=randi(101,1,6);
    agentparam.baseLR=agentparamRange.baseLR(pickids(1));
    agentparam.slopeOmega=agentparamRange.slopeOmega(pickids(2));
    agentparam.thresholdOmega=agentparamRange.thresholdOmega(pickids(3));
    agentparam.omegaLR=agentparamRange.omegaLR(pickids(4));
    agentparam.stateslope=agentparamRange.stateslope(pickids(5));
    agentparam.pokeslope=agentparamRange.pokeslope(pickids(6));

    [fullNLL, log, yPoke, yState] = simulation_function(taskparam,agentparam);
    simNLL(sim,:)=sum(fullNLL);
    
    % populate u
    u(1,:)=[NaN log(1:end-1,1)']; % previous state
    u(2,:)=[NaN log(1:end-1,4)']; % explored state
    u(3,:)=[NaN log(1:end-1,2)']; % action (poke)
    u(4,:)=[NaN log(1:end-1,5)']; % reward
    u(5,:)=log(:,1)'; % current state
    u(6,:)=log(:,2)'; % current poke
    
    % 
    y=[yPoke; yState];
    
    [post out]=VBA_NLStateSpaceModel(y,u, @vba_agent_evof, @vba_agent_obsf,dim,options);
    
    % retransform parameters
    for p=1:length(post.muTheta)
        thetaFitted(p,1)=options.inF.paramtransform{p}(post.muTheta(p));
    end
    
    % retransform parameters
    for p=1:length(post.muPhi)
        phiFitted(p,1)=options.inG.paramtransform{p}(post.muPhi(p));
    end
    
    recovered.baseLR(sim,:)=[agentparam.baseLR thetaFitted(1,1)];
    recovered.slopeOmega(sim,:)=[agentparam.slopeOmega thetaFitted(2,1)];
    recovered.thresholdOmega(sim,:)=[agentparam.thresholdOmega thetaFitted(3,1)];
    recovered.omegaLR(sim,:)=[agentparam.omegaLR thetaFitted(4,1)];
    
    recovered.stateslope(sim,:)=[agentparam.stateslope phiFitted(1,1)];
    recovered.pokeslope(sim,:)=[agentparam.pokeslope phiFitted(2,1)];

    recoveredHidden.OmegaSim(sim,:)=[log(:,7)'];
    recoveredHidden.OmegaFit(sim,:)=post.muX(options.inF.map.Omega,:);
    recoveredHidden.ArbSim(sim,:)=[log(:,8)'];
    recoveredHidden.ArbFit(sim,:)=post.muX(options.inF.map.Arbitrator,:);
        

    fitinfo{sim,1}=out.fit;
    F(sim,1)=out.F;
    
    
end

save('sixport_simulation.mat');
