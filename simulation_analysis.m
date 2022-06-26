clear all

%
load('sixport_simulation.mat');

paramnames=fieldnames(recovered);
clear parval
for p=1:length(paramnames)
    parval{p}=getfield(recovered,paramnames{p});
    [r_rec(p)]=corr(parval{p}(:,1),parval{p}(:,2));       
end

for p=1:length(paramnames)
    for pp=1:length(paramnames)
        [r_mat(p,pp)]=corr(parval{p}(:,2),parval{pp}(:,2));
    end
end

r_mat(eye(length(r_mat))==1)=r_rec;

% parameter recovery (in diagonal) and parameter correlation (off diagonal)
figure('name', 'parameter recovery (in diagonal) and parameter correlation (off diagonal)', 'color','white')
imagesc(r_mat)
xticklabels(paramnames)
yticklabels(paramnames)
colorbar
rotateXLabels(gca(),45);

disp('correlation matrix of model parameters')
disp(num2str(r_mat))

% recovery of controllability-related hidden variables
for s=1:size(recoveredHidden.OmegaSim,1)

    OmegaRec(s,1)=corr(recoveredHidden.OmegaSim(s,:)',recoveredHidden.OmegaFit(s,:)');
    ArbRec(s,1)=corr(recoveredHidden.ArbSim(s,:)',recoveredHidden.ArbFit(s,:)');
    
end

disp(['average within subject correlation of Omega: ' num2str(nanmean(OmegaRec))])
disp(['average within subject correlation of Arb: ' num2str(nanmean(ArbRec))])
disp(['average between subject correlation of Omega: ' num2str(corr(nanmean(recoveredHidden.OmegaSim,2),nanmean(recoveredHidden.OmegaFit,2)))])
disp(['average between subject correlation of Arb: ' num2str(corr(nanmean(recoveredHidden.ArbSim,2),nanmean(recoveredHidden.ArbFit,2)))])
