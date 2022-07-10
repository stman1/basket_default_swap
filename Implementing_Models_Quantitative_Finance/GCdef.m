function tau = GCdef(rho,h,sim) % Gaussian copula default times

randn('state',10) % fixed seed
N = length(h);
S = ones(N,N).*(rho^2) + diag(ones(N,1),0).*(1 -(rho^2) );


% Gaussian Copula Simulation
x = mvnrnd(zeros(1,N),S,sim)'; 
u  = normcdf(x);
tau = -repmat((h.^-1)',1,sim).*log(u);