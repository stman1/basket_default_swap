function nk = N_t(t, tau)
[N_tau, M_tau]=size(tau);
nk = max((tau-ones(N_tau, M_tau).*t)<0,0);