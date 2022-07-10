% graphs

j = 1;
for i=100000:10000:1000000
    bps(j) =ntd('MC',5,0.1,ones(1,5)*0.01,0.05,0.5,1,1,i);
    j = j+1
end

% The function ntd(model,T,rho,h,r,R,k,f) returns the annual premium ( in bps)
% paid f times/yr for a k-th to default basket swap.
% model can be 'FFT' or 'HW';
% T is the maturity of the contract;
% rho can be an inhomogeneous 1xn vector of the correlations to the factor;
% h can be an inhomogeneous 1xn vector of the hazard rates;
% r is the i.i.interest;
% R,the recovery rate, is supposed to be common;
% k specifies the contract (1st, 2nd, 3rd..nth to default);
% f is the interannual number of payments.
% sim is the number of MonteCarlo simulations