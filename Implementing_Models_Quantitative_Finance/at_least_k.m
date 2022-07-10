function y=at_least_k(n,rho,h,t,k,model)

aa=prob(n,h,t,rho,model);

y=sum(aa(k+1:end));
