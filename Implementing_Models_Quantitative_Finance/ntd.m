function bps=ntd(model,T,rho,h,r,R,k,f,sim)

format short

% The function ntd(model,T,rho,h,r,R,k,f, sim) returns the annual premium ( in bps)
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

n=length(h);

if rho<1
    switch model
        case 'MC'
        
        % Premium Leg
 
        tau = GCdef(rho,h,sim); % Gaussian copula default times
        pre = zeros(1,sim); % premium leg pvs (vector)

        for i =1:1:T*f % loop over all cash flow dates T x f (eg 5 years x f payments per year)
            temp = sum(N_t(i/f, tau), 1)
            A = max((sum(N_t(i/f, tau), 1)-ones(1,sim).*k)<0,0);
            pre = pre + A.*exp(-r*(i/f)); % discounting
        end
      
        %  Default Leg
        
        tau = [ones(1,sim)*Inf; sort(tau)];
        A = (max((sum(N_t(T, tau),1)-ones(1,sim).*k)>=0,0).*k)+1;
        temp = -tau(A)
        def = exp(-tau(A).*r).*(1-R);
        mean(def)
        bps=10000*(sum(def)/sum(pre.*f));
    
        otherwise
       
        % Premium Leg
 
        pre=0;
        for i =1:1:T*f 
            pre=pre+exp(-r*i/f)*(1-at_least_k(n,rho,h,i/f,k,model))/f;
        end
        
        %  Default Leg
        
        pr_T=at_least_k(n,rho,h,T,k,model);
        aux_f1=inline('at_least_k(n,rho,h,t,k,model)*exp(-t*r)','t','n','rho','h','k','r','model');
        aux_f2=quadv(aux_f1,0,T,[],[],n,rho,h,k,r,model);
        def=(exp(-r*T)*pr_T+r*aux_f2)*(1-R);
        bps=10000*def/pre;
        def
     end
   
else

    dsc=sort(h,'descend');
    bps=10000*(1-R)*dsc(k);

end