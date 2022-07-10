function y=prob(n,h,t,rho,model)

switch model

    case 'FFT'

        y=quadv(@convolution,-7,7,[],[],h,t,rho,n);

    case 'HW'

        y=quadv(@hw,-7,7,[],[],h,t,rho,n);

end


