function y=convolution(x,h,t,rho,n)

f=1-exp(-h.*t);

p=normcdf(((norminv(f)-rho.*x)./(sqrt(1.-rho.^2))));

p1=zeros(n+1,n);

p1(1,:)=1-p;

p1(n+1,:)=p;

p2=fft(p1);

p3=prod(p2,2)';

p4=ifft(p3);

y=p4.*normpdf(x,0,1);




