function y=hw(x,h,t,rho,n)

u=zeros(1,n);

u1=zeros(1,n+1);

q=exp(-h.*t);

sc=normcdf((norminv(q)-rho.*x)./(sqrt(1-rho.^2)));

w=(1-sc)./sc;

for k=1:n

    v = repmat(sum(w), [1, n]);

    t = w;

    for i = 2:n,

        t = t .* w;

        v(i) = sum(t);

    end

    u = v;

    sv = (-1).^(2:k) .* v(1:k-1);

    sgn = -1;

    for z = 2:k,

        as = 0;

        for i = 1:z-1

            as = as + sv(i)*u(z-i);

        end

        u(z) = (as + sgn*v(z)) / z;

        sgn  = sgn * (-1);

    end

    u1(1,k+1)=u(k);

end

u1(1,1)=1;

y=prod(sc)*u1.*normpdf(x,0,1);

