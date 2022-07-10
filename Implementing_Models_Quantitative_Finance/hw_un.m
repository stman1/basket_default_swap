function y=hw_un(x,h,t,rho,n)

u=zeros(1,n);

u1=zeros(1,n+1);

q=exp(-h.*t);

sc=normcdf((norminv(q)-rho.*x)./(sqrt(1-rho.^2)));

w=(1-sc)./sc;

for k=1:n

    for i =1:n

        v(i)=sum(w.^i);

    end

    u(1)=v(1);

    for z=2:k

        as=0;

        j=z-1;

        for i=1:k

            as=as+((-1)^(i+1)*v(i)*u(j));

            if j==1

                break

            else j=j-1;

            end
        end

        u(z)=(as+(-1)^(z+1)*v(z))/z;

    end

    u1(1,k+1)=u(k);
end

u1(1,1)=1;

y=prod(sc)*u1.*normpdf(x,0,1);