function K = create_kernel(x1,x2,k_type,k_par)

switch k_type
    case 'gauss'
        %K	= exp(-dist(x1,x2)./(k_par^2));  %%%%%
        K	= exp(-dist(x1,x2)*k_par);  %%%%%
    case 'linear'
        K = x1*x2';
    case 'poly'
        K = (x1*x2'+1).^k_par;
end


function distance = dist(X,Y)
nx	= size(X,1);
ny	= size(Y,1);
distance=sum((X.^2),2)*ones(1,ny) +...
    ones(nx,1)*sum((Y.^2),2)' - 2*(X*Y');