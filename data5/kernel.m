%核函数
function K = kernel(X, Y, type, gamma)
    switch type
        case 'linear' %线性核函数
            K = X * Y';
        case 'rbf'%rbf高斯核函数
            k =@(x,z) exp(-(norm(x - z).^2).*gamma);%内置函数
            n = length(X);
            K = zeros(n);
            for i = 1:n
                for j = 1:n
                    K(i,j) = k(X(i, :),Y(j, :));
                end
            end
    end
end