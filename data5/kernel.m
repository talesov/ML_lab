%�˺���
function K = kernel(X, Y, type, gamma)
    switch type
        case 'linear' %���Ժ˺���
            K = X * Y';
        case 'rbf'%rbf��˹�˺���
            k =@(x,z) exp(-(norm(x - z).^2).*gamma);%���ú���
            n = length(X);
            K = zeros(n);
            for i = 1:n
                for j = 1:n
                    K(i,j) = k(X(i, :),Y(j, :));
                end
            end
    end
end