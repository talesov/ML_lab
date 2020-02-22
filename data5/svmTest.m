function res = svmTest(svm, Xt, Yt, type, gamma)
    switch type
        case 'linear'
            omega = (svm.alpha .* svm.Ysv)' * svm.Xsv;
            tmp = kernel(svm.Xsv, svm.Xsv, type) * (svm.Ysv .* svm.alpha);
            b = mean(svm.Ysv - tmp);
            
            Y = 1 - Yt .* (Xt * omega' + b * ones(length(Yt), 1));
            Y = sign(Y); %f(x) 
            
            res.acc = length(find(Y ~= 1)) ./ length(Y);
            res.Y = Y;
            
            %»æ±ß½çÍ¼
            f = @(z)(- b - omega(1)*z) ./ omega(2);
            g = @(z)(1 - b - omega(1)*z) ./ omega(2);
            h = @(z)(-1 - b - omega(1)*z) ./ omega(2);
            lower = floor(min(Xt(:, 1))); upper = ceil(max(Xt(:, 1)));
            hold on
            fplot(@(x) f(x), [lower upper], 'k-');
            hold on
            fplot(@(x) g(x), [lower upper], 'k--');
            hold on
            fplot(@(x) h(x), [lower upper], 'k--');
        case 'rbf'
            K=@(x,z) exp(-(norm(x - z).^2).*gamma);
            Y = zeros(length(Yt),1);
            for i=1:length(Yt)
                for j=1:svm.len
                    Y(i) = Y(i) + svm.alpha(j) * svm.Ysv(j) * K(svm.Xsv(j, :),Xt(i, :));
                end
            end
            Y = sign(Y); %f(x) 
            res.acc = length(find(Y == Yt)) ./ length(Y);
            res.Y = Y;
    end
end