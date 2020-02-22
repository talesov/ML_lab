function [ multiSVMstruct ] = multiSVMTrain(traindata, n, c) 
%多类别的SVM训练器 
    for i = 1:n-1 
        for j = i+1:n 
            X = [traindata(5*(i-1)+1:5*i,:); traindata(5*(j-1)+1:5*j,:)]; 
            Y = [ones(5,1);zeros(5,1)]; 
            multiSVMstruct{i}{j} = fitcsvm(X, Y, 'KernelFunction', 'kernel', 'BoxConstraint', c);
        end
    end
end
