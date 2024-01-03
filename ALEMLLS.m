function model = ALEMLLS( X, Y, optmParameter)
   %% optimization parameters
    lambda1          = optmParameter.lambda1; % YR
    lambda2          = optmParameter.lambda2; % W'W
    lambda3          = optmParameter.lambda3; % W regularization
    lambda4          = optmParameter.lambda4; % R regularization
    lambda5          = optmParameter.lambda5; % Projected Input Space-Instance Similarity laplacian
    lambda6          = optmParameter.lambda6; % Predictions-Correlation Laplacian
    
    d1frac = optmParameter.d1frac;
    
    etaW = optmParameter.etaW; 
    etaP = optmParameter.etaP;
    etaR = optmParameter.etaR;
    J = optmParameter.J;
    
    maxIter          = optmParameter.maxIter;
    rho            = optmParameter.rho;

    num_dim   = size(X,2);  % d
    num_class = size(Y,2);  % q
    num_inst =  size(X,1);  % n
    
    d1 = ceil(d1frac * num_dim); %dimension in projected space, P.
    
    XTX = X'*X;
    XTY = X'*Y;
    YTY = Y'*Y;
    YTX = Y'*X;
    
    %C = pdist2( Y'+eps, Y'+eps, 'cosine' );
    %L = diag(sum(C,2)) - C;
   %% initialization
    %W_S_1   = rand(num_dim,num_class);%(X'*X + alpha*eye(num_dim)) \ (X'*Y);%zeros(num_dim,num_label);
    %W_k   = (X'*X + rho*eye(num_dim)) \ (X'*Y);%zeros(num_dim,num_label)
    %W_k = randn(d1, num_class);
    %P_k = rand(num_dim, d1); % zeros or random, or ones or eye
    %P_k = P_k ./sqrt(sum(P_k.^2,1));
    P_k = ones(num_dim, d1);
    P_k = P_k ./sqrt(sum(P_k.^2,1));

    W_k = pinv(P_k' * P_k) * P_k' * pinv(X' * X) * X' * Y;
    %C_k = eye(num_class,num_class); 
    R_k = zeros(num_class,num_class); %eye(num_class,num_class);
       
    %Feature Similarity
    %S = exp(-squareform(pdist(X')));
    iter = 1; oldloss = 9999999;
    
    %Instance similarity and Laplacian
    %Try different similarity measures 
    SI = exp(-squareform(pdist(X)));
    Linst = diag(sum(SI, 2)) - SI;
      
    epsilon  = eps;
    
    %E = ones(size(Data.Ytrain));
    E = ones(size(Y));
    while iter <= maxIter
      
       delP = X' * (X * P_k * W_k - Y * R_k)*W_k' + lambda5 * X' * Linst * X * P_k * W_k * W_k';
       
       L = diag(sum(R_k)) - R_k;

       grad = delP / norm(delP, 'fro');
       alpha = computeStepSize('P', P_k, grad, etaP, X, Y, W_k, R_k, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6); 
       P_k = P_k - alpha * grad;
       P_k = P_k ./sqrt(sum(P_k.^2,1));

      
       
       
       delR = -Y'*(X * P_k * W_k - Y * R_k) + lambda1 * Y' * (Y*R_k - Y) ...
           + lambda2 * (R_k - W_k' * W_k) + lambda4 * R_k;
       
       %Value of R_k is increasing very rapidly. Look at it.
       L = diag(sum(R_k)) - R_k;

       grad = delR / norm(delR, 'fro');
       alpha = computeStepSize('R', R_k, grad, etaR, X, Y, W_k, R_k, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6); 
       R_k = R_k - alpha * grad;

        L = diag(sum(R_k)) - R_k;
       delW = P_k'*X'*(X * P_k * W_k - Y * R_k) + 2*lambda2 * (-W_k)*(R_k - W_k'*W_k)+lambda3 * W_k ...
           +lambda5*P_k'*X'*Linst*X*P_k*W_k + lambda6*W_k*(L+L');
       
       grad = delW / norm(delW, 'fro');
       alpha = computeStepSize('W', W_k, grad, etaW, X, Y, W_k, R_k, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6); 
       W_k = W_k - alpha * grad;
      
            
      %% Loss
       
       totalloss = ObjectiveValueLS( X, Y, W_k, R_k, P_k, Linst, L, E, ...
           lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
       loss(iter,1) = totalloss;
       iter=iter+1;
       if iter > 50
           iter;
       end
    end
    model.W = W_k;
    model.R = R_k;
    model.P = P_k;
    %model.loss = loss;
    plot(loss)
    model.optmParameter = optmParameter;
end

function loss = ObjectiveValueLS( X, Y, W_k, R_k, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6)
      %% Loss
      T1 = 0.5 * trace((X*P_k*W_k - Y*R_k)'* (X*P_k*W_k - Y*R_k));
      T2 = 0.5 * lambda1 * trace((Y*R_k - Y)'*(Y*R_k - Y));
      T3 = 0.5 * lambda2 * trace((R_k - W_k'*W_k)'*(R_k - W_k'*W_k));
      T4 = 0.5 * lambda3 * trace(W_k' * W_k);
      T5 = 0.5 * lambda4 * trace(R_k' * R_k);
      T6 = 0.5 * lambda5 * trace((X*P_k*W_k)'*Linst*(X*P_k*W_k));
      T7 = lambda6 * trace(W_k * L * W_k');
      loss = T1 + T2 + T3 + T4 + T5 + T6 + T7;
end

function [alpha] = computeStepSize(V, M, grad, alpha, X, Y, W_k, R_k, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6) 
       obj1 = ObjectiveValueLS(X, Y, W_k, R_k, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
       flag = 0; j = 1;
       while ( j > 0 )
           Mnew = M - alpha * grad;
           if V == 'W'
                obj2 = ObjectiveValueLS(X, Y, Mnew, R_k, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
           elseif V == 'P'
               obj2 = ObjectiveValueLS(X, Y, W_k, R_k, Mnew, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
           elseif V == 'R'
               obj2 = ObjectiveValueLS(X, Y, W_k, Mnew, P_k, Linst, L, E, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6);
           end
           if obj2 > obj1
               flag = 1;
               alpha = alpha * 0.5;
           else
               break;
           end
       end 
       %if flag
       %    alpha = alpha * 2;
       %end
end

