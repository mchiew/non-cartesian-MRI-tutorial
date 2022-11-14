%% Non-Cartesian MR Image Reconstruction Without Dependencies
% Mark Chiew (mark.chiew@ndcn.ox.ac.uk)
% http://users.fmrib.ox.ac.uk/~mchiew/docs/recon_tutorial_01.m

% NB: Adding local functions to scripts is only supported from R2016b
% onwards. If using an older version of MATLAB, move the function
% definitions to separate files, or wrap the main script inside of its own
% function.
%
%% Define our test object
N_x = 64;           % size of image
x   = phantom(N_x); % initialise phantom image

% show our source image
figure();imshow(x,[0 1]); title('Source Image');drawnow;


%% Define arbitrary k-space sampling locations
% Perturbed variable density spiral
N_k = 64^2;                         % number of k-samples
t   = linspace(0,sqrt(0.5),N_k)';   % dummy variable to parameterise spiral

k_x = (1 + randn(N_k,1)/20).*t.^2.*cos(2*pi*32*t);  % spiral kx-coords
k_y = (1 + randn(N_k,1)/20).*t.^2.*sin(2*pi*32*t);  % spiral ky-coords

% show scatterplot of trajectory
figure();scatter(k_x,k_y,5,'filled');axis square;grid on 
xlabel('k_x');ylabel('k_y');title('k-space Sampling Locations');drawnow


%% Define discrete Fourier encoding matrix
[xidx yidx] = meshgrid(-N_x/2 : N_x/2 - 1);    % x- and y-coords with "isocentre" at the N_x/2+1 index

% Loop over each k-location to construct its required spatial phase modulation
F = zeros(N_k, N_x, N_x);
for i = 1:N_k
    F(i,:,:) = exp(1j*2*pi*(k_x(i)*xidx+k_y(i)*yidx));   % 2D Fourier exponential
end

F = reshape(F, N_k, []);    % reshape F so that each row is a single k-space encoding

% show an example encoding
figure();imshow(angle(reshape(F(2000,:),N_x,N_x)),[-pi pi],'colormap',hsv);
title('Phase of Example 2D Fourier Encoding Modulation');colorbar();drawnow;


%% Perform forward encoding/measurement
d = F*x(:); % multiply encoding matrix with image vector to get data


%% Direct Inverse Reconstruction?
fprintf(1,'Condition number: %3.2G\n',cond(F)); % condition number of F
est0 = F\d;    % compute a naive inverse (warning: slow)

% print nrmse to the ground truth and show image and difference image
nrmse(est0, x, 'Direct Inverse'),plt(est0, x, N_x, 'Direct Inverse Est.');


%% Direct Regularised Least Squares
lambda  = 1E-4;                        % regularisation weighting
E       = (F'*F+lambda*eye(N_x^2))\F'; % compute linear estimator (warning: slow)

fprintf(1,'Condition number: %3.2G\n',cond(E)); % condition number of E
est1    = E*d;   % regularised reconstruction

% print nrmse to the ground truth and show image and difference image
nrmse(est1, x, 'Regularised Inverse'),plt(est1, x, N_x, 'Regularised Inverse');


%% Iterative Regularised Least Squares
% simplifying substitutions
A = F'*F + lambda*eye(N_x^2); 
b = F'*d;

% define the gradient as a function, grad(x) = gradient(C(x))
grad = @(x) A*x - b;

% define a fixed step size, max number of iterations, and relative change tolerance
step = 1E-6;
iter = 1000;
tol  = 1E-6;

% define an initial guess (starting point)
est2 = zeros(N_x^2,1);

% gradient descent function defined at the bottom
% ===================================================================
%{ 
function x = grad_desc(x, grad, step, max_iter, tol)
% steepest gradient descent
    ii = 0;                          % iteration counter
    dx = inf;                        % relative change measure
    while ii < max_iter && dx > tol  % check loop exit conditions
        tmp = step*grad(x);          % compute gradient step
        dx  = norm(tmp)/norm(x);     % compute relative change metric
        x   = x - tmp;               % update estimate
        ii  = ii+1;                  % update iteration count
    end
end
%}
% ===================================================================

% run through steepest gradient descent 
est2 = grad_desc(est2, grad, step, iter, tol);

% print nrmse to the ground truth and show image and difference image
nrmse(est2, x, 'Iterative L2 Reg'),plt(est2, x, N_x, 'Iterative L2 Reg');


%% Using built-in linear solvers for L2 penalties
% using pcg
est3 = pcg(A,b,tol,iter);
% print nrmse to the ground truth and show image and difference image
nrmse(est3, x, 'pcg'),plt(est3, x, N_x, 'pcg estimate');

% using minres
est4 = minres(A,b,tol,iter);
% print nrmse to the ground truth and show image and difference image
nrmse(est4, x, 'minres'),plt(est4, x, N_x, 'minres estimate');


%% Compressed Sensing (L1 regularised) example using TV
% huber and huber gradient functions defined at bottom
% ======================================================
%{ 
function y = huber(x,a)
% Differentiable Huber loss to replace L1
    y     = zeros(size(x));
    ii    = abs(x) < a;          % find values of x < a
    y(ii) = abs(x(ii)).^2/(2*a); % smooth quadratic part
    ii    = abs(x) >= a;         % find values of x >= a
    y(ii) = abs(x(ii)) - a/2;    % abs(x) part
end
function y = huber_grad(x,a)
% Gradient of differentiable Huber loss
    y     = zeros(size(x));
    ii    = abs(x) < a;        % find values of x < a
    y(ii) = x(ii)/a;           % smooth quadratic part
    ii    = abs(x) >= a;       % find values of x >= a
    y(ii) = x(ii)./abs(x(ii)); % abs(x) part
end
%}
% ======================================================

xx = linspace(-5E-6,5E-6,1000);
clf;

% plot Huber loss
subplot(1,2,1);hold on;

plot(xx, abs(xx),        'linewidth',3);
plot(xx, huber(xx,1E-6), 'linewidth',3);             

title('Smoothed loss');
xlim([-5E-6,5E-6]); ylim([0,5E-6]);
xlabel('x'); ylabel('|x|');
set(gca,'FontSize',12); legend({'Abs','Huber Approx'});
grid on;axis square;drawnow;

% plot gradient of Huber loss
subplot(1,2,2);hold on;

plot(xx(1:end-1), diff(abs(xx))/abs(xx(2)-xx(1)), 'linewidth',3);
plot(xx,          huber_grad(xx,1E-6),            'linewidth',3);        

title('Gradients');
xlim([-5E-6,5E-6]); ylim([-3 3]);
xlabel('x'); ylabel('\nabla|x|');
set(gca,'FontSize',12); legend({'Abs','Huber Approx'});
grid on;axis square;drawnow;

a       = 1E-16;              % Huber-smoothing constant
lambda  = 5E1;                % sparsity regularisation factor
FF      = F'*F;               % pre-compute F'F
b       = F'*d;               % pre-compute F'd

% Finite difference and adjoint functions defined at bottom
% ==========================================================
%{
function b = D_fwd(a,N_x)
% Forward 1st order finite difference operator
% This is like a gradient operator: a -> (Dx(a), Dy(a))
    a  = reshape(a,N_x,N_x);
    Dx = reshape(a - circshift(a,-1,1),[],1);
    Dy = reshape(a - circshift(a,-1,2),[],1);  
    b  = [Dx, Dy];
end
function a = D_adj(b,N_x)
% Adjoint 1st order finite difference operator
% This is like a divergence operator: b -> Dx'(bx) + Dy'(by)
    b  = reshape(b,N_x,N_x,2);
    dx = reshape(b(:,:,1) - circshift(b(:,:,1),1,1),[],1);
    dy = reshape(b(:,:,2) - circshift(b(:,:,2),1,2),[],1);
    a  = dx + dy;
end
%}
% ==========================================================

L       = @(x) D_fwd(x, N_x); % define forward sparsifying transform (see Helper functions)
Ladj    = @(x) D_adj(x, N_x); % define adjoint sparsifying transform (see Helper functions)

% define gradient of CS cost
grad    = @(x) FF*x - b + lambda*Ladj(huber_grad(L(x),a)); 

est_TV  = zeros(N_x^2,1); % initial guess
step    = 1E-6;           % step size    
tol     = 1E-6;           % minimum change tolerance

% run through steepest gradient descent 
est_TV = grad_desc(est_TV, grad, step, iter, tol);  % see Helper functions below

% print nrmse to the ground truth and show image and difference image
nrmse(est_TV, x, 'CS-TV 1'),plt(est_TV, x, N_x, ['CS-TV estimate, ' num2str(iter) ' iters']);

% run for more iterations
est_TV = grad_desc(est_TV, grad, step, iter, tol);

% print nrmse to the ground truth and show image and difference image
nrmse(est_TV, x, 'CS-TV 2'),plt(est_TV, x, N_x, ['CS-TV estimate, ' num2str(2*iter) ' iters']);
% run for even more iterations
est_TV = grad_desc(est_TV, grad, step, iter, tol);

% print nrmse to the ground truth and show image and difference image
nrmse(est_TV, x, 'CS-TV 3'),plt(est_TV, x, N_x, ['CS-TV estimate, ' num2str(3*iter) ' iters']);


%% Helper Functions
function x = grad_desc(x, grad, step, max_iter, tol)
% steepest gradient descent
    ii = 0;                          % iteration counter
    dx = inf;                        % relative change measure
    while ii < max_iter && dx > tol  % check loop exit conditions
        tmp = step*grad(x);          % compute gradient step
        dx  = norm(tmp)/norm(x);     % compute relative change metric
        x   = x - tmp;               % update estimate
        ii  = ii+1;                  % update iteration count
    end
end

function y = huber(x,a)
% Differentiable Huber loss to replace L1
    y     = zeros(size(x));
    ii    = abs(x) < a;          % find values of x < a
    y(ii) = abs(x(ii)).^2/(2*a); % smooth quadratic part
    ii    = abs(x) >= a;         % find values of x >= a
    y(ii) = abs(x(ii)) - a/2;    % abs(x) part
end
function y = huber_grad(x,a)
% Gradient of differentiable Huber loss
    y     = zeros(size(x));
    ii    = abs(x) < a;        % find values of x < a
    y(ii) = x(ii)/a;           % smooth quadratic part
    ii    = abs(x) >= a;       % find values of x >= a
    y(ii) = x(ii)./abs(x(ii)); % abs(x) part
end

function b = D_fwd(a,N_x)
% Forward 1st order finite difference operator
% This is like a gradient operator: a -> (Dx(a), Dy(a))
    a  = reshape(a,N_x,N_x);
    Dx = reshape(a - circshift(a,-1,1),[],1);
    Dy = reshape(a - circshift(a,-1,2),[],1);  
    b  = [Dx, Dy];
end
function a = D_adj(b,N_x)
% Adjoint 1st order finite difference operator
% This is like a divergence operator: b -> Dx'(bx) + Dy'(by)
    b  = reshape(b,N_x,N_x,2);
    dx = reshape(b(:,:,1) - circshift(b(:,:,1),1,1),[],1);
    dy = reshape(b(:,:,2) - circshift(b(:,:,2),1,2),[],1);
    a  = dx + dy;
end

function nrmse(x,y,s)
% normalised root-mean-square error
    fprintf(1,'%s NRMSE: %f\n',s,norm(x(:)-y(:))/norm(y(:)));
end
function plt(x,y,N,T)
% plots image estimate alongside a 5x magnified difference image
    figure();
    subplot(1,2,1);imshow(reshape(abs(x),N,N),[0 1]);title(T);
    subplot(1,2,2);imshow(reshape(abs(x),N,N)-y,[-0.2 0.2]);title('Difference x5');
    drawnow;
end
