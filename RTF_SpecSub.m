function rtf = RTF_SpecSub(x,fs,nfft,ninc)

%% Input
% x : audio signal, with the size of No. of samples x No. of mics
% fs: sampling rate
% nfft: window length of STFT
% ninc: step size of STFT
%
%% Output
% rtf: No. of frequency x No. of mics, the first column is always 1, since the first channel is taken as the reference  

% Author: Xiaofei Li, INRIA Grenoble Rhone-Alpes
% Copyright: Perception Team, INRIA Grenoble Rhone-Alpes
% The algorithm is described in the paper:  
% X. Li, L. Girin, R. Horaud and S. Gannot. Estimation of Relative Transfer Function in the Presence of Stationary Noise 
% Based on Segmental Power Spectral Density Matrix Subtraction. ICASSP, 2015. 


if nargin==3
    ninc = 0.5*nfft;
end 
if nargin<3
    nfft = 512;  
    ninc = 0.5*nfft;
end 

M = size(x,2);            % Microphone number
W = round(0.3*fs/ninc);   % Segment length, corresponds to 0.3 s
R = round(W/2);           % Segment increment, this can be changed to any value in [1 W]

% STFT
X = stft(x(:,1),nfft,ninc,hamming(nfft));
for m = 2:M
    X(:,:,m) = stft(x(:,m),nfft,ninc,hamming(nfft));
end
[Omega,L,~] = size(X);

[r1,r2] = MCMT(W,R,L);    % Two thresholds: minimum controled maximum thresholds

% Segmental PSD matrix computation, for all the frequency and segments
Ls = length(W:R:L);       % Number of segments                    
PSD = zeros(Omega,Ls,M,M);
for w = 0:W-1
    Xw = X(:,W-w:R:end,:);    
    Xw1 = reshape(Xw(:,1:Ls,:),[Omega,Ls,M,1]);
    Xw2 = reshape(Xw(:,1:Ls,:),[Omega,Ls,1,M]);
    psdw = bsxfun(@times,Xw1,conj(Xw2));
    PSD = PSD+psdw;
end

% Segmental classification
psd = PSD(:,:,1,1);
minpsd = min(psd,[],2);
l1 = psd>r1*repmat(minpsd,[1,Ls]);    % speech class
l2 = psd<r2*repmat(minpsd,[1,Ls]);    % noise class

% Frequency-wise rtf computation 
rtf = zeros(Omega,M);
for omega = 1:Omega
    
    % Indices of segments
    ol1 = find(l1(omega,:));    
    if isempty(ol1)
        continue;
    end    
    ol2 = find(l2(omega,:));   
    
    % Nearest noise segments 
    dis = abs(bsxfun(@minus,ol1,ol2'));
    [~,ol12_ind] = min(dis,[],1);
    ol12 = ol2(ol12_ind);
    
    % Spectral subtraction
    psdss = squeeze(sum(PSD(omega,ol1,:,:) - PSD(omega,ol12,:,:),2));
    % Eigendecomposition
    [pv,~] = eigs(psdss,1);
    % Relative to the first (reference) channel
    rtf(omega,:) = pv.'/pv(1);  
end


                    




 
