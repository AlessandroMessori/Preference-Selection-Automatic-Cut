
import scipy.io
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\allem\Desktop\multilink\fun',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\geoTools',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\model_spec',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\selection',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\utils',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink',nargout=0)

temp = scipy.io.loadmat('../multilink/data/ticktoe.mat')  # temp = load('./data/ticktoe.mat');
#print(temp)
X = temp["X"].tolist()
X = matlab.double(X)                                       # X = temp.X;
G = temp["G"]                                              # G = temp.G;

#print(X)

'''figure;
gscatter(X(1,:),X(2,:),G);
legend off, axis off, axis equal;
title('Input Data');'''

# preference embedding with multiple model class

# mixed hypotheses sampling: lines and circles
optsSampling = dict()

optsSampling["m"] = 2000                             # optsSampling.m = 2000; % number of hypotheses
optsSampling["sampling"] = "localized"               # optsSampling.sampling = 'localized';
optsSampling["robust"] = "on"                        # optsSampling.robust = 'on';
optsSampling["voting"] = "gauss"                     # optsSampling.voting = 'gauss';


# sampling lines
optsl = optsSampling                                 # optsl = optsSampling;
optsl["model"] = "line"                              # optsl.model = 'line';
Sl = eng.computeResi(X, optsl);                      # Sl = computeResi(X,optsl);

# sampling circles
optsc = optsSampling                                 # optsc = optsSampling ;
optsc["model"] = "circle"                            # optsc.model = 'circle';
Sc = eng.computeResi(X,optsc)                        # Sc = eng.computeResi(X,optsc);

# preference computation
epsi = 2e-2; # inlier threhsold                      # epsi = 2e-2; % inlier threhsold                       
Sl["P"] = eng.resiToP(Sl["R"], epsi);                # [Sl.P] = resiToP(Sl.R,epsi);
Sc["P"] = eng.resiToP(Sc["R"], epsi);                # [Sc.P] = resiToP(Sc.R,epsi);
#P = matlab.double([Sl["P"], Sc["P"] ]);             # P =[Sl.P,Sc.P];
P = np.concatenate([np.array(Sl["P"]), np.array(Sc["P"])], axis=1)
P = matlab.double(P.tolist())

# agglomerative clustering with model selection

modelType = "lc"                                     # modelType = 'lc';  alternative  models line (l) and circle (c)
gricParam = dict()
gricParam["lambda1"] = 1                             # gricParam.lambda1 = 1;
gricParam["lambda2"] = 2                             # gricParam.lambda2 = 2;
gricParam["sigma"] = epsi;                           # gricParam.sigma = epsi;   

C = eng.multiLink(X,P,modelType,gricParam)           # C = multiLink(X,P,modelType,gricParam);
thCard = 10                                          # thCard = 10; prune small clusters, e.g. using the cardinality of the mss

Cpruned = eng.prune_small_clust(C,thCard);           # Cpruned = prune_small_clust(C,thCard);

print(Cpruned)

'''
%% estimated clusters
figure; 
gscatter(X(1,:),X(2,:),Cpruned);
legend off, axis off, axis equal;
title('Estimated Clusters');
'''