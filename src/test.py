
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# temp = load('./data/ticktoe.mat');

temp = scipy.io.loadmat('../multilink/data/ticktoe.mat')
#print(temp)
X = temp["X"].tolist()

clusters = ['C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C0', 'C0', 'C0', 'C0', 'C1', 'C0', 'C0', 'C0', 'C0', 'C2', 'C1', 'C1', 'C0', 'C0', 'C1', 'C1', 'C0', 'C0', 'C0', 'C0', 'C1', 'C2', 'C0', 'C0', 'C2', 'C0', 'C0', 'C1', 'C1', 'C0', 'C1', 'C0', 'C1', 'C1', 'C0', 'C1', 'C1', 'C0', 'C0', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C0', 'C0', 'C1', 'C0']
                                                     # figure; 
plt.scatter(X[0], X[1], c=clusters ,alpha=0.5)        # gscatter(X(1,:),X(2,:),Cpruned);
plt.show()                                                    
                                                     # legend off, axis off, axis equal;
                                                     # title('Estimated Clusters');

print(len(X))
