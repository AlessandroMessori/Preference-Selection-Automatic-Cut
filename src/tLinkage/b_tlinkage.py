import cv2
import numpy as np
from treeCut.pacBayesianDendogramCut import DendrogramCut
from scipy.io import loadmat
import pandas as pd
import matlab.engine

from treeCut.gmart import bench_methods
from tLinkage.c_get_preference_matrix_h import get_preference_matrix_h
from tLinkage.d_clustering import clustering
from tLinkage.c_get_preference_matrix_fm import get_preference_matrix_fm
from utils.utils_t_linkage import plotMatches, show_pref_matrix, compute_errors, plot_clusters, get_cluster_mask


eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\allem\Desktop\multilink\fun',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\geoTools',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\model_spec',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\selection',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\utils',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink',nargout=0)

OUTLIER_THRESHOLD = 8
OUTLIER_THRESHOLD_GMART = 37

DISPLAY = False

def load_points(path):
    data_dict = loadmat(path)
    points = data_dict['data']
    points = np.transpose(points)
    src_pts = points[:, 0:2]
    dst_pts = points[:, 3:5]
    num_of_points = src_pts.shape[0]

    # Sort points so to graphically emphasize the structures in the preference matrix
    label = data_dict['label'].flatten()
    #return indices that would sort the array
    idx = np.argsort(label)

    src_pts = np.array([src_pts[i] for i in idx])
    dst_pts = np.array([dst_pts[i] for i in idx])
    clusters_mask_gt = np.array([label[i] for i in idx])

    return data_dict, src_pts, dst_pts, num_of_points, clusters_mask_gt

def get_preference_matrix(mode, kp_src, kp_dst, good_matches, tau):
    if mode == "FM":
        while True:
            try:
                pref_m = get_preference_matrix_fm(kp_src, kp_dst, good_matches, tau)
            except:
                print('there was an error computing the svd, trying again…')
                continue
            break
        modelType = "fundamental" 
    else:  # mode == "H"
        while True:
            try:
                pref_m = get_preference_matrix_h(kp_src, kp_dst, good_matches, tau)
            except:
                print('there was an error computing the svd, trying again…')
                continue
            break   
        modelType = "homography" 

    return pref_m, modelType

def evaluation(tau, label_k, mode, OUTLIER_THRESHOLD_GMART,NUMBER_OF_CLUSTER):

    errors_list = []
    # Get image path from label_k
    img_i = "../resources/adel" + mode + "_imgs/" + label_k + "1.png"
    img_j = "../resources/adel" + mode + "_imgs/" + label_k + "2.png"
    # Load data points
    data_dict, src_pts, dst_pts, num_of_points, clusters_mask_gt = load_points("../resources/adel" + mode + "/" + label_k + ".mat")

    # Build kp and good_matches
    kp_src = [cv2.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1)
              for p in src_pts]
    kp_dst = [cv2.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1)
              for p in dst_pts]
    good_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=1) for i in range(len(src_pts))]
    
    DISPLAY and plotMatches(img_i, img_j, kp_src, kp_dst, good_matches)

    # Get preference matrix
    pref_m, modelType = get_preference_matrix(mode, kp_src, kp_dst, good_matches, tau)

    DISPLAY and show_pref_matrix(pref_m, label_k)
    DISPLAY and show_pref_matrix(pref_m, label_k)
    
    # Clustering

    #clusters_dyn, clusters_cut = bench_methods(dst_pts, 8, methods)
    #DISPLAY and show_pref_matrix(pref_m, label_k)
    
    #DISPLAY and plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask_gt, label_k + " - Ground-truth")

    #clusters = []

    '''for i in range(len(clusters_dyn)):
        # Clustering
        #clusters, pref_m = clustering(pref_m)
        clusters_mask = get_cluster_mask(clusters_dyn[i], num_of_points, OUTLIER_THRESHOLD)
    
        # Plot clusters
        # DISPLAY and plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Estimation")
        
        # Compute Misclassification Error
        err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
        me = err / num_of_pts  # compute misclassification error
        print(str(methods[i] + "  dyn"))
        print("ME % = " + str(round(float(me), 4)),'\n')
        
        # Clustering
        #clusters, pref_m = clustering(pref_m)
        clusters_mask = get_cluster_mask(clusters_cut[i], num_of_points, OUTLIER_THRESHOLD)
        
        # Plot clusters
        # DISPLAY and plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Estimation")
        
        # Compute Misclassification Error
        err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
        me = err / num_of_pts  # compute misclassification error
        print(str(methods[i] + "  CUT"))
        print("ME % = " + str(round(float(me), 4)),'\n')
        
    '''
    clusters, _, _ = clustering(pref_m)
  
    clusters_mask = get_cluster_mask(clusters, num_of_points, OUTLIER_THRESHOLD)
    

    # Plot clusters
    #plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - T-Linkage Estimation")
    
    # Compute Misclassification Error
    #err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    #me = err / num_of_pts  # compute misclassification error
    #print('T_Linkage')
    #print("ME % = " + str(round(float(me), 4)), '\n')
    #errors_list.append(round(float(me), 4))
    

    #print(len(dst_pts))
    #clusters, _, linkage_m = clustering(pref_m, dCut = True)

    #clusters_dyn, clusters_cut = bench_methods(dst_pts, NUMBER_OF_CLUSTER, ['tlinkage'], linkage_m)

    #print(clusters)

    #clusters_mask = get_cluster_mask(clusters_dyn[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    
    # Plot clusters
    #DISPLAY and plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - TLinkage + Gmart Dyn Estimation")
    
    # Compute Misclassification Error
    #err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    #me = err / num_of_pts  # compute misclassification error
    #print("T_Linkage with GMART DYN")
    #print("ME % = " + str(round(float(me), 4)),'\n')
    #errors_list.append(round(float(me), 4))


    #clusters_mask = get_cluster_mask(clusters_cut[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    
    # Plot clusters
    #DISPLAY and plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - TLinkage + Gmart Flat Estimation")
    
    # Compute Misclassification Error
    #err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    #me = err / num_of_pts  # compute misclassification error
    #print("T_Linkage with GMART COSTANT CUT")
    #print("ME % = " + str(round(float(me), 4)),'\n')
    #errors_list.append(round(float(me), 4))


    '''dist = pdist(dst_pts)
    dist = squareform(dist)
    df_linkage = pd.DataFrame(linkage_m)
    model = DendrogramCut(k_max=230, method='average').fit(dist, df_linkage)
    model.pac_bayesian_cut()

    clusters_str = dendrogram(model.linkage, no_plot=True)['color_list']
    clusters_dict = dict()
    clusters = []

    for i,p in enumerate(clusters_str):
        if p in clusters_dict:
            clusters_dict[p].append(i)
        else:
            clusters_dict[p] = [i]

    for key in clusters_dict:
        clusters.append(clusters_dict[key])

    #print(clusters)

    clusters_mask = get_cluster_mask(clusters, num_of_points, 42)
    
    # Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - TLinkage + Pac Bayesian Estimation")
    
    # Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("T_LINKAGE with Pac Bayesian")
    print("ME % = " + str(round(float(me), 4)),'\n')'''

    gricParam = dict()
    gricParam["lambda1"] = 1                            
    gricParam["lambda2"] = 2                           
    gricParam["sigma"] = 8;  

    points = matlab.double(data_dict['data'].tolist())
    prefM = matlab.double(pref_m.tolist())

    clusters_str = eng.multiLink(points,prefM,modelType,gricParam)  
    clusters_dict = dict()
    clusters = []

    for i,p in enumerate(clusters_str):
        p = str(p)
        if p in clusters_dict:
            clusters_dict[p].append(i)
        else:
            clusters_dict[p] = [i]

    for key in clusters_dict:
        clusters.append(clusters_dict[key])

    #print(clusters)

    clusters_mask = get_cluster_mask(clusters, num_of_points, OUTLIER_THRESHOLD)
    
    # Plot clusters
    #plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink Estimation")
    
    # Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    #print("Multilink")
    #print("ME % = " + str(round(float(me), 4)),'\n')
    errors_list.append(round(float(me), 4))


    dendro = eng.multiLinkMock(points,prefM,modelType,gricParam)         
    dendro_clean = []
    nums = {i: 1 for i in range(num_of_points)}
    
    for i,row in enumerate(dendro):
        if row[0] != 0 or row[1] != 0:
            index_i = int(row[0]-1)
            index_j = int(row[1]-1)
            nums[num_of_points+i] = nums[index_i] + nums[index_j]
            dendro_clean.append([index_i, index_j, row[2]+0.00001*i, nums[num_of_points+i]])

    #print(linkage_m)
    #print("---------")
    #print(dendro_clean)


    clusters_dyn, clusters_cut = bench_methods(dst_pts, NUMBER_OF_CLUSTER , ['multilink'], dendro_clean)

    #print(clusters)

    clusters_mask = get_cluster_mask(clusters_dyn[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    
    # Plot clusters
    #plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink + Gmart Dyn Estimation")
    
    # Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    #print("Multilink with GMART DYN")
    #print("ME % = " + str(round(float(me), 4)),'\n')
    errors_list.append(round(float(me), 4))

    clusters_mask = get_cluster_mask(clusters_cut[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    
    # Plot clusters
    #plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink + Gmart Cut Estimation")
    
    # Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    #print("Multilink with GMART CUT")
    #print("ME % = " + str(round(float(me), 4)),'\n')
    errors_list.append(round(float(me), 4))
    
    '''dist = pdist(dst_pts)
    dist = squareform(dist)
    df_linkage = pd.DataFrame(dendro_clean)
    model = DendrogramCut(k_max=500, method='average').fit(dist, df_linkage)
    model.pac_bayesian_cut()

    clusters_str = dendrogram(model.linkage, no_plot=True)['color_list']
    clusters_dict = dict()
    clusters = []

    for i,p in enumerate(clusters_str):
        if p in clusters_dict:
            clusters_dict[p].append(i)
        else:
            clusters_dict[p] = [i]

    for key in clusters_dict:
        clusters.append(clusters_dict[key])

    #print(clusters)

    clusters_mask = get_cluster_mask(clusters, num_of_points, 20)
    
    # Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink + Pac Bayesian Estimation")
    
    # Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("Multilink with Pac Bayesian")
    print("ME % = " + str(round(float(me), 4)),'\n')'''
    return errors_list


 