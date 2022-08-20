import cv2
import numpy as np
from scipy.io import loadmat
import matlab.engine

from c_get_preference_matrix_h import get_preference_matrix_h
from d_clustering import clustering
from c_get_preference_matrix_fm import get_preference_matrix_fm
from utils_t_linkage import plotMatches, show_pref_matrix, compute_errors, plot_clusters, get_cluster_mask

OUTLIER_THRESHOLD = 8

eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\allem\Desktop\multilink\fun',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\geoTools',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\model_spec',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\selection',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\utils',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink',nargout=0)


def t_linkage(tau, label_k, mode):

    # region Get image path from label_k
    img_i = "../resources/adel" + mode + "_imgs/" + label_k + "1.png"
    img_j = "../resources/adel" + mode + "_imgs/" + label_k + "2.png"
    # endregion

    # region Load data points
    data_dict = loadmat("../resources/adel" + mode + "/" + label_k + ".mat")
    points = data_dict['data']
    points = np.transpose(points)
    src_pts = points[:, 0:2]
    dst_pts = points[:, 3:5]
    num_of_points = src_pts.shape[0]
    # endregion

    # region Sort points so to graphically emphasize the structures in the preference matrix
    label = data_dict['label'].flatten()
    idx = np.argsort(label)

    src_pts = np.array([src_pts[i] for i in idx])
    dst_pts = np.array([dst_pts[i] for i in idx])
    clusters_mask_gt = np.array([label[i] for i in idx])

    # endregion

    # region Build kp and good_matches
    kp_src = [cv2.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1)
              for p in src_pts]
    kp_dst = [cv2.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1)
              for p in dst_pts]
    good_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=1) for i in range(len(src_pts))]
    plotMatches(img_i, img_j, kp_src, kp_dst, good_matches)
    # endregion

    # | ############################################################################################################# |

    # region Get preference matrix
    if mode == "FM":
        pref_m = get_preference_matrix_fm(kp_src, kp_dst, good_matches, tau)
        modelType = "fundamental" 
    else:  # mode == "H"
        pref_m = get_preference_matrix_h(kp_src, kp_dst, good_matches, tau)
        modelType = "homography" 

    show_pref_matrix(pref_m, label_k)
    # endregion

    # region Clustering
    #clusters, _ = clustering(pref_m)
    #print(clusters)
                                    # modelType = 'lc';  alternative  models line (l) and circle (c)
    gricParam = dict()
    gricParam["lambda1"] = 1                             # gricParam.lambda1 = 1;
    gricParam["lambda2"] = 2                             # gricParam.lambda2 = 2;
    gricParam["sigma"] = OUTLIER_THRESHOLD;  
                         # gricParam.sigma = epsi; 
     
    x_src = []
    y_src = []                       
    x_dst = []
    y_dst = []
    points = []

    for pt in src_pts:
        x_src.append(pt[0])
        y_src.append(pt[1])

    for pt in dst_pts:
        x_dst.append(pt[0])
        y_dst.append(pt[1])

    points = [x_src, y_src, x_dst, x_dst, y_dst, y_dst]

    points = matlab.double(points)
    prefM = matlab.double(pref_m.tolist())

    '''optsSampling = dict()

    optsSampling["m"] = 2000                             # optsSampling.m = 2000; % number of hypotheses
    optsSampling["sampling"] = "localized"               # optsSampling.sampling = 'localized';
    optsSampling["robust"] = "on"                        # optsSampling.robust = 'on';
    optsSampling["voting"] = "gauss"                     # optsSampling.voting = 'gauss';

    # sampling lines
    optsl = optsSampling                                 # optsl = optsSampling;
    optsl["model"] = modelType                           # optsl.model = 'line';
    print(points)
    Sl = eng.computeResi(points, optsl);                # Sl = computeResi(X,optsl);

    # preference computation
    epsi = 2e-2; # inlier threhsold                      # epsi = 2e-2; % inlier threhsold                       
    Sl["P"] = eng.resiToP(Sl["R"], epsi);                # [Sl.P] = resiToP(Sl.R,epsi);
    P = np.concatenate([np.array(Sl["P"])], axis=1)
    P = matlab.double(P.tolist())

    print(len(points))
    print("-------")
    print(len(points[0]))
    print("-------")
    print(len(P))
    print(len(P[0]))
    print("-------")
    print(modelType)
    print(gricParam)
    '''
    clusters_str = eng.multiLink(points,prefM,modelType,gricParam)           # C = multiLink(X,P,modelType,gricParam);
    clusters_dict = dict()
    clusters = []

    print(clusters_str)

    for i,p in enumerate(clusters_str):
        el = int(p[0])
        if el in clusters_dict:
            clusters_dict[el].append(i)
        else:
            clusters_dict[el] = [i]

    for key in clusters_dict:
        clusters.append(clusters_dict[key])
    clusters_mask = get_cluster_mask(clusters, num_of_points, OUTLIER_THRESHOLD)
    
    # endregion

    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask_gt, label_k + " - Ground-truth")
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Estimation")
    # endregion

    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("ME % = " + str(round(float(me), 4)))
    # endregion


