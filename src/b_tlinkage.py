import cv2
import numpy as np
from pacBayesianDendogramCut import DendrogramCut
from scipy.io import loadmat
import pandas as pd
import matlab.engine

from c_get_preference_matrix_h import get_preference_matrix_h
from d_clustering import clustering
from c_get_preference_matrix_fm import get_preference_matrix_fm
from utils_t_linkage import plotMatches, show_pref_matrix, compute_errors, plot_clusters, get_cluster_mask

eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\allem\Desktop\multilink\fun',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\geoTools',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\model_spec',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\selection',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink\utils',nargout=0)
eng.addpath(r'C:\Users\allem\Desktop\multilink',nargout=0)

OUTLIER_THRESHOLD = 8
OUTLIER_THRESHOLD_GMART = 37
def compute_dyncut(data, nbClusters, children_map):
    nbVertices = max(children_map)
    inf = float("inf")
    dp = np.zeros((nbClusters + 1, nbVertices + 1)) + inf
    lcut = np.zeros((nbClusters + 1, nbVertices + 1))


    for i in range(0, dp.shape[1]):
        #dato un cluster esempio 156,
        #in questa riga alla colonna 156 ci sta l'intravarianza
        #degli elementi appartenenti al cluster 156 [40,0,17]
        dp[1, i] = compute_intra_variance(i, children_map, data)

    root = max(children_map)
                                 #(150,299)
    for vertex in range(len(data), root + 1):
        left_child, right_child = children_map[vertex]
        for k in range(2, nbClusters + 1):
            vmin = inf
            kl_min = -1
            for kl in range(1, k):
                #k = 3, kl = 2
                v = dp[kl, left_child] + dp[k - kl, right_child]
                if v < vmin:
                    vmin = v
                    #kl_min = 1
                    kl_min = kl

            dp[k, vertex] = vmin
            lcut[k, vertex] = kl_min

    return dp, lcut


def build_dict_tree(linkage_matrix):
    tree = {}
    n = linkage_matrix.shape[0] + 1
    for i in range(0, n - 1):
        tree[linkage_matrix[i, 0]] = n + i
        tree[linkage_matrix[i, 1]] = n + i
    return tree


def build_children_map(tree):
    children_map = {}
    for k, v in tree.items():
        children_map[v] = children_map.get(v, [])
        children_map[v].append(int(k))
    return children_map

                    #0
def build_children(vertex, children_map):
    children = []
    if vertex in children_map:
        left_child, right_child = children_map[vertex]
        if left_child in children_map:
            #aggiungo elementi alla fine della lista
            children.extend(build_children(left_child, children_map))
        else:
            children.extend([left_child])

        if right_child in children_map:
            children.extend(build_children(right_child, children_map))
        else:
            children.extend([right_child])

    return children


def get_var(data, subpop):
    intravar = 0
    center = np.mean(data[subpop], axis=0)
    for elem in subpop:
        x = data[elem] - center
        intravar += np.dot(x, x)
    return intravar

                            #0
def compute_intra_variance(vertex, children_map, data):
    children = build_children(vertex, children_map)
    intravar = 0
    if children:
        intravar = get_var(data, children)

    return intravar


def compute_centers(data, target):
    centers = []
    for i in set(target):
        id_pts = [index for index, value in enumerate(target) if value == i]
        centers.append(np.mean(data[id_pts], axis=0))

    return centers


def compute_flat_dyn_clusters(cur_vertex, k, lcut, children_map):
    clusters = []
    # leaf
    if k == 1 and not cur_vertex in children_map:
        clusters.append([cur_vertex])
    # one cluster left, get the leaves
    if k == 1 and cur_vertex in children_map:
        leaves = build_children(cur_vertex, children_map)
        clusters.append(leaves)
    # recurse in left and right subtrees
    if k > 1:
        if cur_vertex in children_map:
            left_child, right_child = children_map[cur_vertex]
            clusters.extend(compute_flat_dyn_clusters(left_child, int(lcut[k, cur_vertex]), lcut, children_map))
            clusters.extend(compute_flat_dyn_clusters(right_child, int(k - lcut[k, cur_vertex]), lcut, children_map))

    return clusters


def compute_flat_cut_clusters(nbClusters, linkage_matrix):
    flat = fcluster(linkage_matrix, nbClusters, 'maxclust')
    flat_clusters = []
    for i in range(1, len(set(flat)) + 1):
        flat_clusters.append([index for index, value in enumerate(flat) if value == i])

    return flat_clusters

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target


def bench_methods(data, nbClusters, methods, t_linkage=None):
    # data una riga di una matrice calcola la distanza di quella riga con ogni altra
    # riga
    flat_cut_clusters_list = list()
    flat_dyn_clusters_list = list()
    d = pdist(data)
    for method in methods:
        if method in ['centroid', 'ward', 'median']:
            linkage_matrix = linkage(data, method)
        elif method == "tlinkage" or method == "multilink":
            linkage_matrix = np.array(t_linkage)
        else:
            linkage_matrix = linkage(d, method)

        tree = build_dict_tree(linkage_matrix)
        children_map = build_children_map(tree)
        dp, lcut = compute_dyncut(data, nbClusters, children_map)
        flat_dyn_clusters = compute_flat_dyn_clusters(max(children_map), nbClusters, lcut, children_map)
        flat_cut_clusters = compute_flat_cut_clusters(nbClusters, linkage_matrix)

        tot_dyn = 0
        tot_cut = 0
        for i in range(0, nbClusters):

            if i < len(flat_dyn_clusters):
                tot_dyn += get_var(data, flat_dyn_clusters[i])

            if i < len(flat_cut_clusters):
                tot_cut += get_var(data, flat_cut_clusters[i])

        #print("method:", method)
        #print("intra-variance:", "(DP)", tot_dyn, "\t(cst height)", tot_cut)
        #print("\n")
        flat_dyn_clusters_list.append(flat_dyn_clusters)
        flat_cut_clusters_list.append(flat_cut_clusters)
    return flat_dyn_clusters_list, flat_cut_clusters_list

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

nbClusters = 20
methods = ['single','complete','average','weighted','centroid','median','ward']

def center_proximity(alpha,pt,centers):
    dists = []
    for center in centers:
        dists.append(np.linalg.norm(pt-center))
    minDist = min(dists)
    argmin = np.argmin(dists)
    for i in range(0,len(dists)):
        if not i == argmin:
            if dists[i] <= alpha*minDist:
                return False
    return True

def prop_viol_center_proximity(alpha,data,target):
    nb_viol = 0
    centers = compute_centers(data,target)
    for pt in data:
        if not center_proximity(alpha,pt,centers):
            nb_viol += 1
    return nb_viol / len(target)

def verif_center_proximity(alpha,data,target):
    return prop_viol_center_proximity(alpha,data,target) == 0


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
    #return indices that would sort the array
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
    #plotMatches(img_i, img_j, kp_src, kp_dst, good_matches)
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

    #show_pref_matrix(pref_m, label_k)
    # endregion

    # region Clustering

    clusters_dyn, clusters_cut = bench_methods(dst_pts, 8, methods)
    #show_pref_matrix(pref_m, label_k)
    # endregion
    #plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask_gt, label_k + " - Ground-truth")

    #clusters = []

    for i in range(len(clusters_dyn)):
        # region Clustering
        #clusters, pref_m = clustering(pref_m)
        clusters_mask = get_cluster_mask(clusters_dyn[i], num_of_points, OUTLIER_THRESHOLD)
        # endregion

        # region Plot clusters
        #plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Estimation")
        # endregion
        # region Compute Misclassification Error
        err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
        me = err / num_of_pts  # compute misclassification error
        print(str(methods[i] + "  dyn"))
        print("ME % = " + str(round(float(me), 4)),'\n')
        # endregion
        # region Clustering
        #clusters, pref_m = clustering(pref_m)
        clusters_mask = get_cluster_mask(clusters_cut[i], num_of_points, OUTLIER_THRESHOLD)
        # endregion

        # region Plot clusters
        #plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Estimation")
        # endregion
        # region Compute Misclassification Error
        err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
        me = err / num_of_pts  # compute misclassification error
        print(str(methods[i] + "  CUT"))
        print("ME % = " + str(round(float(me), 4)),'\n')
        # endregion

    clusters, _, _ = clustering(pref_m)
  
    clusters_mask = get_cluster_mask(clusters, num_of_points, OUTLIER_THRESHOLD)
    # endregion

    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - T-Linkage Estimation")
    # endregion
    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print('T_Linkage')
    print("ME % = " + str(round(float(me), 4)), '\n')
    # endregion

    #print(len(dst_pts))
    clusters, _, linkage_m = clustering(pref_m, dCut = True)

    pd.DataFrame(linkage_m).to_csv("./linkage.csv")

    clusters_dyn, clusters_cut = bench_methods(dst_pts, 10, ['tlinkage'], linkage_m)

    #print(clusters)

    clusters_mask = get_cluster_mask(clusters_dyn[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    # endregion
    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - TLinkage + Gmart Dyn Estimation")
    # endregion
    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("T_Linkage with GMART DYN")
    print("ME % = " + str(round(float(me), 4)),'\n')

    clusters_mask = get_cluster_mask(clusters_cut[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    # endregion
    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - TLinkage + Gmart Flat Estimation")
    # endregion
    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("T_Linkage with GMART CUT")
    print("ME % = " + str(round(float(me), 4)),'\n')

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
    # endregion
    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - TLinkage + Pac Bayesian Estimation")
    # endregion
    # region Compute Misclassification Error
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
    # endregion
    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink Estimation")
    # endregion
    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("Multilink")
    print("ME % = " + str(round(float(me), 4)),'\n')


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


    clusters_dyn, clusters_cut = bench_methods(dst_pts, 10, ['multilink'], dendro_clean)

    #print(clusters)

    clusters_mask = get_cluster_mask(clusters_dyn[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    # endregion
    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink + Gmart Dyn Estimation")
    # endregion
    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("Multilink with GMART DYN")
    print("ME % = " + str(round(float(me), 4)),'\n')

    clusters_mask = get_cluster_mask(clusters_cut[0], num_of_points, OUTLIER_THRESHOLD_GMART)
    # endregion
    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink + Gmart Cut Estimation")
    # endregion
    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("Multilink with GMART CUT")
    print("ME % = " + str(round(float(me), 4)),'\n')
    
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
    # endregion
    # region Plot clusters
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Multilink + Pac Bayesian Estimation")
    # endregion
    # region Compute Misclassification Error
    err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    me = err / num_of_pts  # compute misclassification error
    print("Multilink with Pac Bayesian")
    print("ME % = " + str(round(float(me), 4)),'\n')'''
    


