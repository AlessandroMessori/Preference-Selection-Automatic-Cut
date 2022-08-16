import cv2
import numpy as np

from utils_sampling import localized_sampling

K = 6  # the overall number of sampling is computed as K * num_of_points
#MSS minimum sample set del datapoints necessario per stimare un modello
#minimo 8 punti per essere un cluster
FM_MSS = 8   # cardinality of the MSS for the fundamental matrix case
INLIER_THRESHOLD = 3.0  # inlier threshold, needed by OpenCV RanSaC


def get_preference_matrix_fm(kp_src, kp_dst, good_matches, tau):

    # region Initialize variables
    num_of_matches = len(good_matches)
    num_of_samplings = int(K * num_of_matches)
    preference_matrix = []
    src_pts = np.float32([kp_src[m.trainIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_dst[m.queryIdx].pt for m in good_matches])
    # endregion

    for m in range(num_of_samplings):
        # region Sample a MSS

        # region [OPTION 1] Uniform sampling (commented)
        # g = np.random.Generator(np.random.PCG64())
        # mss_idx = np.array(g.choice(num_of_matches, FM_MSS, replace=False))
        # endregion

        # region [OPTION 2] Localized sampling
        mss_idx = localized_sampling(src_pts, dst_pts, FM_MSS)
        # endregion

        # region Retrieve src and dst pts from the MSS indices
        good_matches_current = [good_matches[i] for i in mss_idx]
        src_pts_current = np.float32([kp_src[m.trainIdx].pt for m in good_matches_current]).reshape(-1, 1, 2)
        dst_pts_current = np.float32([kp_dst[m.queryIdx].pt for m in good_matches_current]).reshape(-1, 1, 2)
        # endregion

        # endregion

        # region Fit model
        #trovo matrice fondamentale per trovare epipolar lines e points che descrivono la trasformazione
        F, inliers_mask = cv2.findFundamentalMat(src_pts_current, dst_pts_current, cv2.FM_7POINT,
                                                 ransacReprojThreshold=INLIER_THRESHOLD, confidence=0.999)
        # endregion

        if F is not None:
            # region Fill column

            # region Compute residuals
            #r contiene la sampson distance tra ogni punto src e dst
            r = []
            for src_p, dst_p in zip(src_pts, dst_pts):
                src_p = np.array([src_p[0], src_p[1], 1])  # move to homogeneous coordinates
                dst_p = np.array([dst_p[0], dst_p[1], 1])
                x = cv2.sampsonDistance(src_p, dst_p, F)
                r.append(x)

            r = np.array(r)
            # endregion

            # region Compute m-th column of the preference matrix
            preference_column = np.where(r <= 5 * tau, np.exp(- r / tau), 0)  # T-Linkage
            # preference_column = np.where(r < tau, 1, 0)  # J-Linkage
            # endregion

            preference_matrix.append(preference_column)
            # endregion

    # region Return Preference matrix
    # The preference matrix has the points as rows, we need to compute the transpose
    preference_matrix = np.array(preference_matrix).transpose()
    return preference_matrix
    # endregion
