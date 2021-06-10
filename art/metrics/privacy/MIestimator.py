# import numpy as np
# import scipy
# import math

# from art.utils import check_and_transform_label_format, is_probability

# x= np.ndarray.fromfile("/home/hilla/Hilla/tehnion/IBM project/dataset statlog/german.data-numeric", sep=" ")
# print(x)
# print(x.size)
# def MI(x: np.ndarray,
#      y: np.ndarray) -> int:


# def entropy(x: np.ndarray)->int:


# Python 2.7
# Written by Shuyang Gao (BiLL), email: gaos@usc.edu


from scipy import stats
import numpy as np
import scipy.spatial as ss
from scipy.special import digamma, gamma
import numpy.random as nr
import random
# import matplotlib.pyplot as plt
import re
from scipy.stats.stats import pearsonr
import numpy.linalg as la
from numpy.linalg import eig, inv, norm, det
from scipy import stats
from math import log, pi, hypot, fabs, sqrt
from sklearn import neighbors


class MI:

    @staticmethod
    def zip2(*args):
        # zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
        # E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
        return [sum(sublist, []) for sublist in zip(*args)]

    @staticmethod  ## WE CHANGED ORIGINAL FUNCTION!!! NOW P IS A PARAMETER
    def avgdigamma(points, dvec, p=float('inf')):
        ## calculates the avg over N, in a specific dimension i (1 <= i <= d)! see formula 15
        # This part finds number of neighbors in some radius in the marginal space
        # returns expectation value of <psi(nx)>
        N = len(points)
        if p != float('inf'):
            metric = neighbors.DistanceMetric.get_metric('minkowski', p=p)
        else:
            metric = neighbors.DistanceMetric.get_metric('chebyshev')
        tree = neighbors.BallTree(points, metric=metric)
        avg = 0.
        for i in range(N):
            dist = dvec[i]
            # subtlety, we don't include the boundary point,
            # but we are implicitly adding 1 to kraskov def bc center point is included
            num_points = tree.query_radius([points[i]], dist - 1e-15, count_only=True)[0]
            avg += digamma(num_points) / N
        return avg

    @staticmethod
    def __joint_space_dist(z1, z2, **kwargs):
        ''' This is a custom distance function between samples in the joint space. ||z1-z2|| =  max{||x1-x2||, ||y1-y2||}
            Metric for x - p minkowski distance. Metric for y- chebyshev distance (induces by inf norm), we are assuming for now that y is one dimensional, i.e. y in R.
        '''
        p_x = kwargs["metric_params"]["p_x"]
        p_y = kwargs["metric_params"]["p_y"]
        size_of_y = kwargs["metric_params"]["size_of_y"]
        x1 = z1[:-size_of_y]
        x2 = z2[:-size_of_y]
        if p_x != float('inf'):
            x_dist = ss.distance.minkowski(x1, x2, p=p_x)
        else:
            x_dist = ss.distance.chebyshev(x1, x2)
        y1 = z1[-size_of_y:]
        y2 = z2[-size_of_y:]
        if p_y != float('inf'):
            y_dist = ss.distance.minkowski(y1, y2, p=p_y)
        else:
            y_dist = ss.distance.chebyshev(y1, y2)
        return max(x_dist, y_dist)

    @staticmethod
    def __marginal_space_dist(q1, q2, p=float('inf')):
        if p != float('inf'):
            dist = ss.distance.minkowski(q1, q2, p=p)
        else:
            dist = ss.distance.chebyshev(q1, q2)
        return dist

    @staticmethod
    def mi_Kraskov_HnM(X, Y, k=5, p_x=float('inf'), p_y=float('inf'), base=np.exp(1), intens=1e-10):
        ''' X is Nxd matrix, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]] if X has 3 attributes and we have two samples
            Y is Nxa matrix, e.g. Y = [[1.0],[0.0]] if we have 1 attribute two samples. number of samples in X and Y must match
            p_x\y indicates which p-norm to use on X\Y. float, 1<=p<=inf
        '''

        assert k <= len(Y) - 1, "Set k smaller than num. samples - 1"
        assert len(X) == len(Y), "Number of samples in X and Y must match"
        assert isinstance(p_x, float), "p_x must be float"
        assert p_x >= 1, "p must be larger or equal to 1"
        assert isinstance(p_y, float), "p_x must be float"
        assert p_y >= 1, "p must be larger or equal to 1"

        X = np.array(X)
        Y = np.array(Y, ndmin=2)  ## making sure Y is a column vector
        N = X.shape[0]
        # adding small noise to X and Y, e.g., x<-X+noise, y<-Y_noise
        # QUESTION???? should we add the same noise to each row in X? we think YES
        for i in range(N):
            X[i] += (intens * nr.rand(1)[0])
            Y[i] += (intens * nr.rand(1)[0])

        points = np.concatenate((X, Y),
                                axis=1)  ## Marginal space Z=(X,Y). ||z-z'|| = max {||x-x'||, ||y-y'||}. x is d dimensional and y is 1 dimensional. Using max norm in R^d and in R is equivalent to max norm in R^(d+1). NEED TO MAKE SURE THAT DATA IS NORMALIZED???
        if p_x == float('inf') and p_y == float('inf'):
            joint_space_metric = neighbors.DistanceMetric.get_metric('chebyshev')
        else:
            joint_space_metric = neighbors.DistanceMetric.get_metric('pyfunc', func=MI.__joint_space_dist,
                                                                     metric_params={"p_x": p_x, "p_y": p_y, "size_of_y":Y.shape[1]})
        tree = neighbors.BallTree(points, metric=joint_space_metric)
        d_vec = np.zeros((2, N)) - 1
        ''' Denote the j'th sample in the marginal space Z=(X,Y) by (x_j, y_j).
            Denote the k-th nearest neighbors IN THE MARGINAL SPACE to the j'th sample by (x'_1,y'_1),(x'_2,y'_2),...,(x'_k,y'_k).
            Eventually, after next loop, d_vec[0][j] will be max{||x_j - x'_i|| : 1<=i<=k} , d_vec[1][j] will be max{||y_j - y'_i|| : 1<=i<=k}
            Meaning- d_vec[i][j] is exactly 1/2 * epsilon_j,k{x_i}, (half of) the length of the hyper rectangle in the i'th dimension (i=1,2)   (see section 2.3 KSG Estimator in https://arxiv.org/pdf/1411.2003.pdf)    
        '''
        sample = -1
        for point in points:
            sample += 1
            indices = tree.query([point], k + 1, return_distance=False)[0]
            for i in indices:
                neighbor = points[i]
                ####### old code, see bug fix below
                if d_vec[0][sample] < np.max(np.fabs(points[sample][:-Y.shape[1]] - neighbor[:-Y.shape[1]])):
                    d_vec[0][sample] = np.max(np.fabs(points[sample][:-Y.shape[1]] - neighbor[:-Y.shape[1]]))
                if d_vec[1][sample] < np.max(np.fabs(points[sample][-Y.shape[1]:] - neighbor[-Y.shape[1]:])):
                    d_vec[1][sample] = np.max(np.fabs(points[sample][-Y.shape[1]:] - neighbor[-Y.shape[1]:]))
                ######
                ''' NEED TO TEST MORE BEFORE PUSH!!!
                point_x_dist_from_ith_ngbr = MI.__marginal_space_dist(point[:-Y.shape[1]], neighbor[:-Y.shape[1]], p_x)
                point_y_dist_from_ith_ngbr = MI.__marginal_space_dist(point[-Y.shape[1]:], neighbor[-Y.shape[1]:], p_y)
                if d_vec[0][sample] < point_x_dist_from_ith_ngbr:
                    d_vec[0][sample] = point_x_dist_from_ith_ngbr
                if d_vec[1][sample] < point_y_dist_from_ith_ngbr:
                    d_vec[1][sample] = point_y_dist_from_ith_ngbr
                '''

        avg_digamma = MI.avgdigamma(X, d_vec[0], p=p_x) + MI.avgdigamma(Y, d_vec[1], p=p_y)
        return digamma(N) + digamma(k) - 1 / k - avg_digamma

    @staticmethod
    def mi_Kraskov(X, k=5, base=np.exp(1), intens=1e-10):
        '''The mutual information estimator by Kraskov et al.
           ith row of X represents ith dimension of the data, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]], if X has two dimensions and we have three samples

        '''
        # adding small noise to X, e.g., x<-X+noise
        x = []
        for i in range(len(X)):
            tem = []
            for j in range(len(X[i])):
                tem.append([X[i][j] + intens * nr.rand(1)[0]])
            x.append(tem)

        points = []
        for j in range(len(x[0])):
            tem = []
            for i in range(len(x)):
                tem.append(x[i][j][0])
            points.append(tem)
        tree = ss.cKDTree(points)

        dvec = []
        for i in range(len(x)):
            dvec.append([])
        for point in points:
            # Find k-nearest neighbors in joint space, p=inf means max norm
            knn = tree.query(point, k + 1, p=float('inf'))
            points_knn = []
            for i in range(len(x)):
                dvec[i].append(float('-inf'))
                points_knn.append([])
            for j in range(k + 1):
                for i in range(len(x)):
                    points_knn[i].append(points[knn[1][j]][i])

            # Find distances to k-nearest neighbors in each marginal space
            for i in range(k + 1):
                for j in range(len(x)):
                    if dvec[j][-1] < fabs(points_knn[j][i] - points_knn[j][0]):
                        dvec[j][-1] = fabs(points_knn[j][i] - points_knn[j][0])

        ret = 0.
        for i in range(len(x)):
            ret -= MI.avgdigamma(x[i], dvec[i])
        ret += digamma(k) - (float(len(x)) - 1.) / float(k) + (float(len(x)) - 1.) * digamma(len(x[0]))
        return ret

    @staticmethod
    def mi_LNC(X, k=5, base=np.exp(1), alpha=0.25, intens=1e-10):
        '''The mutual information estimator by PCA-based local non-uniform correction(LNC)
           ith row of X represents ith dimension of the data, e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]], if X has two dimensions and we have three samples
           alpha is a threshold parameter related to k and d(dimensionality), please refer to our paper for details about this parameter
        '''
        # N is the number of samples
        N = len(X[0])

        # First Step: calculate the mutual information using the Kraskov mutual information estimator
        # adding small noise to X, e.g., x<-X+noise
        x = []
        for i in range(len(X)):  ## len(x) = d
            tem = []
            for j in range(len(X[i])):  ## len(x[i]) = N
                tem.append([X[i][j] + intens * nr.rand(1)[0]])
            x.append(tem)

        points = []  ## after next loop points is X transpose (rows are points, each entry is an attribute- makes more sense)
        for j in range(len(x[0])):
            tem = []
            for i in range(len(x)):
                tem.append(x[i][j][0])
            points.append(tem)
        tree = ss.cKDTree(points)
        dvec = []
        for i in range(len(x)):
            dvec.append([])
        for point in points:
            # Find k-nearest neighbors in joint space, p=inf means max norm
            knn = tree.query(point, k + 1, p=float('inf'))
            points_knn = []
            for i in range(len(x)):
                dvec[i].append(float('-inf'))
                points_knn.append([])
            for j in range(k + 1):
                for i in range(len(x)):
                    points_knn[i].append(points[knn[1][j]][i])

            # Find distances to k-nearest neighbors in each marginal space
            for i in range(k + 1):
                for j in range(len(x)):
                    if dvec[j][-1] < fabs(points_knn[j][i] - points_knn[j][0]):
                        dvec[j][-1] = fabs(points_knn[j][i] - points_knn[j][
                            0])  ### d_vec dim is dXN dvec[i][j] is the distance of the kth neighbor of sample j w.r.t to ith dim. AKA n_{x_j}(i) in formula 15

        ret = 0.
        for i in range(len(x)):  ## len(x) = d
            ret -= MI.avgdigamma(x[i], dvec[i])  ## calculating diggama for each dimension
        ret += digamma(k) - (float(len(x)) - 1.) / float(k) + (float(len(x)) - 1.) * digamma(len(x[0]))  ## formula 15

        # Second Step: Add the correction term (Local Non-Uniform Correction)
        e = 0.
        tot = -1
        for point in points:
            tot += 1
            # Find k-nearest neighbors in joint space, p=inf means max norm
            knn = tree.query(point, k + 1, p=float('inf'))
            knn_points = []
            for i in range(k + 1):
                tem = []
                for j in range(len(point)):
                    tem.append(points[knn[1][i]][j])
                knn_points.append(tem)

            # Substract mean	of k-nearest neighbor points
            for i in range(len(point)):
                avg = knn_points[0][i]
                for j in range(k + 1):
                    knn_points[j][i] -= avg

            # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
            covr = []
            for i in range(len(point)):
                tem = 0
                covr.append([])
                for j in range(len(point)):
                    covr[i].append(0)
            for i in range(len(point)):
                for j in range(len(point)):
                    avg = 0.
                    for ii in range(1, k + 1):
                        avg += knn_points[ii][i] * knn_points[ii][j] / float(
                            k)  ## multiplying (K^T)(K) where K dim is kXd
                    covr[i][j] = avg
            w, v = la.eig(covr)

            # Calculate PCA-bounding box using eigen vectors
            V_rect = 0
            cur = []
            for i in range(len(point)):
                maxV = 0.
                for j in range(0, k + 1):
                    tem = 0.
                    for jj in range(len(point)):
                        tem += v[jj, i] * knn_points[j][
                            jj]  ## projecting each neighbor of point to the direction of each eigen vector, and taking the distance of the further one (for a specific entry- 1 to d)
                    if fabs(tem) > maxV:
                        maxV = fabs(tem)
                cur.append(maxV)
                V_rect = V_rect + log(cur[
                                          i])  ## calculatin log(volume of rotated bounding box)  [V_rec is already the log of the vilume!]

            # Calculate the volume of original box
            log_knn_dist = 0.
            for i in range(len(dvec)):
                log_knn_dist += log(dvec[i][tot])

            # Perform local non-uniformity checking
            if V_rect >= log_knn_dist + log(alpha):
                V_rect = log_knn_dist

            # Update correction term
            if (log_knn_dist - V_rect) > 0:
                e += (log_knn_dist - V_rect) / N

        return (ret + e) / log(base)  ## formula 17

    @staticmethod
    def entropy_old(x, p=float('inf'), k=3, base=np.exp(1), intens=1e-10):  ## WE CHANGED p TO BE A PARAMETER!

        """ The classic K-L k-nearest neighbor continuous entropy estimator
            x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        d = len(x[0])
        N = len(x)
        x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
        tree = ss.cKDTree(x)
        nn = [tree.query(point, k + 1, p=p)[0][k] for point in x]  ## in the ith entry 0.5epsilon_i,k
        const = digamma(N) - digamma(k) + d * log(2)  ## last term is to correct 0.5epsilon to epsilon
        return (const + d * np.mean(list(map(log, nn)))) / log(base)  ## formula 14

    @staticmethod
    def entropy(x, p=float('inf'), k=3, base=np.exp(1), intens=1e-10):  ## WE CHANGED p TO BE A PARAMETER!

        """ The classic K-L k-nearest neighbor continuous entropy estimator
            x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        d = len(x[0])
        N = len(x)
        x = [list(a + intens * nr.rand(len(x[0]))) for a in x]
        if p != float('inf'):
            metric = neighbors.DistanceMetric.get_metric('minkowski', p=p)
        else:
            metric = neighbors.DistanceMetric.get_metric('chebyshev')
        tree = neighbors.BallTree(x, metric=metric)
        nn = [tree.query([point], k + 1)[0][0][k] for point in x]  ## in the ith entry 0.5epsilon_i,k
        const = digamma(N) - digamma(k) + d * log(2)  ## last term is to correct 0.5epsilon to epsilonxc
        return (const + d * np.mean(list(map(log, nn)))) / log(base)  ## formula 14

    @staticmethod
    def entropy_with_correction(x, p=float('inf'), k=3, base=np.exp(1),
                                intens=1e-10):  ## WE CHANGED p TO BE A PARAMETER!

        """ The classic K-L k-nearest neighbor continuous entropy estimator
            x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        d = len(x[0])
        N = len(x)
        x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
        if p != float('inf'):
            metric = neighbors.DistanceMetric.get_metric('minkowski', p=p)
        else:
            metric = neighbors.DistanceMetric.get_metric('chebyshev')
        tree = neighbors.BallTree(x, metric=metric)
        nn = [tree.query([point], k + 1)[0][0][k] for point in x]  ## in the ith entry 0.5epsilon_i,k
        const = digamma(N) - digamma(k) + d * log(2)  ## last term is to correct 0.5epsilon to epsilonxc
        if p != float('inf'):
            const += log((gamma(1 + 1 / p)) / gamma(1 + d / p))           ## adding log c_d as in Kraskuv eq. 20 (c_d = {volume of d dim. unit ball in p norm} / 2^d)
        return (const + d * np.mean(list(map(log, nn)))) / log(base)  ## formula 14


'''
N = 5000  # total number of samples

# 2D Linear
noise = 1e-7
x = []
for i in range(N):
    x.append(nr.rand(1)[0])
y = []
y_for_ent = []
for i in range(N):
    y.append(x[i] + nr.rand(1)[0] * noise)
    y_for_ent.append([y[-1]])

usedN = 500  # number of samples used for calculation
print('Testing 2D linear relationship Y=X+Uniform_Noise')
print('noise level=' + str(noise) + ", Nsamples = " + str(usedN))
print('True MI(x:y)', MI.entropy(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('True MI(x:y) ballll', MI.entropy_ball(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('Kraskov MI(x:y)', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=1, base=np.exp(1), intens=0.0))
Y = [[z] for z in y[:usedN]]
X = [[z] for z in x[:usedN]]
print('Kraskov hnmmmmm MI(x:y)', MI.mi_Kraskov_HnM(X, Y, k=1,p_x=2.0, base=np.exp(1), intens=0.0))
print('LNC MI(x:y)', MI.mi_LNC([x[:usedN], y[:usedN]], k=5, base=np.exp(1), alpha=0.25, intens=0.0))
print()

# 2D Quadratic
noise = 1e-7
x = []
for i in range(N):
    x.append(nr.rand(1)[0])
y = []
y_for_ent = []
for i in range(N):
    y.append(x[i] * x[i] + nr.rand(1)[0] * noise)
    y_for_ent.append([y[-1]])

usedN = 1000  # number of samples used for calculation
print('Testing 2D quadratic relationship Y=X^2+Uniform_Noise')
print('noise level=' + str(noise) + ", Nsamples = " + str(usedN))
print('True MI(x:y)', MI.entropy(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('True MI(x:y) ballll', MI.entropy_ball(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('Kraskov MI(x:y)', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=1, base=np.exp(1), intens=0.0))
Y = [[z] for z in y[:usedN]]
X = [[z] for z in x[:usedN]]
print('Kraskov hnmmmm MI(x:y)', MI.mi_Kraskov_HnM(X, Y, k=1, base=np.exp(1), intens=0.0))
print('LNC MI(x:y)', MI.mi_LNC([x[:usedN], y[:usedN]], k=5, base=np.exp(1), alpha=0.25, intens=0.0))
print()

print("blaaaaaaaaaaaaaaaaa")

# 2D Quadratic
noise = 1e-7
x = []
for i in range(N):
    x.append(nr.rand(1)[0])
y = []
y_for_ent = []
for i in range(N):
    y.append(x[i] * x[i] + nr.rand(1)[0] * noise)
    y_for_ent.append([y[-1]])

usedN = 1000  # number of samples used for calculation
print('Testing 2D quadratic relationship Y=X^2+Uniform_Noise')
print('noise level=' + str(noise) + ", Nsamples = " + str(usedN))
print('True MI(x:y)', MI.entropy(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('True MI(x:y) ballll', MI.entropy_ball(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('Kraskov MI(x:y)', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=1, base=np.exp(1), intens=0.0))
Y = [[z,z,z] for z in y[:usedN]]
X = [[z,z+1,z,z+3,z,z+2.3,z] for z in x[:usedN]]
print('Kraskov hnmmmm MI(x:y)', MI.mi_Kraskov_HnM(X, Y, k=1, p_y=2.0,p_x=2.0,  base=np.exp(1), intens=0.0))
print('LNC MI(x:y)', MI.mi_LNC([x[:usedN], y[:usedN]], k=5, base=np.exp(1), alpha=0.25, intens=0.0))
print()
'''
