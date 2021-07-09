"""
This is a Mutual Information and Entropy estimator, implemented by Hilla Schefler and Maya Avidor,
as part of Industrial Project course 234313, CS department, Technion.
Implementation is based on 2 papers:
(1) Efficient Estimation of Mutual Information for Strongly Dependent Variables- https://arxiv.org/pdf/1411.2003.pdf
(2) Estimating mutual information- https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138


Hilla - hillas@campus.technion.ac.il
Maya - mayaavidor1@gmail.com
"""

import numpy as np
import scipy.spatial as ss
from scipy.special import digamma, gamma
import numpy.random as nr
from math import log
from sklearn import neighbors


class MI:

    @staticmethod
    def avgdigamma(points, dvec, p=float('inf')):
        """
        Let rect = [x1(i),y1(i)]x[x2(i),y2(i)] be the minimal rectangle that contains k nearest neighbors of the i-th'
        sample in the joint space XxY.
        Let n_x(i) / n_y(i) be the number of neighbors of the i-th' sample within epsilon_x(i) / epsilon_y(i) distance
        from it in the marginal space X / Y accordingly, where epsilon_x(i) = dist(x1(i),x2(i))/2, epsilon_y(i) = dist(y1(i),y2(i))/2
        (distance function determined by p).
        This function calculates and returns average digamma of number of neighbors of each sample in (one of)
        the marginal space- see formula 15 in paper (1) (In our case d=2).
        @param points: List of points in marginal space W where W is either X or Y.
                        points shape = (N,l) where N is the number of points\ samples and l is the dimension of each point/ sample.
        @param dvec: List of distances. dvec shape = (N,). dvec[i] = epsilon_w(i) where w is either x or y.
        @param p: p norm for calculating distance in marginal space W. float. Note- dim(W)=l.
        @return: <digamma(number of neighbors of each sample)>. (<> is Average over number of samples N. See formula 15 in paper (1)).
        """
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
        """
        This is a custom distance function between samples in the joint space Z = XxY. ||z1-z2|| =  max{||x1-x2||, ||y1-y2||}
        Note- X, Y don't have necessarily the the same dimension (moreover, in most cases dim(X)!=dim(Y)).
        Metric for x - p_x Minkowski distance (induced by p_x norm) or Chebyshev distance (induced by inf norm).
        Metric for y- p_y Minkowski distance or Chebyshev distance (induced by inf norm).
        @param z1: First sample in joint space. A list of length = dim(X) + dim(Y)
        @param z2: Second sample in joint space. A list of length = dim(X) + dim(Y)
        @param kwargs: Additional params- p_x, p_y, and size_of_y (== dim(Y))
        @return: distance between z1 an z2 in joint space: max{||x1-x2||, ||y1-y2||}
        """
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
        """
        This is a custom distance function between samples in marginal space W (where W is either X or Y).
        @param q1: First sample in marginal space W. A list of size dim(W).
        @param q2: Second sample in marginal space W. A list of size dim(W).
        @param p: p- norm used for distance calculation.
        @return: Distance between q1 and q2
        """
        if p != float('inf'):
            dist = ss.distance.minkowski(q1, q2, p=p)
        else:
            dist = ss.distance.chebyshev(q1, q2)
        return dist

    @staticmethod
    def mi_Kraskov_HnM(X, Y, k=5, p_x=float('inf'), p_y=float('inf'), intens=1e-10):
        """
        We want to estimate the mutual information between two random variables given N i.i.d samples from their distributions.
        This is a KNN based Mutual Information estimator between X and Y.
        @param X: Nxa matrix, 'N' is the number of samples and 'a' is number of attributes of X (==dim(X)).
                    e.g. X = [[1.0,3.0,3.0],[0.1,1.2,5.4]] if X has 3 attributes and we have two samples
        @param Y: Nxb matrix, 'N' is the number of samples and 'b' is number of attributes of Y (==dim(Y)).
                    Y = [[1.0],[0.0]] if we have 1 attribute two samples. Number of samples in X and Y must match.
        @param k: Number of neighbors to use in KNN.
        @param p_x: p- norm to use in X space. ( ||x||_p_x = (sum {from i=1 to a} of ((x_i)^p))^(1/p) if p_x is not 'inf', else,
                    ||x||_p_x = max{x_i : 1<=i<=a}  )
        @param p_y: p- norm to use in Y space. ( ||y||_p_y = (sum {from i=1 to b} of ((y_i)^p))^(1/p) if p_y is not 'inf', else,
                    ||y||_p_y = max{y_i : 1<=i<=b}  )
        @param intens: Maximum noise to add to samples.
        @return: MI(X,Y)
        """

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
        for i in range(N):
            X[i] += (intens * nr.rand(1)[0])
            Y[i] += (intens * nr.rand(1)[0])

        points = np.concatenate((X, Y), axis=1)  ## Marginal space Z=XxY. ||z-z'|| = max {||x-x'||, ||y-y'||}
        if p_x == float('inf') and p_y == float('inf'):
            joint_space_metric = neighbors.DistanceMetric.get_metric('chebyshev') ## Using max norm in R^a and in R^b is equivalent to max norm in R^(a+b).
        else:
            joint_space_metric = neighbors.DistanceMetric.get_metric('pyfunc', func=MI.__joint_space_dist,
                                                                     metric_params={"p_x": p_x, "p_y": p_y,
                                                                                    "size_of_y": Y.shape[1]})
        tree = neighbors.BallTree(points, metric=joint_space_metric)
        d_vec = np.zeros((2, N)) - 1
        ''' 
        Denote the j'th sample in the marginal space Z=XxY by (x_j, y_j).
        Denote the k-th nearest neighbors IN THE MARGINAL SPACE to the j'th sample by (x'_1,y'_1),(x'_2,y'_2),...,(x'_k,y'_k).
        Eventually, after next loop, d_vec[0][j] will be max{||x_j - x'_i|| : 1<=i<=k} , d_vec[1][j] will be max{||y_j - y'_i|| : 1<=i<=k}
        Meaning- d_vec[i][j] is exactly 1/2 * epsilon_j,k{x_i}, (half of) the length of the hyper rectangle in the i'th dimension (i=1,2)   (see section 2.3 KSG Estimator in paper (1))    
        '''
        sample = -1
        for point in points:
            sample += 1
            indices = tree.query([point], k + 1, return_distance=False)[0]
            for i in indices:
                neighbor = points[i]
                point_x_dist_from_ith_ngbr = MI.__marginal_space_dist(point[:-Y.shape[1]], neighbor[:-Y.shape[1]], p_x)
                point_y_dist_from_ith_ngbr = MI.__marginal_space_dist(point[-Y.shape[1]:], neighbor[-Y.shape[1]:], p_y)
                if d_vec[0][sample] < point_x_dist_from_ith_ngbr:
                    d_vec[0][sample] = point_x_dist_from_ith_ngbr
                if d_vec[1][sample] < point_y_dist_from_ith_ngbr:
                    d_vec[1][sample] = point_y_dist_from_ith_ngbr

        avg_digamma = MI.avgdigamma(X, d_vec[0], p=p_x) + MI.avgdigamma(Y, d_vec[1], p=p_y)
        return digamma(N) + digamma(k) - 1 / k - avg_digamma    ## See formula 15 paper (1)

    @staticmethod
    def entropy(x, p=float('inf'), k=3, base=np.exp(1), intens=1e-10):
        """
        The classic K-L k-nearest neighbor continuous entropy estimator.
        We want to estimate the entropy of a random variable given N i.i.d samples from it's distribution.
        @param x: Nxa matrix, 'N' is the number of samples and 'a' is number of attributes of X (==dim(X)).
                    e.g. x = [[1.3],[3.7],[5.1],[2.4]] if x is a one-dimensional scalar and we have four samples
        @param p: p- norm. ( ||x||_p_x = (sum {from i=1 to a} of ((x_i)^p))^(1/p) if p_x is not 'inf', else,
                    ||x||_p_x = max{x_i : 1<=i<=a}  )
        @param k: Number of neighbors to use in KNN.
        @param base: base of logarithm in entropy.
        @param intens: Maximum noise to add to samples.
        @return: Entropy of x.
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
        nn = [tree.query([point], k + 1)[0][0][k] for point in x]  ## in the i-th' entry 0.5epsilon_i,k
        const = digamma(N) - digamma(k) + d * log(2)  ## last term is to correct 0.5epsilon to epsilon
        return (const + d * np.mean(list(map(log, nn)))) / log(base)  ## formula 14 paper (1)

    @staticmethod
    def entropy_with_correction(x, p=float('inf'), k=3, base=np.exp(1), intens=1e-10):
        """
        The classic K-L k-nearest neighbor continuous entropy estimator.
        We want to estimate the entropy of a random variable given N i.i.d samples from it's distribution.
        This estimator include a correction term: log({volume of d dim. unit ball in p norm} / 2^d)
        See equation 20 in paper (2) for more details.
        @param x: Nxa matrix, 'N' is the number of samples and 'a' is number of attributes of X (==dim(X)).
                    e.g. x = [[1.3],[3.7],[5.1],[2.4]] if x is a one-dimensional scalar and we have four samples
        @param p: p- norm. ( ||x||_p_x = (sum {from i=1 to a} of ((x_i)^p))^(1/p) if p_x is not 'inf', else,
                    ||x||_p_x = max{x_i : 1<=i<=a}  )
        @param k: Number of neighbors to use in KNN.
        @param base: base of logarithm in entropy.
        @param intens: Maximum noise to add to samples.
        @return: Entropy of x.
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
            const += log((gamma(1 + 1 / p)) / gamma(
                1 + d / p))  ## adding log c_d as in Kraskuv (paper (2)) equation 20 (c_d = {volume of d dim. unit ball in p norm} / 2^d)
        return (const + d * np.mean(list(map(log, nn)))) / log(base)  ## formula 14 paper (1)


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
print('True MI(x:y) entropy with correction',
      MI.entropy_with_correction(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('Kraskov MI(x:y) k=1 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=1, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=3 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=3, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=5 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=5, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=11 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=11, base=np.exp(1), intens=0.0))
Y = [[z] for z in y[:usedN]]
X = [[z] for z in x[:usedN]]
print('Kraskov hnm MI(x:y) p=2, k=1 ', MI.mi_Kraskov_HnM(X, Y, k=1, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=3 ', MI.mi_Kraskov_HnM(X, Y, k=3, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=5 ', MI.mi_Kraskov_HnM(X, Y, k=5, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=11', MI.mi_Kraskov_HnM(X, Y, k=11, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF, k=1 ', MI.mi_Kraskov_HnM(X, Y, k=1, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=3 ', MI.mi_Kraskov_HnM(X, Y, k=3, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=5 ', MI.mi_Kraskov_HnM(X, Y, k=5, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=11', MI.mi_Kraskov_HnM(X, Y, k=11, base=np.exp(1), intens=0.0))

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
print('True MI(x:y) entropy with correction', MI.entropy_with_correction(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('Kraskov MI(x:y) k=1 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=1, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=3 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=3, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=5 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=5, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=11 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=11, base=np.exp(1), intens=0.0))
Y = [[z] for z in y[:usedN]]
X = [[z] for z in x[:usedN]]
print('Kraskov hnm MI(x:y) p=2 k=1 ', MI.mi_Kraskov_HnM(X, Y, k=1, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=3 ', MI.mi_Kraskov_HnM(X, Y, k=3, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=5 ', MI.mi_Kraskov_HnM(X, Y, k=5, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=11 ', MI.mi_Kraskov_HnM(X, Y, k=11, p_x=2.0, p_y=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=1 ', MI.mi_Kraskov_HnM(X, Y, k=1, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=3 ', MI.mi_Kraskov_HnM(X, Y, k=3, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=5 ', MI.mi_Kraskov_HnM(X, Y, k=5, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=11 ', MI.mi_Kraskov_HnM(X, Y, k=11, base=np.exp(1), intens=0.0))
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
print('True MI(x:y) entropy with correction', MI.entropy_with_correction(y_for_ent[:1000], k=1, base=np.exp(1), intens=0.0) - log(noise))
print('Kraskov MI(x:y) k=1 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=1, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=3 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=3, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=5 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=5, base=np.exp(1), intens=0.0))
print('Kraskov MI(x:y) k=11 ', MI.mi_Kraskov([x[:usedN], y[:usedN]], k=11, base=np.exp(1), intens=0.0))
Y = [[z, z, z] for z in y[:usedN]]
X = [[z, z + 1, z, z + 3, z, z + 2.3, z] for z in x[:usedN]]
print('Kraskov hnm MI(x:y) p=2 k=1 ', MI.mi_Kraskov_HnM(X, Y, k=1, p_y=2.0, p_x=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=3 ', MI.mi_Kraskov_HnM(X, Y, k=5, p_y=2.0, p_x=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=5 ', MI.mi_Kraskov_HnM(X, Y, k=3, p_y=2.0, p_x=2.0, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=2 k=11 ', MI.mi_Kraskov_HnM(X, Y, k=11, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=1 ', MI.mi_Kraskov_HnM(X, Y, k=1, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=3 ', MI.mi_Kraskov_HnM(X, Y, k=3, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=5 ', MI.mi_Kraskov_HnM(X, Y, k=5, base=np.exp(1), intens=0.0))
print('Kraskov hnm MI(x:y) p=INF k=11 ', MI.mi_Kraskov_HnM(X, Y, k=11, base=np.exp(1), intens=0.0))
print('LNC MI(x:y)', MI.mi_LNC([x[:usedN], y[:usedN]], k=5, base=np.exp(1), alpha=0.25, intens=0.0))
print()

'''
