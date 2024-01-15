import numpy as np

from scipy.stats import wasserstein_distance
from scipy.special import kl_div


def HellingerDistance(i, o):
    i = np.array(i)
    o = np.array(o)
    i = np.divide(i, len(i))
    o = np.divide(o, len(o))
    HD = np.linalg.norm(np.sqrt(i) - np.sqrt(o)) / np.sqrt(2)
    return HD

def WassersteinDistance(i, o):
    dists = np.divide([j + 0.5 for j in range(len(i))], len(i))
    wasserstein = wasserstein_distance(dists, dists, i, o)
    return wasserstein

def KullbackLeibler(i, o):
    i = np.array(i)
    o = np.array(o)
    i = np.divide(i, len(i)) + 1e-14
    o = np.divide(o, len(o)) + 1e-14
    KL = sum(kl_div(o, i))
    return KL

def ThetaDistance(i, o):
    i = np.array(i)
    o = np.array(o)
    variance = i - o
    distance = np.divide(sum(j for j in variance if j > 0), len(variance))
    return distance

def ChiDistance(i, o):
    i = np.array(i)
    o = np.array(o)
    variance = i - o
    distance = np.divide(sum(j for j, k in zip(variance, i) if k > 0), len(variance))
    return distance

def IDistance(i, o):
    i = np.array(i)
    o = np.array(o)
    variance = i - o
    distance = np.divide(sum(i * variance), len(variance))
    return distance

def ImaxDistance(i, o):
    i = np.array(i)
    o = np.array(o)
    i_max = np.max(i)
    distance = i_max - np.divide(sum(i * o), len(i))
    return distance