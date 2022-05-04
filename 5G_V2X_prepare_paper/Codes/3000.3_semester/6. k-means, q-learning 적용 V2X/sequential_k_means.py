
import math
import random
import functools

def euclideanDistance(p, q):
    return math.sqrt(sum([(q[i] - p[i]) ** 2 for i in range(len(p))]))


def hammingDistance(s1, s2):
    count = 0
    for i in range(0, len(s1)):
        if s1[i] is not s2[i]:
            count += 1
    return count


def point2str(p, c):
    return p


def dna2str(d, c):
    return (''.join(d), hammingDistance(d, c))

def quickSelect(data, n):
    """Find the nth rank ordered element (the least value has rank 0)."""
    # Source: http://code.activestate.com/recipes/269554/
    data = list(data)
    if not 0 <= n < len(data):
        raise ValueError('not enough elements for the given rank')
    while True:
        pivot = random.choice(data)
        pcount = 0
        under, over = [], []
        uappend, oappend = under.append, over.append
        for elem in data:
            if elem < pivot:
                uappend(elem)
            elif elem > pivot:
                oappend(elem)
            else:
                pcount += 1
        if n < len(under):
            data = under
        elif n < len(under) + pcount:
            return pivot
        else:
            data = over
            n -= len(under) + pcount

def kmeanpp(X, k, distance):
    """Perform K-Mean++ initial centroid selection."""
    # Source: http://yongsun.me/2008/10/k-means-and-k-means-with-python/
    ntries = int(2 + math.log(k))
    n = len(X)
    centers = [X[random.randint(0, n)]]
    D = [distance(x, centers[0]) ** 2 for x in X]
    Dsum = functools.reduce(lambda x, y: x + y, D)
    for _ in range(k - 1):
        bestDsum = bestIdx = -1
        for _ in range(ntries):
            randVal = random.random() * Dsum
            for i in range(n):
                if randVal <= D[i]:
                    break
                else:
                    randVal -= D[i]
            tmpDsum = functools.reduce(lambda x, y: x + y,
                             (min(D[j], distance(X[j], X[i]) ** 2)
                             for j in range(n)))
            if bestDsum < 0 or tmpDsum < bestDsum:
                bestDsum, bestIdx = tmpDsum, i
        Dsum = bestDsum
        centers.append(X[bestIdx])
        D = [min(D[i], distance(X[i], X[bestIdx]) ** 2) for i in range(n)]
    return centers

class Cluster:
    def __init__(self, points):
        self.points = points
        self.centroid = self.calcCentroid()

    def add(self, point):
        self.points.append(point)

    def clear(self):
        self.points = []

    def calcCentroid(self):
        # Perform point-wise averaging to calculate centroid for 2D points.
        if (isinstance(self.points[0][0], float)
            or isinstance(self.points[0][0], int)):
            return [(sum([p[i] for p in self.points], 0.0) /
                    float(len(self.points))) for i in
                    range(len(self.points[0]))]

        # Begin DNA String centroid calculation
        centroid = []
        randIndex = []
        countPicked = [0 for x in range(len(self.points))]
        # Iterate over the characters in the string.
        for i in range(len(self.points[0])):
            aCount = 0
            tCount = 0
            gCount = 0
            cCount = 0
            picked = 'r'
            # Iterate over all the strings in the cluster.
            for j in range(len(self.points)):
                # Increment the counter for a letter hit.
                if self.points[j][i] == 'a':
                    aCount += 1
                elif self.points[j][i] == 'g':
                    gCount += 1
                elif self.points[j][i] == 'c':
                    cCount += 1
                elif self.points[j][i] == 't':
                    tCount += 1
            # Mark the highest occuring letter in the group.
            if aCount > gCount and aCount > cCount and aCount > tCount:
                centroid.append('a')
                picked = 'a'
            elif gCount > aCount and gCount > tCount and gCount > cCount:
                centroid.append('g')
                picked = 'g'
            elif cCount > aCount and cCount > gCount and cCount > tCount:
                centroid.append('c')
                picked = 'c'
            elif tCount > aCount and tCount > gCount and tCount > cCount:
                centroid.append('t')
                picked = 't'
            else:
                # There was a tie in highest counts.
                centroid.append('r')
                maxVal = max([aCount, gCount, tCount, cCount])
                tied = []
                if aCount == maxVal:
                    tied.append('a')
                if cCount == maxVal:
                    tied.append('c')
                if gCount == maxVal:
                    tied.append('g')
                if tCount == maxVal:
                    tied.append('t')
                randIndex.append({'index': i, 'vals': tied})
            # Count the number of times the strings' character was chosen.
            for j in range(len(self.points)):
                if self.points[j][i] == picked:
                    countPicked[j] += 1
        # Pick the char that helps the string that "won" the least.
        for x in randIndex:
            tiedIndex = []
            for j in range(len(self.points)):
                for nuec in x['vals']:
                    if self.points[j][x['index']] == nuec:
                        tiedIndex.append(j)
            min = countPicked[tiedIndex[0]]
            minindex = 0
            for i in tiedIndex:
                if countPicked[i] < min:
                    min = countPicked[i]
                    minindex = i
            centroid[x['index']] = self.points[minindex][x['index']]

        self.centroid = centroid
        return centroid