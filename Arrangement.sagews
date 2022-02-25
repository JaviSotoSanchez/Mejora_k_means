︠8826a625-224b-4b56-8706-30e6181900d4s︠
#Class Polytope
K = QQ
PREC = 5
from scipy.cluster.vq import vq, kmeans, kmeans2,vq
from scipy import linalg, matrix
from itertools import izip
from copy import deepcopy as copy
data=[]
for line in open("linux2.txt"):
    mylist=[QQ(int(float(i)*100000)) for i in line.strip().split(",")]
    data.append(mylist)
    
data=data[:100]
print(data)
#data = [[QQ(0),QQ(0)],[QQ(0),QQ(1)],[QQ(0),QQ(2)],[QQ(0),QQ(3)],[QQ(1),QQ(5)],[QQ(3),QQ(2)]]

class Polytope:
    """
    New class of polytope. It is suppose that it is
    faster than the polytope class of sage
    """

    def __init__(self, dimension, ineq):
        """
        This is the constructor of the class Polytope.
        ineq is a list of inequalities and we distinguish between
        the homogenous and inhomogenous inequalities using the
        dimension
        """
        print dimension, ineq
        self.dimension = dimension
        self.length = len(ineq[0])
        self.ineq = copy(ineq)
        d = dimension + 1 -self.length
        listTemp = [([K(0)] * (d) + [K(i) for i in l]) for l in ineq ]
        self.P = Polyhedron(ieqs = listTemp, base_ring = K)
        self.indexes = self.necessary(listTemp)

    def indexes(self):
        """
        This method returns a list with the indexes (with respect to
        the list of all inequalities) of the non redundant inequalities
        """
        return self.indexes

    def necessary(self, ineq):
        """
        This method returns the indexes of the
        elements which appears as inequalities in the representation
        of the polytope
        """
        return [ineq.index(l) for l in self.P.inequalities_list()]

    def interiorPoint(self):
        """
        This method calculates an interior point of the polytope
        """
        if self.P.dim()<self.dimension:
            raise Exception( "dimension of P:" + str(self.P.dim())
                             + "attribute: "+ str(self.dimension))

        problem = MixedIntegerLinearProgram(maximization = True)
        w = problem.new_variable(nonnegative=True)
        problem.set_objective(w[0])
        problem.set_max(w[0],1)
        for l in self.P.inequality_generator():
            restriction=w[0]
            l_i = (j for j in l)
            restriction=restriction + RDF(l_i.next())
            for i, k in enumerate(l_i):
                restriction -= RDF(k)*w[i + 1]
                problem.set_min(w[i + 1], None)
            problem.add_constraint(restriction<=0)
        valueProblem = problem.solve()
        solution=[]
        long = 0
        for i,v in problem.get_values(w).iteritems():
            solution.append(v)
            long=long+1
        if long<self.dimension:
            raise Exception( "Error! empty polytope")
        return [K(element) for element in solution[1:]]

#Class Enumerator
import pickle, os
from multiprocessing import Process

def ensure_dir(d):
    '''
    creates a directory if it does not exists
    '''
    if not os.path.exists(d):
        os.makedirs(d)


class Enumerator:
    """
    This class is an abstract class to implement, in the most possible
    general way. The notation follow the one given in the article
    'Reverse search for enumeration' by Avis and Fukuda.
    There is one sutility, though: delta is not necessary for the
    implementation BUT it is necessary to exists.
    """

    def __init__(self, processes = 1, filename = 'enumeration',
                 directory = '/tmp' ):
        """
        Constructor
        processes is the number of
        processes which our enumerator will launch, filename and
        directory are used for bookmarking
        """
        self.filename = filename
        self.directory = directory
        self.m_file = os.path.join(self.directory,
                                   self.filename)
        self.processes = processes
        ensure_dir(self.directory)


    def f(self, v):
        """
        This function is the local search function
        Arguments:
        - `self`:
        - `v`: this is the vertex
        """
        pass

    def Adj(self, v):
        """
        This is the adjacency list oracle. It suppose to be a
        generator that returns, one by one, each of the adjacencies of v.
        Arguments:
        - `self`:
        - `v`: denotes the vertex for which we are looking the
               adjacencies
        """
        pass

    def reverse_m_pid(self, S, pid):
        """
        This is an auxiliary method, it shouldn't be called from
        another programmed (indeed, it should be private)
        """
        m_file = self.m_file + str(pid)
        results = {l for l in self.reverse_search(S)}
        with open(m_file,'w') as f:
            pickle.dump(results, f)

    def reverse_m(self, S):
        """
        This is the wrapper to make things go parallel. This functions
        greates as many processes as needed where each of the
        processes operates in the different branches of the
        algorithm. All the clustering is written to the hard disk and
        then given back to the user as a generator.
        """
        several_elements = S
        while len(several_elements) <= self.processes:
            for element in several_elements:
                yield element
            new_elements = []
            for s in several_elements:
                for v in self.Adj(s):
                    if self.f(v) == s:
                        new_elements.append(v)
            several_elements = new_elements
        length = len(several_elements)
        proc = self.processes
        p_list = []
        pid = 0
        for element in several_elements:
            p_list.append( Process(
                target = self.reverse_m_pid,
                args = (set([element]), pid)
                ))
            pid += 1
        for i in xrange(length/proc):
            print len(p_list)
            temp, p_list = p_list[:proc], p_list[proc:]
            [p.start() for p in temp]
            [p.join() for p in temp]
        print len(p_list)
        t1 =[p.start() for p in p_list]
        t2 = [p.join() for p in p_list]
        for i in xrange(pid):
            m_file = self.m_file + str(i)
            with open(m_file) as f:
                for l in pickle.load(f):
                    yield l


    def reverse_search(self, S):
        """
        This is the main algorithm, which is a generator
        function. This means that outputs all the elements without
        keeping them in memory.
        """
        for s in S:
            yield s
            v = s
            do = True #In the original paper there was a do-while
            Adjacencies = self.Adj(v) #I suppose that it is a generator
            while do:
                b_while = True #to avoid the use of delta and j
                while b_while:
                    try:
                        #This is a minor thing, Avis and Fukuda
                        #proposed to use a distinguished element in
                        #the case there were no more adjacencies. In
                        #Python, it is customary to throw an exception
                        #which signals the end of the adjacencies
                        Next = Adjacencies.next()
                        if self.f(Next) == v:
                            v = Next
                            Adjacencies = self.Adj(v)
                            yield v

                    except StopIteration:
                        #no more adjacencies, Avis and Fukuda would
                        #write j>= delta
                        b_while = False
                try:
                    if v == s:
                        #This is a little different because the
                        #implementation this way is simpler, but it is
                        #actually the same thing
                        v = Adjacencies.next()
                        Adjacencies = self.Adj(v)
                    else:
                        u, v = v, self.f(v)
                        Adjacencies = self.Adj(v)
                        different = u != Adjacencies.next()
                        while different:
                            different = u != Adjacencies.next()
                except StopIteration:
                    do = False


#Class Arrangement
class Arrangement(Enumerator):

    """
    This class  represents an arrangement of hyperplanes
    """

    def __init__(self, hyperplanes, pointR = [], processes = 1, filename = 'arrange',
                 directory = './tmp'):
        """
        Constructor, which takes a list of hyperplanes, representing
        by lists of the same length and a initial point pointR.
        This point can be empty and the class generates it at random.
        """
        Enumerator.__init__(self, processes, filename, directory)
        self.hyperplanes = [[i/gcd(l) for i in l] for l in hyperplanes]
        self.V = VectorSpace(K, len(hyperplanes[0]))
        if pointR:
            self.pointR = self.V(pointR)
        else:
            #Pick a first cell at
            #random, it is represented by a point
            self.pointR=self.firstPoint()

        self.initial_c = tuple( sign(self.V(L).dot_product(self.pointR))
                                for L in self.hyperplanes)
        listAux = []
        for l in self.hyperplanes:
            L=self.V(l)
            listAux.append(L*sign(L.dot_product(self.pointR)))
        self.hyperplanes = listAux


    def firstPoint(self):
        """
        This method selects a random point which does not belong in
        any of the hyperplanes
        """
        product=0
        pointR=self.V.random_element()
        while any(self.V(l).dot_product(pointR)==0
                  for l in self.hyperplanes):
            pointR=self.V.random_element()
        return pointR


    def interior_point(self,c):
        """
        Returns an interior point of the cell that have
        the signs given by c, which must be non-empty.
        """
        cell = []
        for c_i, h_i  in izip(c, self.hyperplanes):
            cell.append((c_i * h_i).list())
        p_aux =Polytope(self.V.dimension(),cell).interiorPoint()
        return self.V(p_aux)

    def f(self, v):
        """
        Returns which elements go before this element
        """
        p = self.interior_point(v)
        self.p = p
        if p == self.V(0):
            print "Empty cell!"+str(v)
            return c
        distance=infinity
        position, counter = 0, -1
        betterHyperplane=self.hyperplanes[0]/(-p.dot_product(self.hyperplanes[0]))
        for  hyperplane, c_counter in izip(self.hyperplanes, v):
            counter += 1
            dotProduct = hyperplane.dot_product(-p)
            if hyperplane.dot_product(self.pointR-p)!=0:
                auxDistance = dotProduct/hyperplane.dot_product(self.pointR-p)
            else:
                auxDistance = infinity
            if ((distance > auxDistance and 0 < auxDistance < 1 and
                 c_counter ==-1) or
                 (distance==auxDistance and
                  self.order(betterHyperplane,hyperplane/(dotProduct)))
                ):
                position, distance = counter, auxDistance
                betterHyperplane=hyperplane/(dotProduct)
        e=list(v)
        e[position]=1
        return tuple(e)

    def order(self,V,V1):
        """
        This function is returns True if V> V1, where > represents the
        lexicographic order
        """
        for v, v1 in izip(V, V1):
            if v < v1:
                 return true
            elif v> v1:
                return false
        return false

    def Adj(self, v):
        """Search for all hyperplanes that forms the faces of the polytope"""
        ineq = [(v_i* h_i).list()
                for h_i, v_i in izip(self.hyperplanes, v)]
        l = Polytope(self.V.dimension(),ineq).indexes
        for index in  l:
            yield tuple( c_i if index != j else c_i*(-1)
                    for j, c_i in enumerate(v))


#Class for generating all possible voronoi clusters when the number of clusters is two.

class Cluster2:
    """
    This class is a generator, which gives all possible clustering defined by a voronoi
    region using a tuple t, where t[i] == 1 iff p[i] belongs to cluster 1.
    """

    def __init__(self, points, filename = '2_clusters', processes = 1):
        """
        Constructor
        """
        self.filename = filename
        self.directory = '/tmp'
        ensure_dir(self.directory)
        self.processes = processes
        self.m_file = os.path.join(self.directory, self.filename)
        self.points = copy(points)
        self.d = len(points[0])

    def codes(self):
        """
        This is a generator, which returns all possible clusters
        codified as a tuple t satisfying:
        t[i] == 1 iff points[i] belong to cluster 1
        """

        bound = max(max(self.points))
        points_aux = [[1] + [ 2 * i for i in l]
                      for l in self.points]
        pointR = [bound] + [0] * self.d
        ar = Arrangement(points_aux, pointR, filename = self.filename)
        for l in ar.reverse_search({tuple([1]*len(self.points))}):
            yield l


    def codes_m_launch(self):
        '''
        Function that launchs several processes to generate all
        possible two clusters. It is needed writing to disc first in
        as many separated files as processes all possible two clusterings
        '''
        bound = max(max(self.points))
        points_aux = [[1] + [ 2 * i for i in l]
                      for l in self.points]
        pointR = [bound] + [0] * self.d
        ar =  Arrangement(points_aux, pointR, processes = self.processes)
        for l in ar.reverse_m({tuple([1]*len(self.points))}):
            yield l


from itertools import combinations_with_replacement, combinations

class Cluster(Enumerator):
    """
    This class is a representation of the set of all possible
    clustering of given data
    """

    def __init__(self, points, k, filename = 'k_clusters', processes = 4):
        """
        Constructor
        """
        Enumerator.__init__(self, processes, filename, '/tmp')
        self.points = copy(points)
        self.d = len(points[0])
        self.k = k
        self.n = len(points)

    def find_k_clusters(self):
        """
        This function launch several processes to calculate all
        possible clusters that form part of k-partition
        """
        c2 = Cluster2(self.points, processes = self.processes)
        all_p = {p for p in c2.codes_m_launch()}
        self.all_clusters = set()
        for s in combinations_with_replacement(all_p, self.k -1):
            temp = tuple( all(val > 0 for val in t)
                                  for t in izip(*s))
            self.all_clusters.add(tuple(1 if val else 0 for val in temp) )

    def generate_dicts(self):
        """
        Private function, that it is used to generate the neighbours
        of a clusters, that is:
        Given a set S and a set of sets T, return the sets S1 such
        that S1 contains S and another point OR S contains S and
        another point
        """
        self.find_k_clusters()
        remove, add = dict(), dict()
        for cluster in self.all_clusters:
            list_c = list(cluster)
            add_point, remove_point = set([]), set([])
            for i in xrange(self.n):
                try:
                    t = list_c[i]
                except:
                    print t, self.n, i, list_c
                    raise Exception("NO FUNCIONA")
                c = 1 if t == 0 else 0
                list_c[i] = c
                if tuple(list_c) in self.all_clusters:
                    if t == 0:
                        add_point.add(i)
                    else:
                        remove_point.add(i)
                list_c[i] = t
            add[cluster] = add_point
            remove[cluster] = remove_point
        self.add = add
        self.remove = remove

    def codes(self):
        """
        This is a generator, which returns all possible partitions
        into k clusterings
        """
        if self.k > 2:
            self.generate_dicts()
            for l in self.reverse_search([[tuple([1]*self.n),]]):
                yield l
        else:
            c2 = Cluster2(self.points, processes = self.processes)
            for l in c2.codes():
                cluster1 = tuple(1 if i>0 else 0 for i in l)
                cluster2 = tuple(1 if i<0 else 0 for i in l)
                yield cluster1, cluster2

    def codes_m_launch(self):
        """
        Method that launchs several processes to generate all possible
        partitions into k clusters.
        """
        if self.k > 2:
            self.generate_dicts()
            for l in self.reverse_m([[tuple([1]*self.n),],]):
                yield l
        else:
            c2 = Cluster2(self.points, processes = self.processes)
            for l in c2.codes_m_launch():
                cluster1 = tuple(1 if i>0 else 0 for i in l)
                cluster2 = tuple(1 if i<0 else 0 for i in l)
                yield cluster1, cluster2

    def f(self, v):
        '''
        This comes from the article of Avis and Fukuda to enumerate
        using a local search.
        This function is a local search, f is an algorithm wich
        applied several times to the same node it gives the trivial
        partition, i. e. all the points belong to the same cluster.

        f receives a partition (represented by a list of tuples
        ordered) a return another partition defined by a Voronoi
        diagram where one of the points of the cluster with less
        points has been put in the cluster with more points
        '''
        i = len(v) - 1
        j = i
        if i == 0:
            #TODO: Understand why I need this
            return v
        u = [list(t) for t in v]
        B = True
        while B :
            j -= 1
            
            a, r = self.add[tuple(u[j])], self.remove[tuple(u[i])]
            intersection = a.intersection(r)
            if intersection:
                B  = False
            elif j  ==  0 :
                i -= 1
                j = i
            if not i:
                return v
        pos = min(intersection)
        u[j][pos], u[i][pos] = 1, 0
        if not any(u[i]):
            del u[i]
        result = [tuple(t) for t in u]
        result.sort()
        result.sort(key = sum, reverse = True)
        return result

    def Adj(self, v):
        '''
        This function returns a list of all the adjacents of a
        partition. We say that a partition is adjacent to another iff
        they are only different in one point
        '''
        u = [list(t) for t in v]
        result = [tuple(t) for t in u]
        length = len(u)
        repetitions = set()
        if length < self.k :
            for t in u:
                for r in self.remove[tuple(t)]:
                    result.remove(tuple(t))
                    t[r] = 0
                    temp = [0]*self.n
                    temp[r] = 1
                    result.append(tuple(temp))
                    result.append(tuple(t))
                    result.sort()
                    result.sort(key = sum, reverse = True)
                    if tuple(result) not in repetitions:
                        yield result
                        repetitions.add(tuple(result))
                    result.remove(tuple(t))
                    result.remove(tuple(temp))
                    t[r] = 1
                    result.append(tuple(t))
        if length > 1:
            for t0, t1 in combinations(u, 2):
                inter = self.remove[tuple(t0)]
                inter = inter.intersection(self.add[tuple(t1)])
                for r in inter:
                    result.remove(tuple(t0))
                    result.remove(tuple(t1))
                    t0[r] = 0
                    t1[r] = 1
                    if any(t0):
                        result.append(tuple(t0))
                    result.append(tuple(t1))
                    result.sort()
                    result.sort(key = sum, reverse = True)
                    if tuple(result) not in repetitions:
                        yield result
                        repetitions.add(tuple(result))
                    if any(t0):
                        result.remove(tuple(t0))
                    result.remove(tuple(t1))
                    t0[r] = 1
                    t1[r] = 0
                    result.append(tuple(t0))
                    result.append(tuple(t1))
                # TODO: Eliminate the duplicate code
                inter = self.add[tuple(t0)]
                inter = inter.intersection(self.remove[tuple(t1)])
                for r in inter:
                    result.remove(tuple(t0))
                    result.remove(tuple(t1))
                    t0[r] = 1
                    t1[r] = 0
                    result.append(tuple(t0))
                    if any(t1):
                        result.append(tuple(t1))
                    result.sort()
                    result.sort(key = sum, reverse = True)
                    if tuple(result) not in repetitions:
                        yield result
                        repetitions.add(tuple(result))
                    result.remove(tuple(t0))
                    if any(t1):
                        result.remove(tuple(t1))
                    t0[r] = 0
                    t1[r] = 1
                    result.append(tuple(t0))
                    result.append(tuple(t1))




#Intracluster measure
def intra_cluster(points):
    V = VectorSpace(K, len(points[0]))
    vecs = [V(p) for p in points]
    center = sum(vecs)/len(points)
    result = 0
    for v in vecs:
        result +=  (v-center).norm()**2
    return result

#Intercluster measure

def inter_cluster(data,clusters):
    '''
    return the inter cluster measure of a partition.
    clusters is a list of indicator functions, e. g.
    clusters[i] is a vector of 0 and 1's such that if
    self.data[j] in C_i iff clusters[i][j] == 1.
    To save just a litte of memory, the length of clusters is
    going to be k-1. The points in C_{k-1} are the points that are
    not in any of the other clusters. It is nice to notice that
    the mass centers are important, the intra_cluster measure
    depends heavily in the intra_cluster measure and it is not
    difficult to find examples, where the intracluster measure is
    smaller taking "fake" centroids.
    '''

    partition = []
    dist = 0
    for cluster in clusters:
        part = []
        for p, i in izip (data, cluster):
            if i>0:
                part.append(p)
        if part:
            dist += intra_cluster(part)
        part = []
        for d in izip(data,*clusters):
            if not any(i>0 for i in d[1:]):
                part.append(d[0])
        if part:
            dist += intra_cluster(part)
    return dist

def points_in_line(l, max):
    rest = []
    a, b = l
    if a:
        rest.append([K((b*max-1/2)/a),K(-max)])
        rest.append([K((-b*max-1/2)/a),K(max)])
    if b:
        rest.append([K(-max),K((a*max-1/2)/b)])
        rest.append([K(max),K((-a*max-1/2)/b)])
    for r in rest:
        if all([-max <= i <= max for i in r]):
            yield r
def all_points(data,max):
    result = line2d([[-max,-max],[max,-max],[max,max],[-max,max],[-max,-max]])
    for p in data:
        temp = list(points_in_line(p,max))
        result += line2d(temp)
    return result
ar = Arrangement(data)
result = []
for i in ar.reverse_search({tuple([1]*len(data))}):
    result.append(i)
print("los signos son",result)

︡ff0710aa-88f8-45a7-bb1f-49d77852c9c4︡{"stdout":"[[100000, 200000], [300000, 400000], [500000, -100000], [200000, 200000]]\n"}︡{"stdout":"2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2"}︡{"stdout":" [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n"}︡{"stdout":"('los signos son', [(1, 1, 1, 1), (1, 1, -1, 1), (-1, 1, -1, 1), (-1, -1, -1, 1), (1, 1, 1, -1), (1, -1, 1, -1), (-1, -1, 1, -1), (-1, -1, -1, -1)])\n"}︡{"done":true}
︠687ce177-6bce-4230-9832-570895a1f4cc︠

data
︡fcb518af-3050-4e7d-b095-1fa1104d7e35︡{"stdout":"[[100000, 200000], [300000, 400000], [500000, -100000], [200000, 200000]]\n"}︡{"done":true}
︠277fa5e6-ac92-4116-929b-3c699106d26c︠
ar = Arrangement(data)
result = []
for i in ar.reverse_search({tuple([1]*len(data))}):
    result.append(i)
︡3b982d54-258e-438f-b934-0a403fd3e6dd︡{"stdout":"2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[-1, -2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [5, -1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [5, -1], [1, 1]]\n2 [[-1, -2], [-3, -4], [5, -1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[-1, -2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [-3, -4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n2 [[1, 2], [3, 4], [-5, 1], [-1, -1]]\n2 [[1, 2], [3, 4], [-5, 1], [1, 1]]\n"}︡{"done":true}
︠14e02836-526d-484d-9b87-4d3e12f726ce︠
print(result)
︡6ca554ed-bb4c-4ff8-83f4-7da1a7a7ef34︡{"stdout":"[(1, 1, 1, 1), (1, 1, -1, 1), (-1, 1, -1, 1), (-1, -1, -1, 1), (1, 1, 1, -1), (1, -1, 1, -1), (-1, -1, 1, -1), (-1, -1, -1, -1)]\n"}︡{"done":true}
︠13ef5630-f9d0-4f69-882b-c9495e095a42︠









