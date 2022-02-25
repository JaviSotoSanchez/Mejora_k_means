#Class Polytope
K = QQ
PREC = 5
from scipy.cluster.vq import vq, kmeans, kmeans2,vq
from scipy import linalg, matrix
from itertools import izip
from copy import deepcopy as copy
data=[]
#for line in open("linux2.txt"):
    #mylist=[QQ(int(float(i)*100000)) for i in line.strip().split(",")]
    #data.append(mylist)
 
   
#data=data[:100]
#print(data)
data = [[QQ(10),QQ(10)],[QQ(10),QQ(1)],[QQ(10),QQ(2)],[QQ(10),QQ(3)],[QQ(1),QQ(5)],[QQ(3),QQ(2)]]

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
        #print dimension, ineq
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


class Enumerator:
    """
    This class is an abstract class to implement, in the most possible
    general way. The notation follow the one given in the article
    'Reverse search for enumeration' by Avis and Fukuda.
    There is one sutility, though: delta is not necessary for the
    implementation BUT it is necessary to exists.
    """

    def __init__(self):
        """
        Constructor
        processes is the number of
        processes which our enumerator will launch, filename and
        directory are used for bookmarking
        """
        pass


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

    

    
   


    def reverse_search(self, S, heuristic = lambda x: 0, minimo = infinity, coste = lambda s: infinity):
        """
        This is the main algorithm, which is a generator
        function. This means that outputs all the elements without
        keeping them in memory.
        El heuristico contiene la funcion de coste
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
                        if self.f(Next) == v and heuristic(Next) < minimo:
                            minimo = min(minimo, coste(Next))
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

    def __init__(self, hyperplanes, pointR = []):
        """
        Constructor, which takes a list of hyperplanes, representing
        by lists of the same length and a initial point pointR.
        This point can be empty and the class generates it at random.
        """
        Enumerator.__init__(self)
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

    def __init__(self, points):
        """
        Constructor
        """
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
        ar = Arrangement(points_aux, pointR)
        for l in ar.reverse_search({tuple([1]*len(self.points))}):
            yield l
            
    def cost(self, s):
        return inter_cluster(self.points, [s]) 




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

ar = Arrangement(data)
result = []
for i in ar.reverse_search({tuple([1]*len(data))}):
    result.append(i)
print("los signos son",result)