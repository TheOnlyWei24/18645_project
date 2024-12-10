""" PROGRAM FOR TESTING VORNOI CORRECTNESS """
import numpy as np
import matplotlib.pyplot as plt

NUM_POINTS = 16000
MAX_VAL = 500000

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Edge:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1

    def __eq__(self, other):
        return ((self.v0 == other.v0 and self.v1 == other.v1) or 
               (self.v0 == other.v1 and self.v1 == other.v0))


class Triangle:
    def __init__(self, v0, v1, v2):
        if self.isCounterClockwise(v0, v1, v2):
            self.v0 = v0
            self.v1 = v1
            self.v2 = v2
        else:
            self.v0 = v0
            self.v1 = v2
            self.v2 = v1
        self.e0 = Edge(self.v0, self.v1)
        self.e1 = Edge(self.v1, self.v2)
        self.e2 = Edge(self.v2, self.v0)
        self.neighbors = []
        self.circumcenter = None
        self.vornoi_e0 = None
        self.vornoi_e1 = None
        self.vornoi_e2 = None

    def inCircumcircle(self, vertex):
        matrix = np.array([[self.v0.x, self.v0.y, self.v0.x**2 + self.v0.y**2, 1],
                           [self.v1.x, self.v1.y, self.v1.x**2 + self.v1.y**2, 1],
                           [self.v2.x, self.v2.y, self.v2.x**2 + self.v2.y**2, 1],
                           [vertex.x, vertex.y, vertex.x**2 + vertex.y**2, 1]])
        return np.linalg.det(matrix) > 0

    @staticmethod
    def isCounterClockwise(v0, v1, v2):
        matrix = np.array([[v0.x, v0.y, 1],
                           [v1.x, v1.y, 1],
                           [v2.x, v2.y, 1]])
        return np.linalg.det(matrix) > 0

    def isNeighbor(self, other):
        return (self.e0 == other.e0 or self.e0 == other.e1 or self.e0 == other.e2 or
                self.e1 == other.e0 or self.e1 == other.e1 or self.e1 == other.e2 or
                self.e2 == other.e0 or self.e2 == other.e1 or self.e2 == other.e2)

    #FIXME: Call kernels here
    def calculateCircumcenter(self):
        D = 2 * ((self.v0.x * (self.v1.y - self.v2.y)) +
                 (self.v1.x * (self.v2.y - self.v0.y)) +
                 (self.v2.x * (self.v0.y - self.v1.y)))
        assert D != 0.0, "Points are colinear"
        Ux = (((self.v0.x**2 + self.v0.y**2) * (self.v1.y - self.v2.y)) + 
              ((self.v1.x**2 + self.v1.y**2) * (self.v2.y - self.v0.y)) +
              ((self.v2.x**2 + self.v2.y**2) * (self.v0.y - self.v1.y))) / D
        Uy = (((self.v0.x**2 + self.v0.y**2) * (self.v2.x - self.v1.x)) +
              ((self.v1.x**2 + self.v1.y**2) * (self.v0.x - self.v2.x)) +
              ((self.v2.x**2 + self.v2.y**2) * (self.v1.x - self.v0.x))) / D
        self.circumcenter = Vertex(Ux, Uy)


def superTriangle(points_list):
    # minX = int(100000); minY = int(100000)
    # maxX = int(-100000); maxY = int(-100000)

    # for vertex in points_list:
    #     minX = min(minX, vertex.x)
    #     minY = min(minY, vertex.y)
    #     maxX = max(maxX, vertex.x)
    #     maxY = max(maxY, vertex.y)

    # dx = (maxX - minX) * 10
    # dy = (maxY - minY) * 10

    # v0 = Vertex(minX - dx, minY - dy * 3)
    # v1 = Vertex(minX - dx, maxY + dy)
    # v2 = Vertex(maxX + dx * 3, maxY + dy)
    # return Triangle(v0, v1, v2)
    v0 = Vertex(-100000000, 0)
    v1 = Vertex(0, 100000000)
    v2 = Vertex(100000000, 0)
    return Triangle(v0, v1, v2)


def addVertex(vertex, triangles):
    edges = []
    filtered_triangles = []

    # Remove triangles with circumcircles containing the vertex
    for triangle in triangles:
        if triangle.inCircumcircle(vertex):
            edges.append(Edge(triangle.v0, triangle.v1))
            edges.append(Edge(triangle.v1, triangle.v2))
            edges.append(Edge(triangle.v2, triangle.v0))
        else:
            filtered_triangles.append(triangle)

    # Get unique edges
    unique_edges = []
    for i in range(0, len(edges)):
        valid = True
        edge = edges[i]

        for j in range(0, len(edges)):
            if i != j and edge == edges[j]:
                valid = False
                break

        if valid:
            unique_edges.append(edge)

    # Create new triangles from the unique edges and new vertex
    for edge in unique_edges:
        filtered_triangles.append(Triangle(edge.v0, edge.v1, vertex)) 

    return filtered_triangles
        

# https://www.gorillasun.de/blog/bowyer-watson-algorithm-for-delaunay-triangulation/#the-super-triangle
def bowyerWatson(points_list):
    triangles = []

    # add super triangle
    st = superTriangle(points_list)
    triangles.append(st)

    # Triangulate each vertex
    for vertex in points_list:
        triangles = addVertex(vertex, triangles)
        
    # done inserting points, clean up (remove triangles that share edges with super triangle)
    filtered_triangles = []
    for triangle in triangles:
        if not (triangle.v0 == st.v0 or triangle.v0 == st.v1 or triangle.v0 == st.v2 or
                triangle.v1 == st.v0 or triangle.v1 == st.v1 or triangle.v1 == st.v2 or
                triangle.v2 == st.v0 or triangle.v2 == st.v1 or triangle.v2 == st.v2):
            filtered_triangles.append(triangle)

    return filtered_triangles


def print_delaunay(triangles, ax):
    for triangle in triangles:
        e0 = Edge(triangle.v0, triangle.v1)
        e1 = Edge(triangle.v1, triangle.v2)
        e2 = Edge(triangle.v2, triangle.v0)

        xdata = [e0.v0.x, e0.v1.x]
        ydata = [e0.v0.y, e0.v1.y]
        ax.plot(xdata, ydata, color="black")
        ax.scatter(xdata, ydata, color="black")

        xdata = [e1.v0.x, e1.v1.x]
        ydata = [e1.v0.y, e1.v1.y]
        ax.plot(xdata, ydata, color="black")
        ax.scatter(xdata, ydata, color="black")

        xdata = [e2.v0.x, e2.v1.x]
        ydata = [e2.v0.y, e2.v1.y]
        ax.plot(xdata, ydata, color="black")
        ax.scatter(xdata, ydata, color="black")


def vornoi(triangles):
    # Calculate circumcenters
    for triangle in triangles:
        triangle.calculateCircumcenter()
    
    # Get triangle neighbors
    for i in range(0, len(triangles)):
        for j in range(0, len(triangles)):
            if i != j and triangles[i].isNeighbor(triangles[j]):
                triangles[i].neighbors.append(triangles[j])

    # Get vornoi diagram
    for triangle in triangles:
        num_neighbors = len(triangle.neighbors)
        assert num_neighbors <= 3, "More than 3 neighbors for a triangle"
        if (1 <= len(triangle.neighbors)):
            triangle.vornoi_e0 = Edge(triangle.circumcenter, triangle.neighbors[0].circumcenter)
        if (2 <= len(triangle.neighbors)):
            triangle.vornoi_e1 = Edge(triangle.circumcenter, triangle.neighbors[1].circumcenter)
        if (3 <= len(triangle.neighbors)):
            triangle.vornoi_e2 = Edge(triangle.circumcenter, triangle.neighbors[2].circumcenter)


def print_vornoi(triangles, ax):
    #FIXME: THIS IS INEFFICEINT, there are duplicates of the vornoi edges    
    for triangle in triangles:
        if triangle.vornoi_e0:
            xdata = [triangle.vornoi_e0.v0.x, triangle.vornoi_e0.v1.x]
            ydata = [triangle.vornoi_e0.v0.y, triangle.vornoi_e0.v1.y]
            ax.plot(xdata, ydata, color="red")
            ax.scatter(xdata, ydata, color="red")

        if triangle.vornoi_e1:
            xdata = [triangle.vornoi_e1.v0.x, triangle.vornoi_e1.v1.x]
            ydata = [triangle.vornoi_e1.v0.y, triangle.vornoi_e1.v1.y]
            ax.plot(xdata, ydata, color="red")
            ax.scatter(xdata, ydata, color="red")

        if triangle.vornoi_e2:
            xdata = [triangle.vornoi_e2.v0.x, triangle.vornoi_e2.v1.x]
            ydata = [triangle.vornoi_e2.v0.y, triangle.vornoi_e2.v1.y]
            ax.plot(xdata, ydata, color="red")
            ax.scatter(xdata, ydata, color="red")


def output_points_list(points_list):
    with open("points.txt", 'w') as file:
        for vertex in points_list:
            file.write(f"{vertex.x} {vertex.y}\n")


def input_triangles_list():
    triangles = []
    with open("triangles.txt", 'r') as file:
            for line in file:
                coordinates = list(map(float, line.split()))
                triangle = Triangle(Vertex(coordinates[0], coordinates[1]),
                                    Vertex(coordinates[2], coordinates[3]),
                                    Vertex(coordinates[4], coordinates[5]))
                triangles.append(triangle)
    return triangles


if __name__ == '__main__':
    # Generate random points
    points_list = [Vertex(np.random.randint(0, MAX_VAL), np.random.randint(0, MAX_VAL)) for _ in range(0, NUM_POINTS)]

    output_points_list(points_list)

    triangles = bowyerWatson(points_list)

    print(len(triangles))

    #fig = plt.figure()
    #ax = fig.add_subplot()

    #triangles = input_triangles_list()

    #print_delaunay(triangles, ax)

    #vornoi(triangles)

    # print_vornoi(triangles, ax)

    #plt.show()

