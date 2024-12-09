""" PROGRAM FOR TESTING VORNOI CORRECTNESS """
import numpy as np
import matplotlib.pyplot as plt

NUM_POINTS = 100
MAX_VAL = 100000

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
        self.circumcircle = None

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
        filtered_triangles.append(Triangle(edge.v0, edge.v1, vertex)) #FIXME THIS IS WEIRD

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
                triangle.v2 == st.v0 or triangle.v1 == st.v1 or triangle.v2 == st.v2):
            print(triangle.v2.y)
            filtered_triangles.append(triangle)

    return filtered_triangles


if __name__ == '__main__':
    # Generate random points
    points_list = [Vertex(np.random.randint(0, MAX_VAL), np.random.randint(0, MAX_VAL)) for _ in range(0, NUM_POINTS)]

    triangles = bowyerWatson(points_list)

    fig = plt.figure()
    ax = fig.add_subplot()
    
    for triangle in triangles:
        e0 = Edge(triangle.v0, triangle.v1)
        e1 = Edge(triangle.v1, triangle.v2)
        e2 = Edge(triangle.v2, triangle.v0)

        xdata = [e0.v0.x, e0.v1.x]
        ydata = [e0.v0.y, e0.v1.y]
        ax.plot(xdata, ydata)
        ax.scatter(xdata, ydata)

        xdata = [e1.v0.x, e1.v1.x]
        ydata = [e1.v0.y, e1.v1.y]
        ax.plot(xdata, ydata)
        ax.scatter(xdata, ydata)

        xdata = [e2.v0.x, e2.v1.x]
        ydata = [e2.v0.y, e2.v1.y]
        ax.plot(xdata, ydata)
        ax.scatter(xdata, ydata)

    plt.show()
