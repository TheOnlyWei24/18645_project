#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// Vertex class
struct Vertex {
    double x, y;
    Vertex() : x(0), y(0) {}
    Vertex(double x_, double y_) : x(x_), y(y_) {}
    bool operator==(const Vertex& v) const {
        return x == v.x && y == v.y;
    }
};

// Edge class
struct Edge {
    Vertex v0, v1;
    Edge(const Vertex& v0_, const Vertex& v1_) : v0(v0_), v1(v1_) {}
    bool operator==(const Edge& e) const {
        return (v0 == e.v0 && v1 == e.v1) || (v0 == e.v1 && v1 == e.v0);
    }
};

// Triangle class
struct Triangle {
    Vertex v0, v1, v2;
    Vertex circumcenter;
    double circumradius;

    Triangle(const Vertex& v0_, const Vertex& v1_, const Vertex& v2_)
        : v0(v0_), v1(v1_), v2(v2_) {
        calculateCircumcircle();
    }

    void calculateCircumcircle() {
        double ax = v0.x, ay = v0.y;
        double bx = v1.x, by = v1.y;
        double cx = v2.x, cy = v2.y;

        double d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if (d == 0) {  // Degenerate triangle check
            circumcenter = Vertex(0, 0);
            circumradius = 0;
            return;
        }

        double ux = ((ax * ax + ay * ay) * (by - cy) +
                     (bx * bx + by * by) * (cy - ay) +
                     (cx * cx + cy * cy) * (ay - by)) / d;
        double uy = ((ax * ax + ay * ay) * (cx - bx) +
                     (bx * bx + by * by) * (ax - cx) +
                     (cx * cx + cy * cy) * (bx - ax)) / d;
        circumcenter = Vertex(ux, uy);
        circumradius = std::sqrt((ux - ax) * (ux - ax) + (uy - ay) * (uy - ay));
    }

    bool inCircumcircle(const Vertex& v) const {
        double dist = std::sqrt((v.x - circumcenter.x) * (v.x - circumcenter.x) +
                                (v.y - circumcenter.y) * (v.y - circumcenter.y));
        return dist <= circumradius;
    }
};

// To remove duplicate edges and keep only boundary edges
std::vector<Edge> getBoundaryEdges(const std::vector<Edge>& edges) {
    std::vector<Edge> boundaryEdges;
    for (const auto& edge : edges) {
        auto it = std::find(boundaryEdges.begin(), boundaryEdges.end(), edge);
        if (it != boundaryEdges.end()) {
            // If edge is already in list, remove it -- because it is shared
            boundaryEdges.erase(it);
        } else {
            // Else add edge to list
            boundaryEdges.push_back(edge);
        }
    }
    return boundaryEdges;
}

// Create a super-triangle with all vertices
Triangle createSuperTriangle(const std::vector<Vertex>& vertices) {
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    for (const auto& vertex : vertices) {
        if (vertex.x < minX) minX = vertex.x;
        if (vertex.y < minY) minY = vertex.y;
        if (vertex.x > maxX) maxX = vertex.x;
        if (vertex.y > maxY) maxY = vertex.y;
    }

    double dx = (maxX - minX) * 10;
    double dy = (maxY - minY) * 10;

    Vertex v0(minX - dx, minY - dy * 3);
    Vertex v1(minX - dx, maxY + dy);
    Vertex v2(maxX + dx * 3, maxY + dy);
    return Triangle(v0, v1, v2);
}

// Bowyer-Watson algorithm for Delaunay triangulation
std::vector<Triangle> bowyerWatson(const std::vector<Vertex>& vertices) {
    std::vector<Triangle> triangles;
    Triangle superTriangle = createSuperTriangle(vertices);
    triangles.push_back(superTriangle);

    for (const auto& vertex : vertices) {
        std::vector<Edge> edges;
        // Find all triangles whose circumcircle contains the new vertex
        for (auto it = triangles.begin(); it != triangles.end();) {
            if (it->inCircumcircle(vertex)) {
                edges.push_back(Edge(it->v0, it->v1));
                edges.push_back(Edge(it->v1, it->v2));
                edges.push_back(Edge(it->v2, it->v0));
                it = triangles.erase(it);
            } else {
                ++it;
            }
        }

        // Remove duplicate edges and keep only boundary edges
        edges = getBoundaryEdges(edges);

        // Re-triangulate the polygonal hole using the new vertex
        for (const auto& edge : edges) {
            triangles.emplace_back(edge.v0, edge.v1, vertex);
        }
    }

    // Remove triangles that share vertices with the super-triangle
    triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
                                   [&superTriangle](const Triangle& triangle) {
                                       return (triangle.v0 == superTriangle.v0 ||
                                               triangle.v0 == superTriangle.v1 ||
                                               triangle.v0 == superTriangle.v2 ||
                                               triangle.v1 == superTriangle.v0 ||
                                               triangle.v1 == superTriangle.v1 ||
                                               triangle.v1 == superTriangle.v2 ||
                                               triangle.v2 == superTriangle.v0 ||
                                               triangle.v2 == superTriangle.v1 ||
                                               triangle.v2 == superTriangle.v2);
                                   }),
                    triangles.end());

    return triangles;
}

int main() {
    std::vector<Vertex> vertices = {
        Vertex(0, 0), Vertex(1, 0), Vertex(0, 1), Vertex(1, 1)
    };

    std::vector<Triangle> triangulation = bowyerWatson(vertices);

    for (const auto& triangle : triangulation) {
        std::cout << "Triangle: (" << triangle.v0.x << ", " << triangle.v0.y << ") -> ("
                  << triangle.v1.x << ", " << triangle.v1.y << ") -> ("
                  << triangle.v2.x << ", " << triangle.v2.y << ")\n";
    }

    return 0;
}
