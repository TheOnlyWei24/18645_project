#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <immintrin.h> 
#include "kernels/det3/kernel.h"

#define NUM_POINTS 16

// Vertex class 
struct Vertex {
    float x, y;
    Vertex() : x(0), y(0) {}
    Vertex(float x_, float y_) : x(x_), y(y_) {}
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
    float circumradius;

    Triangle(const Vertex& v0_, const Vertex& v1_, const Vertex& v2_)
        : v0(v0_), v1(v1_), v2(v2_) {
        // Check if the vertices are in counter-clockwise order
        if (!isCounterClockwise(v0, v1, v2)) {
            std::swap(v1, v2);
        }
        calculateCircumcircle();
    }

    bool isCounterClockwise(const Vertex& a, const Vertex& b, const Vertex& c) const {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
    }

    void calculateCircumcircle() {
        float ax = v0.x, ay = v0.y;
        float bx = v1.x, by = v1.y;
        float cx = v2.x, cy = v2.y;

        float d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if (d == 0) {
            circumcenter = Vertex(0, 0);
            circumradius = 0;
            return;
        }

        float ux = ((ax * ax + ay * ay) * (by - cy) +
                    (bx * bx + by * by) * (cy - ay) +
                    (cx * cx + cy * cy) * (ay - by)) / d;
        float uy = ((ax * ax + ay * ay) * (cx - bx) +
                    (bx * bx + by * by) * (ax - cx) +
                    (cx * cx + cy * cy) * (bx - ax)) / d;
        circumcenter = Vertex(ux, uy);
        circumradius = std::sqrt((ux - ax) * (ux - ax) + (uy - ay) * (uy - ay));
    }

    bool inCircumcircle(const Vertex& v) const {
        float dist = std::sqrt((v.x - circumcenter.x) * (v.x - circumcenter.x) +
                               (v.y - circumcenter.y) * (v.y - circumcenter.y));
        return dist <= circumradius;
    }

    // Find the area of the triangle
    float area() const {
        return 0.5f * std::fabs(v0.x * (v1.y - v2.y) + v1.x * (v2.y - v0.y) + v2.x * (v0.y - v1.y));
    }

    // Check if the triangle is degenerate
    bool isDegenerate(float tolerance = 1e-6f) const {
        return area() < tolerance;
    }
};

// To remove duplicate edges and keep only boundary edges
std::vector<Edge> getBoundaryEdges(const std::vector<Edge>& edges) {
    std::vector<Edge> boundaryEdges;
    for (const auto& edge : edges) {
        auto it = std::find(boundaryEdges.begin(), boundaryEdges.end(), edge);
        if (it != boundaryEdges.end()) {
            boundaryEdges.erase(it);
        } else {
            boundaryEdges.push_back(edge);
        }
    }
    return boundaryEdges;
}

// Create a super-triangle with all vertices
Triangle createSuperTriangle(const std::vector<Vertex>& vertices) {
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();

    for (const auto& vertex : vertices) {
        minX = std::min(minX, vertex.x);
        minY = std::min(minY, vertex.y);
        maxX = std::max(maxX, vertex.x);
        maxY = std::max(maxY, vertex.y);
    }

    float dx = (maxX - minX) * 2;
    float dy = (maxY - minY) * 2;

    Vertex v0(minX - dx, minY - dy * 3);
    Vertex v1(minX - dx, maxY + dy);
    Vertex v2(maxX + dx * 3, maxY + dy);
    return Triangle(v0, v1, v2);
}

struct PackedPoints {
    float Ax[NUM_POINTS];
    float Ay[NUM_POINTS];
    float Bx[NUM_POINTS];
    float By[NUM_POINTS];
    float Cx[NUM_POINTS];
    float Cy[NUM_POINTS];
    float Dx[NUM_POINTS];
    float Dy[NUM_POINTS];
};

// Function to print PackedPoints
void printPackedData(const PackedPoints& packedData, size_t numVertices) {
    std::cout << "Packed Data:" << std::endl;
    for (size_t i = 0; i < numVertices; ++i) {
        std::cout << "Index " << i << ": "
                  << "Ax: " << packedData.Ax[i] << ", Ay: " << packedData.Ay[i]
                  << ", Bx: " << packedData.Bx[i] << ", By: " << packedData.By[i]
                  << ", Cx: " << packedData.Cx[i] << ", Cy: " << packedData.Cy[i]
                  << ", Dx: " << packedData.Dx[i] << ", Dy: " << packedData.Dy[i]
                  << std::endl;
    }
}

// Pack all vertices aka Dx, Dy
void packAllVertices(const std::vector<Vertex>& vertices, PackedPoints& packedData) {
    for (size_t i = 0; i < vertices.size(); ++i) {
        packedData.Dx[i] = vertices[i].x;
        packedData.Dy[i] = vertices[i].y;
    }
}

// Main loop for Delaunay triangulation
std::vector<Triangle> bowyerWatson(const std::vector<Vertex>& vertices) {
    std::vector<Triangle> triangles;
    Triangle superTriangle = createSuperTriangle(vertices);
    triangles.push_back(superTriangle);

    // Pack all vertices
    PackedPoints packedData;
    packAllVertices(vertices, packedData);

    for (const auto& vertex : vertices) {
        std::vector<Edge> edges;

        for (auto it = triangles.begin(); it != triangles.end();) {
            // Set the current triangle vertices as Ax, Ay, Bx, By, Cx, Cy
            std::fill(std::begin(packedData.Ax), std::end(packedData.Ax), it->v0.x);
            std::fill(std::begin(packedData.Ay), std::end(packedData.Ay), it->v0.y);
            std::fill(std::begin(packedData.Bx), std::end(packedData.Bx), it->v1.x);
            std::fill(std::begin(packedData.By), std::end(packedData.By), it->v1.y);
            std::fill(std::begin(packedData.Cx), std::end(packedData.Cx), it->v2.x);
            std::fill(std::begin(packedData.Cy), std::end(packedData.Cy), it->v2.y);

            // Print packed data
            // printPackedData(packedData, vertices.size());

            float det3_out[NUM_POINTS];
            kernel(packedData.Ax, packedData.Ay, packedData.Bx, packedData.By,
                   packedData.Cx, packedData.Cy, packedData.Dx, packedData.Dy, det3_out);

            bool inCircle = false;
            for (size_t i = 0; i < vertices.size(); ++i) {
                
                // Print the triangle and corresponding point details
                /*
                std::cout << "Triangle: (" << it->v0.x << ", " << it->v0.y << ") -> ("
                          << it->v1.x << ", " << it->v1.y << ") -> ("
                          << it->v2.x << ", " << it->v2.y << ") \t ";
                std::cout << "Point: (" << packedData.Dx[i] << ", " << packedData.Dy[i] << ") \t ";
                std::cout << "it->inCircumcircle(vertex): " << it->inCircumcircle(vertices[i]) << "\t";
                std::cout << "Determinant: " << det3_out[i] << "\n";
                */

                if (det3_out[i] > 0) {
                    edges.push_back(Edge(it->v0, it->v1));
                    edges.push_back(Edge(it->v1, it->v2));
                    edges.push_back(Edge(it->v2, it->v0));
                    it = triangles.erase(it);
                    inCircle = true;
                    break;
                }
            }

            if (!inCircle) {
                ++it;
            }
        }

        edges = getBoundaryEdges(edges);

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
        Vertex(0, 0), Vertex(1, 2), Vertex(2, 4), Vertex(3, 1),
        Vertex(4, 3), Vertex(5, 5), Vertex(6, 2), Vertex(7, 4),
        Vertex(1, 5), Vertex(2, 6), Vertex(3, 7), Vertex(4, 6),
        Vertex(5, 1), Vertex(6, 0), Vertex(7, 3), Vertex(8, 5)
    };


    std::vector<Triangle> triangulation = bowyerWatson(vertices);

    for (const auto& triangle : triangulation) {
        std::cout << "Triangle: (" << triangle.v0.x << ", " << triangle.v0.y << ") -> ("
                  << triangle.v1.x << ", " << triangle.v1.y << ") -> ("
                  << triangle.v2.x << ", " << triangle.v2.y << ")\n";
    }

    return 0;
}