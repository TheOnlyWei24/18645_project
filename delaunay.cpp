#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cassert>
#include <immintrin.h> 
#include "kernels/det3/kernel.h"

#define NUM_TRIANGLES 2048 // NOTE: NEEDS TO MULTIPLE OF KERNEL_SIZE AS OF RN
#define NUM_ELEMS 256
#define ALIGNMENT 32
#define SIMD_SIZE 8

#define NUM_SIMD_IN_KERNEL (NUM_TRIANGLES / SIMD_SIZE)

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
};

// Comparator for Edge type
struct CompareEdges {
    bool operator()(const Edge& e1, const Edge& e2) const {
        if (e1.v0.x != e2.v0.x) return e1.v0.x < e2.v0.x;
        if (e1.v0.y != e2.v0.y) return e1.v0.y < e2.v0.y;
        if (e1.v1.x != e2.v1.x) return e1.v1.x < e2.v1.x;
        return e1.v1.y < e2.v1.y;
    }
};

// To remove duplicate edges and keep only boundary edges -- TODO: IS THIS CORRECT? OR SHOULD I JUST DO: VECTOR -> SET -> VECTOR?
std::vector<Edge> getBoundaryEdges(const std::vector<Edge>& edges) {
    // Use a map to count the occurrences of each edge
    std::map<Edge, int, CompareEdges> edgeCount;

    for (const auto& edge : edges) {
        edgeCount[edge]++;
    }

    // Collect edges that appear only once -- boundary edges ??
    std::vector<Edge> boundaryEdges;
    for (const auto& [edge, count] : edgeCount) {
        if (count == 1) {
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

    float dx = (maxX - minX) * 2;       // TODO: SOME OTHER VALUE?
    float dy = (maxY - minY) * 2;       // TODO: SOME OTHER VALUE?

    Vertex v0(minX - dx, minY - dy * 3);
    Vertex v1(minX - dx, maxY + dy);
    Vertex v2(maxX + dx * 3, maxY + dy);
    return Triangle(v0, v1, v2);
}

// PackedPoints struct for SIMD
struct PackedPoints {
    float Ax[NUM_ELEMS];
    float Ay[NUM_ELEMS];
    float Bx[NUM_ELEMS];
    float By[NUM_ELEMS];
    float Cx[NUM_ELEMS];
    float Cy[NUM_ELEMS];
    float Dx[NUM_ELEMS];
    float Dy[NUM_ELEMS];
};

// Function to pack triangles and one vertex for SIMD
void packTrianglesAndVertex(const std::vector<Triangle>& triangles, const Vertex& vertex, PackedPoints* packedData) {
    size_t numTriangles = triangles.size();
    size_t numSimdGroups = (numTriangles + SIMD_SIZE - 1) / SIMD_SIZE; // Correct rounding

    for (size_t group = 0; group < numSimdGroups; ++group) {
        for (size_t j = 0; j < SIMD_SIZE; ++j) {
            size_t index = group * SIMD_SIZE + j;
            if (index < triangles.size()) {
                packedData->Ax[j] = triangles[index].v0.x;
                packedData->Ay[j] = triangles[index].v0.y;
                packedData->Bx[j] = triangles[index].v1.x;
                packedData->By[j] = triangles[index].v1.y;
                packedData->Cx[j] = triangles[index].v2.x;
                packedData->Cy[j] = triangles[index].v2.y;
                packedData->Dx[j] = vertex.x;
                packedData->Dy[j] = vertex.y;
            } else {
                packedData->Ax[j] = 0.0f;
                packedData->Ay[j] = 0.0f;
                packedData->Bx[j] = 0.0f;
                packedData->By[j] = 0.0f;
                packedData->Cx[j] = 0.0f;
                packedData->Cy[j] = 0.0f;
                packedData->Dx[j] = 0.0f;
                packedData->Dy[j] = 0.0f;
            }
        }
    }
}

// Main loop for Delaunay triangulation
std::vector<Triangle> bowyerWatson(const std::vector<Vertex>& vertices, PackedPoints* packedData, float* det3_out) {
    std::vector<Triangle> triangles;
    Triangle superTriangle = createSuperTriangle(vertices);
    triangles.push_back(superTriangle);

    for (const auto& vertex : vertices) {
        std::vector<Edge> edges;
        packTrianglesAndVertex(triangles, vertex, packedData);

        size_t numTriangles = triangles.size();
        size_t numSimdGroups = (numTriangles + SIMD_SIZE - 1) / SIMD_SIZE; // Round up to nearest 16

        // Initialize the allocated memory
        memset(det3_out, 0, sizeof(float) * NUM_ELEMS);

        kernel(packedData->Ax, packedData->Ay,
                packedData->Bx, packedData->By,
                packedData->Cx, packedData->Cy,
                packedData->Dx, packedData->Dy,
                det3_out, numSimdGroups * SIMD_SIZE
        );
        
        for (size_t i = 0; i < numSimdGroups; i++) {
            for (size_t j = 0; j < SIMD_SIZE; j++) {
                size_t index = i * SIMD_SIZE + j;
                // std::cerr << "det3_out[" << i << "][" << j << "]: " << det3_out[index] << std::endl;
            }
        }

        std::cerr << "numSimdGroups: " << numSimdGroups << ", numTriangles: " << numTriangles << "\n" << std::endl;

        for (auto it = triangles.begin(); it != triangles.end();) {
            // Use indices or references instead of pointers
            size_t index = std::distance(triangles.begin(), it);

            size_t group = index / SIMD_SIZE;
            size_t offset = index % SIMD_SIZE;

            // Avoid using invalidated pointers after reallocation
            if (group < numSimdGroups && det3_out[group * SIMD_SIZE + offset] > 0) {
                edges.push_back(Edge(it->v0, it->v1));
                edges.push_back(Edge(it->v1, it->v2));
                edges.push_back(Edge(it->v2, it->v0));
                it = triangles.erase(it); // Safe as no other iterator depends on `it`
            } else {
                ++it;
            }
        }


        edges = getBoundaryEdges(edges);
        for (const auto& edge : edges) {
            triangles.emplace_back(edge.v0, edge.v1, vertex);
        }
    }
    
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

    PackedPoints* packedData = nullptr;
    if (posix_memalign((void**)&packedData, ALIGNMENT, sizeof(PackedPoints)) != 0) {
        std::cerr << "Memory allocation failed for packedData" << std::endl;
        return -1;
    }
    if (!packedData) {
        std::cerr << "packedData is null after allocation" << std::endl;
        return -1;
    }
    memset(packedData, 0, sizeof(PackedPoints));

    float* det3_out;
    if (posix_memalign((void**)&det3_out, ALIGNMENT, sizeof(float) * NUM_ELEMS) != 0) {
        std::cerr << "Memory allocation failed for det3_out" << std::endl;
        return -1;
    }

    std::vector<Triangle> triangulation = bowyerWatson(vertices, packedData, det3_out);

    for (const auto& triangle : triangulation) {
        std::cout << "Triangle: (" << triangle.v0.x << ", " << triangle.v0.y << ") -> ("
                  << triangle.v1.x << ", " << triangle.v1.y << ") -> ("
                  << triangle.v2.x << ", " << triangle.v2.y << "),\t" << "area (non-linearity check): " << triangle.area() << "\n";
    }

    // Free the allocated memory
    free(packedData);

    return 0;
}