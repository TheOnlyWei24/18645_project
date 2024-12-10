#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <chrono>
#include <algorithm>
//#include "kernels/det3/kernel.h"

#define SUPER_TRIANGLE_MAX 10000000

#define NUM_TRIANGLES 2048

#define NUM_ELEMS 256

#define ALIGNMENT 32

#define SIMD_SIZE 8

#define NUM_SIMD_IN_KERNEL (NUM_TRIANGLES / SIMD_SIZE)

struct Vertex {
    float x, y;

    Vertex() : x(0), y(0) {}
    Vertex(float x_, float y_) : x(x_), y(y_) {}

    bool operator==(const Vertex& v) const {
        return x == v.x && y == v.y;
    }
};


struct Edge {
    Vertex v0, v1;

    Edge(const Vertex& v0_, const Vertex& v1_) : v0(v0_), v1(v1_) {}

    bool operator==(const Edge& e) const {
        return (v0 == e.v0 && v1 == e.v1) || (v0 == e.v1 && v1 == e.v0);
    }
};


struct Triangle {
    Vertex v0, v1, v2;
    Vertex circumcenter;

    Triangle(const Vertex& v0_, const Vertex& v1_, const Vertex& v2_)
        : v0(v0_), v1(v1_), v2(v2_) {
        // Check if the vertices are in counter-clockwise order
        if (!isCounterClockwise(v0, v1, v2)) {
            std::swap(v1, v2);
        }
    }

    bool isCounterClockwise(const Vertex& a, const Vertex& b, const Vertex& c) const {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
    }

    bool inCircumcircle(const Vertex& v) const {
        float a = v0.x - v.x;
        float b = v0.y - v.y;
        float d = v1.x - v.x;
        float e = v1.y - v.y;
        float g = v2.x - v.x;
        float h = v2.y - v.y;

        float c = (a * a) + (b * b);
        float f = (d * d) + (e * e);
        float i = (g * g) + (h * h);

        float out0 = a * ((e * i) - (f * h));
        float out1 = b * ((f * g) - (d * i));
        float out2 = c * ((d * h) - (e * g));

        float out = out0 + out1 + out2;

        return out > 0;
    }

    void calculateCircumcenter() {
        float Ax2_Ay2 = (v0.x * v0.x) + (v0.y * v0.y);
        float Bx2_Bx2 = (v1.x * v1.x) + (v1.y * v1.y);
        float Cx2_Cx2 = (v2.x * v2.x) + (v2.y * v2.y);

        float D = 2 * (((v1.y - v2.y) * v0.x) + ((v2.y - v0.y) * v1.x) + ((v0.y - v1.y) * v2.x));
        float Ux = ((Ax2_Ay2 * (v1.y - v2.y)) + (Bx2_Bx2 * (v2.y - v0.y)) + (Cx2_Cx2 * (v0.y - v1.y))) / D;
        float Uy = ((Ax2_Ay2 * (v2.x - v1.x)) + (Bx2_Bx2 * (v0.x - v2.x)) + (Cx2_Cx2 * (v1.x - v0.x))) / D;

        circumcenter = Vertex(Ux, Uy);
    }
};


struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
        std::size_t h1 = std::hash<float>()(edge.v0.x) ^ std::hash<float>()(edge.v0.y);
        std::size_t h2 = std::hash<float>()(edge.v1.x) ^ std::hash<float>()(edge.v1.y);
        return h1 ^ h2;
    }
};


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


int readPointsFile(std::string& filename, std::vector<Vertex>& points) {
    std::ifstream file(filename);
    assert(file.is_open());

    float x, y;
    while (file >> x >> y) {
        points.push_back(Vertex(x, y));
    }

    return points.size();
}


void writeTrianglesFile(const std::string& filename, const std::vector<Triangle>& triangles) {
    std::ofstream file(filename);
    assert(file.is_open());

    for (const Triangle& triangle : triangles) {
        file << triangle.v0.x << " " << triangle.v0.y << " "
             << triangle.v1.x << " " << triangle.v1.y << " "
             << triangle.v2.x << " " << triangle.v2.y << std::endl;
    }
}


Triangle superTriangle(void) {
    Vertex v0(-SUPER_TRIANGLE_MAX, 0);
    Vertex v1(0, SUPER_TRIANGLE_MAX);
    Vertex v2(SUPER_TRIANGLE_MAX, 0);
    return Triangle(v0, v1, v2);
}


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


std::vector<Triangle> addVertex(Vertex& vertex, std::vector<Triangle>& triangles) {
    // std::unordered_set<Edge, EdgeHash> unique_edges;
    std::vector<Edge> edges;
    std::vector<Triangle> filtered_triangles;

    //TODO: use unordered list for better perf, hashing isnt working??

    // Remove triangles with circumcircles containing the vertex
    for (Triangle& triangle : triangles) {
        if (triangle.inCircumcircle(vertex)) {
            // unique_edges.insert(Edge(triangle.v0, triangle.v1));
            // unique_edges.insert(Edge(triangle.v1, triangle.v2));
            // unique_edges.insert(Edge(triangle.v2, triangle.v0)); 
            edges.emplace_back(Edge(triangle.v0, triangle.v1));
            edges.emplace_back(Edge(triangle.v1, triangle.v2));
            edges.emplace_back(Edge(triangle.v2, triangle.v0));
        } else {
          filtered_triangles.emplace_back(triangle);
        }
    }

    // Get unique edges
    std::vector<Edge> unique_edges;
    for (int i = 0; i < edges.size(); i++) {
        bool valid = true;
        for (int j = 0; j < edges.size(); j++) {
            if (i != j && edges[i] == edges[j]) {
                valid = false;
                break;
            }
        }

        if (valid) {
            unique_edges.emplace_back(edges[i]);
        }
    }

    // Create new triangles from the unique edges and new vertex
    for (Edge edge : unique_edges) {
        filtered_triangles.emplace_back(Triangle(edge.v0, edge.v1, vertex));
    }
    
    return filtered_triangles;
}


std::vector<Triangle> bowyerWatson(std::vector<Vertex>& points) {
    // Add super triangle
    std::vector<Triangle> triangles;
    Triangle st = superTriangle();
    triangles.push_back(st);

    // Triangulate each vertex
    for (Vertex& vertex : points) {
        triangles = addVertex(vertex, triangles);
    }

    // Remove triangles that share verticies with super triangle
    triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
                                   [&st](const Triangle& triangle) {
                                       return (triangle.v0 == st.v0 ||
                                               triangle.v0 == st.v1 ||
                                               triangle.v0 == st.v2 ||
                                               triangle.v1 == st.v0 ||
                                               triangle.v1 == st.v1 ||
                                               triangle.v1 == st.v2 ||
                                               triangle.v2 == st.v0 ||
                                               triangle.v2 == st.v1 ||
                                               triangle.v2 == st.v2);}), triangles.end());

    return triangles;
}


void vornoi(std::vector<Triangle>& triangles) {
    for (Triangle triangle : triangles) {
        triangle.calculateCircumcenter();
    }

    // TODO: get neighbors + unique edges
}


int main(int argc, char **argv) {
    assert(argc == 2);

    // Get points list
    std::string filename = argv[1];
    std::vector<Vertex> points;
    int num_points = readPointsFile(filename, points);

    std::cout << "Num points:" << num_points << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Get delaunay triangulation
    std::vector<Triangle> triangles = bowyerWatson(points);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Num triangles:" << triangles.size() << std::endl;

    std::cout << "Delaunay time: " << duration.count() << "ms" << std::endl;

    // writeTrianglesFile("triangles.txt", triangles);

    start = std::chrono::high_resolution_clock::now();

    vornoi(triangles);

    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Vornoi time: " << duration.count() << "ms" << std::endl;
}