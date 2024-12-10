#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include "kernels/det3/kernel1.h"
#include <omp.h>

#define SUPER_TRIANGLE_MAX 10000000

#define NUM_TRIANGLES 32768

#define NUM_ELEMS 256

#define ALIGNMENT 32

#define SIMD_SIZE 8

#define NUM_SIMD_IN_KERNEL (NUM_TRIANGLES / SIMD_SIZE)

#define DET3_KERNEL_SIZE 2

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

    float inCircumcircle(const Vertex& v) const {
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

        return out;
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
        auto hashVertex = [](const Vertex& v) {
            return std::hash<float>()(v.x) ^ std::hash<float>()(v.y);
        };

        std::size_t hash1 = hashVertex(edge.v0);
        std::size_t hash2 = hashVertex(edge.v1);

        return hash1 < hash2 ? hash1 ^ hash2 : hash2 ^ hash1;
    }
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

struct delaunay_points {
    float Ax[DET3_KERNEL_SIZE*SIMD_SIZE];
    float Ay[DET3_KERNEL_SIZE*SIMD_SIZE];
    float Bx[DET3_KERNEL_SIZE*SIMD_SIZE];
    float By[DET3_KERNEL_SIZE*SIMD_SIZE];
    float Cx[DET3_KERNEL_SIZE*SIMD_SIZE];
    float Cy[DET3_KERNEL_SIZE*SIMD_SIZE];
};
typedef struct delaunay_points delaunay_points_t;

struct packed_delaunay_points {
    delaunay_points_t packedPoints[NUM_TRIANGLES/(DET3_KERNEL_SIZE*SIMD_SIZE)];
};
typedef struct packed_delaunay_points packed_delaunay_points_t;

void packDelaunay(const std::vector<Triangle>& triangles, const Vertex& vertex, packed_delaunay_points_t* packedData) {
    size_t numTriangles = triangles.size();
    size_t numKernelIter = (numTriangles + (SIMD_SIZE*DET3_KERNEL_SIZE) - 1) / (SIMD_SIZE*DET3_KERNEL_SIZE); // Correct rounding

    for (size_t i = 0; i < numKernelIter; i++){
        for (size_t j = 0; j < SIMD_SIZE*DET3_KERNEL_SIZE; j++){
            size_t currentTriangle = i * (SIMD_SIZE*DET3_KERNEL_SIZE) + j;
            if (currentTriangle < triangles.size()){
                packedData->packedPoints[i].Ax[j] = triangles[currentTriangle].v0.x;
                packedData->packedPoints[i].Ay[j] = triangles[currentTriangle].v0.y;
                packedData->packedPoints[i].Bx[j] = triangles[currentTriangle].v1.x;
                packedData->packedPoints[i].By[j] = triangles[currentTriangle].v1.y;
                packedData->packedPoints[i].Cx[j] = triangles[currentTriangle].v2.x;
                packedData->packedPoints[i].Cy[j] = triangles[currentTriangle].v2.y;
            } 
            else{
                packedData->packedPoints[i].Ax[j] = 0.0f;
                packedData->packedPoints[i].Ay[j] = 0.0f;
                packedData->packedPoints[i].Bx[j] = 0.0f;
                packedData->packedPoints[i].By[j] = 0.0f;
                packedData->packedPoints[i].Cx[j] = 0.0f;
                packedData->packedPoints[i].Cy[j] = 0.0f;
            }
        }
    }
}


std::vector<Triangle> addVertex(Vertex& vertex, std::vector<Triangle>& triangles) {
    // std::unordered_set<Edge, EdgeHash> unique_edges;
    std::vector<Edge> edges;
    std::vector<Triangle> filtered_triangles;

    //TODO: use unordered list for better perf, hashing isnt working??

    // Pack delaunay data
    packed_delaunay_points_t* packedData;
    posix_memalign((void**) &packedData, ALIGNMENT, sizeof(packed_delaunay_points_t));
    packDelaunay(triangles, vertex, packedData);

    // Run kernel
    int kernelIter = (triangles.size() + (SIMD_SIZE*DET3_KERNEL_SIZE) - 1) / (SIMD_SIZE*DET3_KERNEL_SIZE);
    float *det3_out;
    posix_memalign((void**) &det3_out, ALIGNMENT, kernelIter * DET3_KERNEL_SIZE * SIMD_SIZE * sizeof(float));
    float x = vertex.x;
    float y = vertex.y;
    for (int i = 0; i < kernelIter; i++){
        kernel( (packedData->packedPoints[i].Ax),
                (packedData->packedPoints[i].Ay),
                (packedData->packedPoints[i].Bx),
                (packedData->packedPoints[i].By),
                (packedData->packedPoints[i].Cx),
                (packedData->packedPoints[i].Cy),
                x, // Dx
                y, // Dy
                &det3_out[DET3_KERNEL_SIZE * SIMD_SIZE*i]);
    }

    // int correct = 1;
    // for (int j = 0; j < triangles.size(); j++){
    //     int kernel_idx = j/16;
    //     int idx = j%16;
    //     float res = triangles[j].inCircumcircle(vertex);
    //     correct &= (fabs(det3_out[j] - res) < 1e-13);
    // }
    // if (!correct){
    //     printf("INCORRECT\n");
    // }

    // Remove triangles with circumcircles containing the vertex
    int t = 0;
    for (Triangle& triangle : triangles) {
        if (det3_out[t] > 0) {
            // unique_edges.insert(Edge(triangle.v0, triangle.v1));
            // unique_edges.insert(Edge(triangle.v1, triangle.v2));
            // unique_edges.insert(Edge(triangle.v2, triangle.v0)); 
            edges.emplace_back(Edge(triangle.v0, triangle.v1));
            edges.emplace_back(Edge(triangle.v1, triangle.v2));
            edges.emplace_back(Edge(triangle.v2, triangle.v0));
        } else {
          filtered_triangles.emplace_back(triangle);
        }
        t++;
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
    
    free(det3_out);
    free(packedData);
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