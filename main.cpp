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

#define NUM_TRIANGLES 32000

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

        return out>0;
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
    #pragma omp parallel for num_threads(2) schedule(static)
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


std::vector<Triangle> addVertex(Vertex& vertex, std::vector<Triangle>& triangles, packed_delaunay_points_t* packedData, float* det3_out) {
    std::unordered_set<Edge, EdgeHash> unique_edges; 
    std::vector<Triangle> filtered_triangles;

    // Pack delaunay data
    packDelaunay(triangles, vertex, packedData);

    int kernelIter = (triangles.size() + (SIMD_SIZE * DET3_KERNEL_SIZE) - 1) / (SIMD_SIZE * DET3_KERNEL_SIZE);
    float x = vertex.x;
    float y = vertex.y;

<<<<<<< Updated upstream
    for (int i = 0; i < kernelIter; i++) {
        kernel((packedData->packedPoints[i].Ax),
               (packedData->packedPoints[i].Ay),
               (packedData->packedPoints[i].Bx),
               (packedData->packedPoints[i].By),
               (packedData->packedPoints[i].Cx),
               (packedData->packedPoints[i].Cy),
               x, // Dx
               y, // Dy
               &det3_out[DET3_KERNEL_SIZE * SIMD_SIZE * i]);
    }
=======
    #pragma omp parallel for num_threads(2) schedule(static)
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
>>>>>>> Stashed changes

    // Process triangles and collect edges
    int t = 0;
    for (Triangle& triangle : triangles) {
        if (det3_out[t] > 0) {
            // Add edges of the triangle to the unique edges set
            Edge e1(triangle.v0, triangle.v1);
            Edge e2(triangle.v1, triangle.v2);
            Edge e3(triangle.v2, triangle.v0);

            // Insert or erase edges to ensure uniqueness
            if (!unique_edges.insert(e1).second) unique_edges.erase(e1);
            if (!unique_edges.insert(e2).second) unique_edges.erase(e2);
            if (!unique_edges.insert(e3).second) unique_edges.erase(e3);
        } else {
            filtered_triangles.push_back(triangle);
        }
        t++;
    }

    for (const Edge& edge : unique_edges) {
        filtered_triangles.emplace_back(Triangle(edge.v0, edge.v1, vertex));
    }

    return filtered_triangles;
}


std::vector<Triangle> bowyerWatson(std::vector<Vertex>& points) {
    // Add super triangle
    std::vector<Triangle> triangles;
    Triangle st = superTriangle();
    triangles.push_back(st);

    packed_delaunay_points_t* packedData;
    posix_memalign((void**) &packedData, ALIGNMENT, sizeof(packed_delaunay_points_t));
    int kernelIter = (NUM_TRIANGLES + (SIMD_SIZE*DET3_KERNEL_SIZE) - 1) / (SIMD_SIZE*DET3_KERNEL_SIZE) + 100;
    float *det3_out;
    posix_memalign((void**) &det3_out, ALIGNMENT, kernelIter * DET3_KERNEL_SIZE * SIMD_SIZE * sizeof(float));

    // Triangulate each vertex
    for (Vertex& vertex : points) {
        triangles = addVertex(vertex, triangles, packedData, det3_out);
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

    free(packedData);
    free(det3_out);
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