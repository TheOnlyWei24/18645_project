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
#include <cstring>
#include "kernels/circumcenter/kernel.h"

#define SUPER_TRIANGLE_MAX 10000000

#define NUM_TRIANGLES 32000

#define NUM_ELEMS 256

#define ALIGNMENT 32

#define DET3_KERNEL_SIZE 2

#define NUM_THREADS 4

float* kernel_buffer0;
float* kernel_buffer1;
float* kernel_buffer2;
float* kernel_buffer3;

float* det3_out0;
float* det3_out1;
float* det3_out2;
float* det3_out3;

float* out_buff;

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
    Edge e0, e1, e2;

    Triangle(const Vertex& v0_, const Vertex& v1_, const Vertex& v2_)
        : v0(v0_), v1(v1_), v2(v2_), e0(v0_, v1_), e1(v1_, v2_), e2(v2_, v0_) {
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

/**
 * @brief Rounds x to the nearest multiple of n
 */
int roundUpToNearest(int x, int n) {
    return ((x + n - 1) / n) * n;
}


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
    float Dx;
    float Dy;
};
typedef struct delaunay_points delaunay_points_t;

struct packed_delaunay_points {
    delaunay_points_t packedPoints[NUM_TRIANGLES/(DET3_KERNEL_SIZE*SIMD_SIZE)];
};
typedef struct packed_delaunay_points packed_delaunay_points_t;

void packDelaunay(const std::vector<Triangle>& triangles, const Vertex& vertex, packed_delaunay_points_t* packedData, float* det3_out) {
    size_t numTriangles = triangles.size();
    size_t numKernelIter = (numTriangles + (SIMD_SIZE*DET3_KERNEL_SIZE) - 1) / (SIMD_SIZE*DET3_KERNEL_SIZE); // Correct rounding
    //printf("numKernelIter: %d\n", numKernelIter);
    if (numKernelIter<=32){
        //printf("tri: %d, iter: %d\n", numTriangles, numKernelIter);
        float* buffer;
        buffer = kernel_buffer0;
        buffer[96] = vertex.x;
        buffer[97] = vertex.y;
        for (size_t i = 0; i < numKernelIter; i++){
            for (size_t j = 0; j < SIMD_SIZE*DET3_KERNEL_SIZE; j++){
                size_t currentTriangle = i * (SIMD_SIZE*DET3_KERNEL_SIZE) + j;
                if (currentTriangle < triangles.size()){
                    buffer[j]    = triangles[currentTriangle].v0.x;
                    buffer[j+16] = triangles[currentTriangle].v0.y;
                    buffer[j+32] = triangles[currentTriangle].v1.x;
                    buffer[j+48] = triangles[currentTriangle].v1.y;
                    buffer[j+64] = triangles[currentTriangle].v2.x;
                    buffer[j+80] = triangles[currentTriangle].v2.y;
            } 
                else{
                    buffer[j]    = 0.0;
                    buffer[j+16]  = 0.0;
                    buffer[j+32] = 0.0;
                    buffer[j+48] = 0.0;
                    buffer[j+64] = 0.0;
                    buffer[j+80] = 0.0;
                }
            }
            kernel( (&buffer[0]),
                    (&buffer[16]),
                    (&buffer[32]),
                    (&buffer[48]),
                    (&buffer[64]),
                    (&buffer[80]),
                    (buffer[96]), // Dx
                    (buffer[97]), // Dy
                    &det3_out[DET3_KERNEL_SIZE * SIMD_SIZE*i]);
        }
    }
    else{
        #pragma omp parallel num_threads(4)
        {
            int thread_id = omp_get_thread_num();
            size_t start, end;
            float* buffer;

            // Divide the iterations between the 4 threads
            if (thread_id == 0) {
                start = 0;
                end = numKernelIter / 4;
                buffer = kernel_buffer0;
            } else if (thread_id == 1) {
                start = numKernelIter / 4;
                end = numKernelIter / 2;
                buffer = kernel_buffer1;
            } else if (thread_id == 2) {
                start = numKernelIter / 2;
                end = 3 * numKernelIter / 4;
                buffer = kernel_buffer2;
            } else if (thread_id == 3) {
                start = 3 * numKernelIter / 4;
                end = numKernelIter;
                buffer = kernel_buffer3;
            }
            buffer[96] = vertex.x;
            buffer[97] = vertex.y;
            for (size_t i = start; i < end; i++){
                for (size_t j = 0; j < SIMD_SIZE*DET3_KERNEL_SIZE; j++){
                    size_t currentTriangle = i * (SIMD_SIZE*DET3_KERNEL_SIZE) + j;
                    if (currentTriangle < triangles.size()){
                        buffer[j]    = triangles[currentTriangle].v0.x;
                        buffer[j+16] = triangles[currentTriangle].v0.y;
                        buffer[j+32] = triangles[currentTriangle].v1.x;
                        buffer[j+48] = triangles[currentTriangle].v1.y;
                        buffer[j+64] = triangles[currentTriangle].v2.x;
                        buffer[j+80] = triangles[currentTriangle].v2.y;
                } 
                    else{
                        buffer[j]    = 0.0;
                        buffer[j+16]  = 0.0;
                        buffer[j+32] = 0.0;
                        buffer[j+48] = 0.0;
                        buffer[j+64] = 0.0;
                        buffer[j+80] = 0.0;
                    }
                }
                kernel( (&buffer[0]),
                        (&buffer[16]),
                        (&buffer[32]),
                        (&buffer[48]),
                        (&buffer[64]),
                        (&buffer[80]),
                        (buffer[96]), // Dx
                        (buffer[97]), // Dy
                        &det3_out[DET3_KERNEL_SIZE * SIMD_SIZE*i]);
            }
        }

    }
    
}


std::vector<Triangle> addVertex(Vertex& vertex, std::vector<Triangle>& triangles, packed_delaunay_points_t* packedData, float* det3_out) {
    std::unordered_set<Edge, EdgeHash> unique_edges; 
    std::vector<Triangle> filtered_triangles;

    // Pack delaunay data
    packDelaunay(triangles, vertex, packedData, det3_out);

    //printf("DEBUG: %f, %f, %f, %f\n", det3_out0[0], det3_out1[0], det3_out2[0], det3_out3[0]);

    // int kernelIter = (triangles.size() + (SIMD_SIZE * DET3_KERNEL_SIZE) - 1) / (SIMD_SIZE * DET3_KERNEL_SIZE);
    // float x = vertex.x;
    // float y = vertex.y;

    // #pragma omp parallel for num_threads(4) schedule(static)
    // for (int i = 0; i < kernelIter; i++){
    //     kernel( (packedData->packedPoints[i].Ax),
    //             (packedData->packedPoints[i].Ay),
    //             (packedData->packedPoints[i].Bx),
    //             (packedData->packedPoints[i].By),
    //             (packedData->packedPoints[i].Cx),
    //             (packedData->packedPoints[i].Cy),
    //             x, // Dx
    //             y, // Dy
    //             &det3_out[DET3_KERNEL_SIZE * SIMD_SIZE*i]);
    // }

    // Process triangles and collect edges
    int t = 0;
    for (Triangle& triangle : triangles) {

        if (det3_out[t] > 0) {
        // if (triangle.inCircumcircle(vertex)) {
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
    posix_memalign((void**) &kernel_buffer0, ALIGNMENT, 98 * sizeof(float));
    posix_memalign((void**) &kernel_buffer1, ALIGNMENT, 98 * sizeof(float));
    posix_memalign((void**) &kernel_buffer2, ALIGNMENT, 98 * sizeof(float));
    posix_memalign((void**) &kernel_buffer3, ALIGNMENT, 98 * sizeof(float));

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


std::vector<Edge> vornoi(std::vector<Triangle>& triangles) {
    size_t numTriangles = roundUpToNearest(triangles.size(), (SIMD_SIZE * NUM_SIMD_IN_KERNEL));
    size_t numKernelIters = numTriangles / (SIMD_SIZE * NUM_SIMD_IN_KERNEL);
    
    // Pack data
    kernel_data_t *data;
    kernel_buffer_t *buffer;
    posix_memalign((void **)&data, ALIGNMENT, numKernelIters * sizeof(kernel_data_t));
    posix_memalign((void **)&buffer, ALIGNMENT, NUM_THREADS * sizeof(kernel_buffer_t));

   #pragma omp parallel for num_threads(NUM_THREADS) 
    for (int i = 0; i < numKernelIters; i++) {
      for (int j = 0; j < NUM_SIMD_IN_KERNEL; j++) {
        for (int k = 0; k < SIMD_SIZE; k++) {
            size_t currTriangle = (i * NUM_SIMD_IN_KERNEL * SIMD_SIZE) + 
                                  (j * SIMD_SIZE) + k;
            if (currTriangle < triangles.size()) {
                data[i].data[j].Ax[k] = triangles[currTriangle].v0.x;
                data[i].data[j].Ay[k] = triangles[currTriangle].v0.y;
                data[i].data[j].Bx[k] = triangles[currTriangle].v1.x;
                data[i].data[j].By[k] = triangles[currTriangle].v1.y;
                data[i].data[j].Cx[k] = triangles[currTriangle].v2.x;
                data[i].data[j].Cy[k] = triangles[currTriangle].v2.y;
                data[i].data[j].Ux[k] = 0.0;
                data[i].data[j].Uy[k] = 0.0;
            }
        }
      }
    }

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < numKernelIters; i++) {
        vornoi_kernel0(&(data[i]), buffer);
        vornoi_kernel1(&(data[i]), buffer);
        // vornoi_baseline(&data[i]);
    }

    // Unpack data
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < numKernelIters; i++) {
      for (int j = 0; j < NUM_SIMD_IN_KERNEL; j++) {
        for (int k = 0; k < SIMD_SIZE; k++) {
            size_t currTriangle = (i * NUM_SIMD_IN_KERNEL * SIMD_SIZE) + 
                                  (j * SIMD_SIZE) + k;
            if (currTriangle < triangles.size()) {
                triangles[currTriangle].circumcenter.x = data[i].data[j].Ux[k];
                triangles[currTriangle].circumcenter.y = data[i].data[j].Uy[k];
            }
        }
      }
    }

    // for (Triangle triangle : triangles) {
    //     triangle.calculateCircumcenter();
    // }

    // Get vornoi edges
    std::unordered_set<Edge, EdgeHash> vornoi_edges; 
    for (int i = 0; i < triangles.size(); i++) {
        for (int j = 0; j < triangles.size(); j++) {
            if (i != j) {
                if (triangles[i].e0 == triangles[j].e0 || 
                    triangles[i].e0 == triangles[j].e1 || 
                    triangles[i].e0 == triangles[j].e2) {
                    vornoi_edges.insert(triangles[i].e0);
                } else if (triangles[i].e1 == triangles[j].e0 || 
                           triangles[i].e1 == triangles[j].e1 || 
                           triangles[i].e1 == triangles[j].e2) {
                    vornoi_edges.insert(triangles[i].e1);
                } else if (triangles[i].e2 == triangles[j].e0 || 
                           triangles[i].e2 == triangles[j].e1 || 
                           triangles[i].e2 == triangles[j].e2) {
                    vornoi_edges.insert(triangles[i].e2);
                }
            }
        }
    }

    std::vector<Edge> edges(vornoi_edges.begin(), vornoi_edges.end());

    free(data);
    free(buffer);

    return edges;
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

    std::vector<Edge> edges = vornoi(triangles);

    end = std::chrono::high_resolution_clock::now();

    //auto tmp = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Vornoi time: " << tmp.count() << "ns" << std::endl;
}