#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>

#define SUPER_TRIANGLE_MAX 10000000

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
};


struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
        // Hash vertices in a consistent order (ignoring edge direction)
        auto hashVertex = [](const Vertex& v) {
            return std::hash<float>()(v.x) ^ std::hash<float>()(v.y);
        };

        // Ensure the smaller vertex is always first for consistent hashing
        const Vertex& v0 = edge.v0.x < edge.v1.x ? edge.v0 : edge.v1;
        const Vertex& v1 = edge.v0.x < edge.v1.x ? edge.v1 : edge.v0;

        std::size_t hash1 = hashVertex(v0);
        std::size_t hash2 = hashVertex(v1);

        return hash1 ^ hash2;
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


std::vector<Triangle> addVertex(Vertex& vertex, std::vector<Triangle>& triangles) {
    // std::unordered_set<Edge, EdgeHash> unique_edges;
    std::vector<Edge> edges;
    std::vector<Triangle> filtered_triangles;

    //TODO: use unordered list for better perf

    // Remove triangles with circumcircles containing the vertex
    for (Triangle& triangle : triangles) {
        if (triangle.inCircumcircle(vertex)) {
            // unique_edges.insert(Edge(triangle.v0, triangle.v1));
            // unique_edges.insert(Edge(triangle.v1, triangle.v2));
            // unique_edges.insert(Edge(triangle.v2, triangle.v0)); 
            edges.push_back(Edge(triangle.v0, triangle.v1));
            edges.push_back(Edge(triangle.v1, triangle.v2));
            edges.push_back(Edge(triangle.v2, triangle.v0));
        } else {
          filtered_triangles.push_back(triangle);
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
            unique_edges.push_back(edges[i]);
        }
    }

    // Create new triangles from the unique edges and new vertex
    for (Edge edge : unique_edges) {
        filtered_triangles.push_back(Triangle(edge.v0, edge.v1, vertex));
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
    std::vector<Triangle> filtered_triangles;
    for (Triangle& triangle : triangles) {
        if (!(triangle.v0 == st.v0 || triangle.v0 == st.v1 || triangle.v0 == st.v2 ||
              triangle.v1 == st.v0 || triangle.v1 == st.v1 || triangle.v1 == st.v2 ||
              triangle.v2 == st.v0 || triangle.v2 == st.v1 || triangle.v2 == st.v2)) {
            filtered_triangles.push_back(triangle);
        }
    }

    return filtered_triangles;
}


int main(int argc, char **argv) {
    assert(argc == 2);

    // Get points list
    std::string filename = argv[1];
    std::vector<Vertex> points;
    int num_points = readPointsFile(filename, points);

    std::cout << num_points << std::endl;

    // Get delaunay triangulation
    std::vector<Triangle> triangles = bowyerWatson(points);

    std::cout << triangles.size() << std::endl;

    // writeTrianglesFile("triangles.txt", triangles);
}