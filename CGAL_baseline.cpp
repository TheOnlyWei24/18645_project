#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/Object.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random> 
#include <gmp.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
typedef K::Point_2 Point;
typedef K::Segment_2 Segment;
typedef K::Ray_2 Ray;
typedef K::Line_2 Line;


// Source: https://doc.cgal.org/latest/Triangulation_2/index.html#title14
void benchmark_delaunay_voronoi(int num_points, const std::string& point_type, unsigned seed) {
    std::vector<Point> points;

    // Initialize random number generator with a known seed
    std::mt19937 rng(seed); 
    std::uniform_real_distribution<double> dist(-1.0, 1.0); // Generate points in the range [-1, 1]

    // Generate points based on type
    if (point_type == "random") {
        for (int i = 0; i < num_points; ++i) {
            points.push_back(Point(dist(rng), dist(rng)));
        }
    } else if (point_type == "grid") {
        int grid_size = static_cast<int>(std::sqrt(num_points));
        for (int i = 0; i <= grid_size; ++i) {
            for (int j = 0; j <= grid_size; ++j) {
                points.push_back(Point(i * 0.1, j * 0.1));
            }
        }
        // Adjust the number of points in case grid_size^2 > num_points
        while (points.size() > static_cast<size_t>(num_points)) {
            points.pop_back();
        }
    } else if (point_type == "clustered") {
        int clusters = 5;
        int points_per_cluster = num_points / clusters;
        for (int c = 0; c < clusters; ++c) {
            double cluster_center_x = (c % clusters) * 10.0;
            double cluster_center_y = (c % clusters) * 10.0;
            for (int i = 0; i < points_per_cluster; ++i) {
                points.push_back(Point(
                    cluster_center_x + (dist(rng) * 0.1),
                    cluster_center_y + (dist(rng) * 0.1)));
            }
        }
        // Adjust in case the number of points is less than num_points
        while (points.size() < static_cast<size_t>(num_points)) {
            points.push_back(Point(
                dist(rng) * 5.0,
                dist(rng) * 5.0));
        }
    }

    // Measure time for Delaunay triangulation construction
    auto start_delaunay = std::chrono::high_resolution_clock::now();
    Delaunay dt;
    dt.insert(points.begin(), points.end());
    auto end_delaunay = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_delaunay = end_delaunay - start_delaunay;

    // Measure time for Voronoi diagram calculation
    auto start_voronoi = std::chrono::high_resolution_clock::now();
    // Iterate over all edges to compute the Voronoi diagram
    int voronoi_edge_count = 0;
    for (auto eit = dt.edges_begin(); eit != dt.edges_end(); ++eit) {
        CGAL::Object o = dt.dual(eit);
        if (const Segment* s = CGAL::object_cast<Segment>(&o)) {
            voronoi_edge_count++;
        } else if (const Ray* r = CGAL::object_cast<Ray>(&o)) {
            voronoi_edge_count++;
        } else if (const Line* l = CGAL::object_cast<Line>(&o)) {
            voronoi_edge_count++;
        }
    }
    auto end_voronoi = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_voronoi = end_voronoi - start_voronoi;

    // Print performance data
    std::cout << "Data Type: " << point_type 
              << " | Points: " << num_points 
              << " | Delaunay Time: " << elapsed_delaunay.count() << " seconds"
              << " | Voronoi Time: " << elapsed_voronoi.count() << " seconds"
              << " | Voronoi Edges: " << voronoi_edge_count << std::endl;
}

int main() {
    
    std::vector<int> data_sizes = {1000}; 
    std::vector<std::string> point_types = {"random", "grid", "clustered"};
    
    unsigned seed = 42;

    for (int i = 0; i < 5; i++) {
        for (const auto& type : point_types) {
            for (int size : data_sizes) {
                benchmark_delaunay_voronoi(size, type, seed);
            }
        }
    }

    return 0;
}

/*
CMakeLists.txt 
cmake_minimum_required(VERSION 3.10)
project(voronoi_project)

# Set paths for local installations of CGAL, GMP, and Boost
set(CMAKE_PREFIX_PATH "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/local")
set(CMAKE_MODULE_PATH "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/local/lib/cmake")

# Specify the CGAL directory explicitly
set(CGAL_DIR "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/cgal")

# Find CGAL
find_package(CGAL REQUIRED)

# Find GMP
find_library(GMP_LIBRARY NAMES gmp PATHS "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/local/lib")
find_path(GMP_INCLUDE_DIR gmp.h PATHS "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/local/include")

include_directories(${GMP_INCLUDE_DIR})

# Find Boost
set(BOOST_ROOT "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/local")
set(BOOST_INCLUDEDIR "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/local/include")
set(BOOST_LIBRARYDIR "/afs/ece.cmu.edu/usr/bsridha2/Private/18645/local/lib")

find_package(Boost 1.72 REQUIRED COMPONENTS system filesystem)

# Disable CUDA in Boost (if CUDA is causing issues)
add_definitions(-DBOOST_NO_CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add your executable
add_executable(voronoi voronoi.cpp)

# Link libraries (CGAL, GMP, and Boost)
target_link_libraries(voronoi CGAL::CGAL ${GMP_LIBRARY} Boost::system Boost::filesystem)



 */