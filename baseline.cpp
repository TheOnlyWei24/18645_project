#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Function to draw the Delaunay triangulation
void drawDelaunay(Mat& img, Subdiv2D& subdiv, Scalar color) {
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    for (size_t i = 0; i < triangleList.size(); i++) {
        Vec6f t = triangleList[i];
        Point pt1(cvRound(t[0]), cvRound(t[1]));
        Point pt2(cvRound(t[2]), cvRound(t[3]));
        Point pt3(cvRound(t[4]), cvRound(t[5]));

        if (pt1.inside(Rect(0, 0, img.cols, img.rows)) &&
            pt2.inside(Rect(0, 0, img.cols, img.rows)) &&
            pt3.inside(Rect(0, 0, img.cols, img.rows))) {
            line(img, pt1, pt2, color, 1, LINE_AA, 0);
            line(img, pt2, pt3, color, 1, LINE_AA, 0);
            line(img, pt3, pt1, color, 1, LINE_AA, 0);
        }
    }
}

// Function to draw the Voronoi diagram
void drawVoronoi(Mat& img, Subdiv2D& subdiv) {
    vector<vector<Point2f>> facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

    for (size_t i = 0; i < facets.size(); i++) {
        vector<Point> ifacet;
        for (size_t j = 0; j < facets[i].size(); j++) {
            ifacet.push_back(facets[i][j]);
        }
        polylines(img, ifacet, true, Scalar(0, 255, 0), 1, LINE_AA, 0); // Green lines for Voronoi
        circle(img, centers[i], 3, Scalar(0, 0, 255), FILLED, LINE_AA); // Red dots for Voronoi centers
    }
}

int main() {
    const int window_size = 16000;
    Mat img = Mat::zeros(window_size, window_size, CV_8UC3);
    img = Scalar(255, 255, 255);  // White background

    // Initialize the Subdiv2D object
    Rect rect(0, 0, window_size, window_size);
    Subdiv2D subdiv(rect);

    // Generate 1000 random points
    vector<Point2f> points;
    srand(time(0));
    for (int i = 0; i < 8000; ++i) {
        float x = rand() % window_size;
        float y = rand() % window_size;
        points.push_back(Point2f(x, y));
    }

    // Measure time for Delaunay triangulation core computation
    auto startDelaunay = high_resolution_clock::now();
    for (const auto& point : points) {
        subdiv.insert(point);
    }
    auto endDelaunay = high_resolution_clock::now();
    auto delaunayDuration = duration_cast<milliseconds>(endDelaunay - startDelaunay).count();
    cout << "Time taken for Delaunay triangulation (core computation): " << delaunayDuration << " ms" << endl;

    // Measure time for Voronoi diagram core computation
    auto startVoronoi = high_resolution_clock::now();
    vector<vector<Point2f>> facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
    auto endVoronoi = high_resolution_clock::now();
    auto voronoiDuration = duration_cast<milliseconds>(endVoronoi - startVoronoi).count();
    cout << "Time taken for Voronoi diagram (core computation): " << voronoiDuration << " ms" << endl;

    // Draw Delaunay triangulation
    // drawDelaunay(img, subdiv, Scalar(0, 0, 255));

    // Draw Voronoi diagram
    // drawVoronoi(img, subdiv);

    // imwrite("delaunay_voronoi_result.png", img);

    return 0;
}

