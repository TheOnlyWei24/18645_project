Delaunay Triangulation and Voronoi Diagram

To compile, run `make`.
This generates the binary `main.x`.

To run the program:
    Usage: ./main.x <input_file>
    Example: ./main.x points1600.txt

Available input files for testing:
- points200.txt
- points400.txt
- points800.txt
- points1600.txt
- points2000.txt
- points4000.txt
- points8000.txt
- points16000.txt
- points32000.txt
- points64000.txt

The input file should contain coordinates in the format:
    x1 y1
    x2 y2
    ...

Output includes:
- Number of points
- Number of triangles
- Execution times for Delaunay triangulation and Voronoi diagram generation.

Machines: This program is designed/optimized to run on Intel Xeon 2640 ECE machines (e.g., `ece010.ece.local.cmu.edu`).
