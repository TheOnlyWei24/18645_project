all:
	g++ -O3 -std=c++17 -mavx -mavx2 -fopenmp main.cpp -o main.x