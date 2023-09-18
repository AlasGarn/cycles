#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


using namespace Eigen;

std::vector<std::vector<int>> replMatrixData = { {0, 1, 0}, {1, 0, 1}, {0, 1, 0} };
Eigen::Tensor<int, 2> replMatrix(3, 3);

// "up", "left", "down", "right"
const std::vector<std::vector<int>> boostLogic = {
    {{4,5,1,3}}, // up
    {{4,6,0,2}}, // left
    {{6,7,1,3}}, // down
    {{5,7,0,2}} // right
};


std::vector<Eigen::Tensor<int, 2>> roll(const Eigen::Tensor<int, 2>& mat, int shiftX, int shiftY);

Eigen::Tensor<int, 3> onehotMult(const Eigen::Tensor<int, 2>& mat, int nSpecies, int a);

Eigen::Tensor<int, 2> flattenClaims(Eigen::Tensor<int, 3>& claims, int nSpecies, int N);


Eigen::Tensor<int, 2> initLattice(Eigen::Tensor<int, 2>& lat, int nSpecies);

//void zeroClaims(Eigen::Tensor<int, 3>& claims, int nSpecies);

std::vector<Eigen::Tensor<int, 2>> computeNeighbours(const Eigen::Tensor<int, 2>& lattice);
