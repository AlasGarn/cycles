#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor<int, 2> roll(const Eigen::Tensor<int, 2>& mat, int shiftX, int shiftY) {
    int rows = mat.dimensions()[0];
    int cols = mat.dimensions()[1];
    Eigen::Tensor<int, 2> rolled(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int newX = (i + shiftX) % rows;
            int newY = (j + shiftY) % cols;

            if (newX < 0) newX += rows;
            if (newY < 0) newY += cols;

            rolled(newX, newY) = mat(i, j);
        }
    }
    return rolled;
}
Eigen::Tensor<int, 3> onehotMult(const Eigen::Tensor<int, 2>& mat, int nSpecies, int a) {
    int rows = mat.dimensions()[0];
    int cols = mat.dimensions()[1];

    // Create a Tensor of zeros initially
    Eigen::Tensor<int, 3> onehotTensor(rows, cols, nSpecies + 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int speciesId = mat(i, j);
            if (speciesId >= 0 && speciesId < nSpecies + 1) {
                onehotTensor(speciesId, i, j) = a;
            }
        }
    }

    return onehotTensor;
}

Eigen::Tensor<int, 2> flattenClaims(Eigen::Tensor<int, 3> &claims, int nSpecies, int N) {
    Eigen::array<int, 1> dimensions({ 2 });
    /*
    int rows = claims.size()[0];
    int cols = claims.size()[1];
    MatrixXi flat = MatrixXi::Zero(rows, cols);
  
    for (int i = 0; i <= nSpecies; ++i) {
        flat += i * claims[i];
    }*/
    return claims.sum(dimensions);
}


Eigen::Tensor<int, 2> initLattice(const Eigen::Tensor<int, 2>& lat, int nSpecies) {
    // Initial random assignment

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> speciesDis(1, nSpecies);


    int N = lat.dimensions()[0];

    Eigen::Tensor<int, 2> initDistribution(N, N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            initDistribution(i, j) = speciesDis(gen);
        }
    }
    int Rsq = int(float(N) / 2);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double distance = std::pow(i - N / 2.0, 2) + std::pow(j - N / 2.0, 2);
            if (distance > Rsq) {
                initDistribution(i, j) = 0;
            }
        }
    }

    return initDistribution;
}
/*
void zeroClaims(Eigen::Tensor<int, 3> &claims, int nSpecies) {
    Eigen::Tensor<int, 2> lat = claims[0];
    int rows = lat.rows();
    int cols = lat.cols();

    for (int i = 0; i < nSpecies + 1; i++) {
   
        claims[i] = MatrixXi::Zero(rows, cols);
    }
}
*/

std::vector<Eigen::Tensor<int, 2>> computeNeighbours(const Eigen::Tensor<int, 2>& lattice) {
    Eigen::Tensor<int, 2> up = roll(lattice, -1, 0);
    Eigen::Tensor<int, 2> down = roll(lattice, 1, 0);
    Eigen::Tensor<int, 2> left = roll(lattice, 0, -1);
    Eigen::Tensor<int, 2> right = roll(lattice, 0, 1);

    Eigen::Tensor<int, 2> up_left = roll(up, 0, -1);
    Eigen::Tensor<int, 2> up_right = roll(up, 0, 1);
    Eigen::Tensor<int, 2> down_left = roll(down, 0, -1);
    Eigen::Tensor<int, 2> down_right = roll(down, 0, 1);

    return { up, left, down, right, up_left, up_right, down_left, down_right };
}


int main() { return 0; }