#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>


using namespace Eigen;
int N = 3;
int nSpecies = 2;
std::vector<MatrixXd> lattice;


MatrixXd flattenLattice(std::vector<MatrixXd> lattice) {
    MatrixXd tmpLattice(N, N);
    tmpLattice.setZero();

    for (int i = 0; i < lattice.size(); i++) {
        tmpLattice += i * lattice[i];
    }
    return tmpLattice;
}


int main() {
    for (int i = 0; i < N; i++) {
        MatrixXd tmpLattice(N, N);
        tmpLattice <<
            1, 1, 1,
            1, 1, 1,
            1, 1, 1;
        lattice.push_back(tmpLattice);
       }
    MatrixXd latticeFlat(N, N);

    latticeFlat = flattenLattice(lattice);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << latticeFlat(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
}