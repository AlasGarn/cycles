#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace Eigen;

const int N = 300;
const int nSpecies = 3;
const float pDeath = 0.2;
const int nSteps = 10;
const int wDead = 11;
const int wRepl = 1;
const int wBoost = 100;

std::vector<MatrixXi> lattice;
std::vector<MatrixXi> newLattice;
MatrixXi latticeFlat;

std::vector<MatrixXi> claims;
// Define filter containers
std::vector<MatrixXi> filtersNWSE;
std::vector<MatrixXi> filtersBoost;


// Initialize OpenGL window and context

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

// Function to initialize OpenGL context using GLFW
bool Initialize() {
    if (!glfwInit()) {
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    return true;
}

// Drawing function
void DrawMatrix(const MatrixXi& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();

    // Calculate cell size based on window size
    float cellWidth = static_cast<float>(WINDOW_WIDTH) / cols;
    float cellHeight = static_cast<float>(WINDOW_HEIGHT) / rows;

    glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int species = matrix(i, j);

            // Set color based on species
            switch (species) {
            case 0: glColor3f(0.0f, 0.0f, 0.0f); break; // Black for dead cells
            case 1: glColor3f(1.0f, 0.0f, 0.0f); break; // Example color for species 1
            case 2: glColor3f(1.0f, 1.0f, 0.0f); break; // Example color for species 1
            default: glColor3f(1.0f, 1.0f, 1.0f); break; // Default white color
            }

            // Draw the cell as a rectangle
            float x = j * cellWidth;
            float y = i * cellHeight;

            glBegin(GL_QUADS);
            glVertex2f(x, y);
            glVertex2f(x + cellWidth, y);
            glVertex2f(x + cellWidth, y + cellHeight);
            glVertex2f(x, y + cellHeight);
            glEnd();
        }
    }
}

MatrixXi flattenLattice(std::vector<MatrixXi> lattice) {
    MatrixXi flatLattice(N, N);
    flatLattice.setZero();

    for (int i = 0; i < lattice.size(); i++) {
        flatLattice += i * lattice[i];
    }
    return flatLattice;
}

std::vector<MatrixXi> unflattenLattice(MatrixXi latticeFlat) {
    std::vector<MatrixXi> outLattice;

    for (int i = 0; i < nSpecies+1; i++) {
        MatrixXd tmp(N, N);
        tmp.setZero();
        outLattice.push_back(tmp);
    }
    int s = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            s = latticeFlat(i, j);
            outLattice[s](i, j) = 1;
        }
    }
    return outLattice;
}

MatrixXi convolve2D(const MatrixXi& input, const MatrixXi& kernel) {
    int rows = input.rows();
    int cols = input.cols();

    int kRows = kernel.rows();
    int kCols = kernel.cols();

    int outRows = rows;
    int outCols = cols;

    MatrixXi output = MatrixXi::Zero(outRows, outCols);

    // Iterate over the input image
    for (int i = 0; i < outRows; ++i) {
        for (int j = 0; j < outCols; ++j) {

            // Iterate over the kernel
            for (int ki = 0; ki < kRows; ++ki) {
                for (int kj = 0; kj < kCols; ++kj) {

                    int ii = i + ki - kRows / 2;
                    int jj = j + kj - kCols / 2;

                    // Wrap around logic for boundary
                    ii = (ii + rows) % rows;
                    jj = (jj + cols) % cols;

                    output(i, j) += input(ii, jj) * kernel(ki, kj);
                }
            }
        }
    }
    return output;
}

void initLattice() {

    // init main lattice
    MatrixXi latticeFlat(N, N);
    MatrixXf random = MatrixXf::Random(N, N);

    std::uniform_int_distribution<> distr(1, nSpecies + 1); // define the range

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // TODO: if i+j out of radius - continue
            latticeFlat(i, j) = distr(random(i, j));
        }
    }

    lattice = unflattenLattice(latticeFlat);
}

void zeroClaims() {
    for (int i = 0; i < nSpecies + 1; i++) {
        MatrixXi tmp(N,N);
        tmp.setZero();
        claims[i] = tmp;
    }
}


void runSimulationStep() {

    MatrixXi latticeFlat(N,N);
    latticeFlat = flattenLattice(lattice);



    // Replication claims
    zeroClaims();
    claims[0].setOnes();
    claims[0] *= wDead; // add dead weights

    for (int s = 1; s <= nSpecies; s++) {
        int sBoost = (s == 1) ? nSpecies : s - 1;

        for (int f = 0; f < filtersNWSE.size(); f++) {
            MatrixXi tmpClaim = convolve2D(lattice[s], filtersNWSE[f]);  // Convolution function needs to be implemented
            claims[s] += wRepl * tmpClaim;

            MatrixXi allBoosts = convolve2D(lattice[sBoost], filtersBoost[f]); // Convolution function needs to be implemented
            claims[s] += wBoost * allBoosts * tmpClaim;
        }
    }

    // Sample along species axis
    std::vector<MatrixXi> cumClaims;
    MatrixXi tmpSum(N,N);
    tmpSum.setZero();
    // accumulate
    for (int s = 1; s <= nSpecies; s++) {
        tmpSum += claims[s];
        cumClaims.push_back(tmpSum);
    }

    // to probability
    for (int s = 1; s <= nSpecies; s++) {
        cumClaims[s] = cumClaims[s].array() / tmpSum.array();
    }

    // rotate
    VectorXf colProbs(nSpecies + 1);
    MatrixXf rand = MatrixXf::Random(N, N);
    MatrixXi newLatticeFlat(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int s = 1; s <= nSpecies; s++) {
                colProbs << cumClaims[s](i, j);
            }
            newLatticeFlat(i, j) = (colProbs.array() >= rand(i, j)).count() - 1;
        }
    }


    // Combine claims and death/survival

    MatrixXf r = MatrixXf::Random(N, N);

    MatrixXi fullCells(N, N);
    fullCells.setZero();
    fullCells = (latticeFlat.array() != 0).select(1, fullCells);

    MatrixXi fullCellsSurvived(N, N);
    fullCellsSurvived = (r.array() >= pDeath).select(latticeFlat, fullCells);
    newLatticeFlat = (fullCellsSurvived.array() == 1).select(latticeFlat, newLatticeFlat);

    MatrixXi fullCellsDied(N, N);
    fullCellsDied = (r.array() < pDeath).select(latticeFlat, fullCells);
    newLatticeFlat = (fullCellsDied.array() == 1).select(0, newLatticeFlat);

    newLattice = unflattenLattice(newLatticeFlat);
    lattice = newLattice;
}

void initFilters() {
    MatrixXi filterN(3, 3);
    filterN << 0, 0, 0, 0, 0, 0, 0, 1, 0;
    filtersNWSE.push_back(filterN);

    MatrixXi filterE(3, 3);
    filterE << 0, 0, 0, 1, 0, 0, 0, 0, 0;
    filtersNWSE.push_back(filterE);

    MatrixXi filterS(3, 3);
    filterS << 0, 1, 0, 0, 0, 0, 0, 0, 0;
    filtersNWSE.push_back(filterS);

    MatrixXi filterW(3, 3);
    filterW << 0, 0, 0, 0, 0, 1, 0, 0, 0;
    filtersNWSE.push_back(filterW);


    MatrixXi filterBoostN(3, 3);
    filterBoostN << 1, 0, 1, 1, 0, 1, 0, 0, 0;
    filtersBoost.push_back(filterBoostN);

    MatrixXi filterBoostW(3, 3);
    filterBoostW << 1, 1, 0, 0, 0, 0, 1, 1, 0;
    filtersBoost.push_back(filterBoostW);

    MatrixXi filterBoostS(3, 3);
    filterBoostS << 0, 0, 0, 1, 0, 1, 1, 0, 1;
    filtersBoost.push_back(filterBoostS);

    MatrixXi filterBoostE(3, 3);
    filterBoostE << 0, 1, 1, 0, 0, 0, 0, 1, 1;
    filtersBoost.push_back(filterBoostE);
}



int main() {
    initFilters();
    initLattice();
    


    if (!Initialize()) {
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Cell Visualization", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window)) {
        
        runSimulationStep();
        
        DrawMatrix(latticeFlat);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}

