#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace Eigen;


int N = 5;
int nSpecies = 3;
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
int wDead = 11;
int wRepl = 1;
int wBoost = 100;


const float pDeath = 0.2;

std::vector<MatrixXi> lattice;
MatrixXi latticeFlat;

std::vector<MatrixXi> claims;
std::vector<MatrixXi> filtersNWSE;
std::vector<MatrixXi> filtersBoost;


void controls(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GL_TRUE);
}
GLFWwindow* initWindow(const int resX, const int resY)
{
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return NULL;
    }
    glfwWindowHint(GLFW_SAMPLES, 1); // 1x antialiasing

    // Open a window and create its OpenGL context
    GLFWwindow* window = glfwCreateWindow(resX, resY, "TEST", NULL, NULL);

    if (window == NULL)
    {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return NULL;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, controls);

    // Get info of GPU and supported OpenGL version
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL version supported %s\n", glGetString(GL_VERSION));

    glEnable(GL_DEPTH_TEST); // Depth Testing
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    return window;
}

void DrawMatrix() {
    const float cellWidth = static_cast<float>(WINDOW_WIDTH) / N;
    const float cellHeight = static_cast<float>(WINDOW_HEIGHT) / N;

    glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int species = latticeFlat(i, j);
            // Set color based on species
            switch (species) {
                case 0: glColor3f(0.1f, 0.1f, 0.1f); break; // Dead
                case 1: glColor3f(1.0f, 0.0f, 0.0f); break; // Species 1
                case 2: glColor3f(0.0f, 1.0f, 0.0f); break; // Species 2
                case 3: glColor3f(0.0f, 1.0f, 1.0f); break; // Species 3
                default: glColor3f(1.0f, 1.0f, 1.0f);      // Others
            }

            float x = j * cellWidth, y = i * cellHeight;
            glBegin(GL_QUADS);
            glVertex2f(x, y);
            glVertex2f(x + cellWidth, y);
            glVertex2f(x + cellWidth, y + cellHeight);
            glVertex2f(x, y + cellHeight);
            glEnd();
        }
    }
}




MatrixXi flattenLattice(const std::vector<MatrixXi>& lat) {
    MatrixXi flat(N, N);
    flat.setZero();
    for (int i = 0; i <= nSpecies; ++i) {
        flat += i * lat[i];
    }
    return flat;
}
std::vector<MatrixXi> unflattenLattice(const MatrixXi& flatLattice) {
    std::vector<MatrixXi> outLattice(nSpecies + 1, MatrixXi::Zero(N, N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outLattice[flatLattice(i, j)](i, j) = 1;
        }
    }
    return outLattice;
}
void initFilters() {
    // Filters
    filtersNWSE.push_back((MatrixXi(3, 3) << 0, 0, 0, 0, 0, 0, 0, 1, 0).finished());
    filtersNWSE.push_back((MatrixXi(3, 3) << 0, 0, 0, 1, 0, 0, 0, 0, 0).finished());
    filtersNWSE.push_back((MatrixXi(3, 3) << 0, 1, 0, 0, 0, 0, 0, 0, 0).finished());
    filtersNWSE.push_back((MatrixXi(3, 3) << 0, 0, 0, 0, 0, 1, 0, 0, 0).finished());

    // Boost filters
    filtersBoost.push_back((MatrixXi(3, 3) << 1, 0, 1, 1, 0, 1, 0, 0, 0).finished());
    filtersBoost.push_back((MatrixXi(3, 3) << 1, 1, 0, 0, 0, 0, 1, 1, 0).finished());
    filtersBoost.push_back((MatrixXi(3, 3) << 0, 0, 0, 1, 0, 1, 1, 0, 1).finished());
    filtersBoost.push_back((MatrixXi(3, 3) << 0, 1, 1, 0, 0, 0, 0, 1, 1).finished());

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
    lattice.clear();
    for (int i = 0; i < nSpecies + 1; ++i) {
        lattice.push_back(MatrixXi::Zero(N, N));
        claims.push_back(MatrixXi::Zero(N, N));
    }

    // Initial random assignment
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, nSpecies);  // For species including dead

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int randVal = distrib(gen);
            lattice[randVal](i, j) = 1;
        }
    }
/*
    for (int k = 0; k < lattice.size(); k++) {
        std::cout << "species" << k << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << lattice[k](i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    */
    latticeFlat = flattenLattice(lattice);
}
void zeroClaims() {
    for (int i = 0; i < nSpecies + 1; i++) {
        MatrixXi tmp(N, N);
        tmp.setZero();
        claims[i] = tmp;
    }
}

void runSimulationStep() {
    latticeFlat = flattenLattice(lattice);
    zeroClaims();
    claims[0] = MatrixXi::Constant(N, N, wDead); // Initialize dead cells with their weights


    // compute claims
    for (int s = 1; s <= nSpecies; s++) {
        int sBoost = (s == 1) ? nSpecies : s - 1;

        for (int f = 0; f < filtersNWSE.size(); f++) {
            MatrixXi tmpClaim = convolve2D(lattice[s], filtersNWSE[f]);
            claims[s] += wRepl * tmpClaim;

            MatrixXi allBoosts = convolve2D(lattice[sBoost], filtersBoost[f]);
            claims[s] += wBoost * allBoosts.cwiseProduct(tmpClaim);
        }
    }

    // compute cumulative claims
    std::vector<MatrixXi> cumClaims(nSpecies + 1);

    cumClaims[0] = claims[0];
    for (int s = 1; s <= nSpecies; s++) {
        cumClaims[s] = cumClaims[s - 1] + claims[s];
    }

    // convert to probabilities
    std::vector<MatrixXf> cumClaimsProb(nSpecies + 1);
    MatrixXi totalClaims = cumClaims[nSpecies]; // total sum
    for (int s = 1; s <= nSpecies; s++) {
        cumClaimsProb[s] = (cumClaims[s].array().cast<float>() / totalClaims.array().cast<float>()).matrix();
    }

    // select replicant by the cumulative prob
    MatrixXi newLatticeFlat(N, N);
    MatrixXf rand = MatrixXf::Random(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float r = (rand(i, j) + 1) / 2;  // Scale to [0,1]
            int s;
            for (s = 1; s <= nSpecies; s++) {
                if (r <= cumClaims[s](i, j)) {
                    break;
                }
            }
            newLatticeFlat(i, j) = s;
        }
    }

    // 5. Combine claims and survival/death logic
    rand = MatrixXf::Random(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (latticeFlat(i, j) != 0 && (rand(i, j) + 1) / 2 < pDeath) {
                newLatticeFlat(i, j) = 0;  // A full cell dies
            }
            if (latticeFlat(i, j) != 0 && (rand(i, j) + 1) / 2 >= pDeath) {
                newLatticeFlat(i, j) = latticeFlat(i, j);  // A full cell survives
            }
        }
    }

    lattice = unflattenLattice(newLatticeFlat);
}

void display(GLFWwindow* window)
{
    while (!glfwWindowShouldClose(window))
    {
        // Scale to window size
        GLint WINDOW_WIDTH, WINDOW_HEIGHT;
        glfwGetWindowSize(window, &WINDOW_WIDTH, &WINDOW_HEIGHT);
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

        // Draw stuff
        glClearColor(0.0, 0.8, 0.3, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION_MATRIX);
        glLoadIdentity();
        gluPerspective(60, (double)WINDOW_WIDTH / (double)WINDOW_HEIGHT, 0.1, 1000);

        glMatrixMode(GL_MODELVIEW_MATRIX);
        glTranslatef(-WINDOW_WIDTH/2, -WINDOW_HEIGHT/2, -1000);
        DrawMatrix();
        runSimulationStep();

        // Update Screen
        glfwSwapBuffers(window);

        // Check for any input, or window movement
        glfwPollEvents();
    }
}

int main() {
    initLattice();
    initFilters();

    GLFWwindow* window = initWindow(WINDOW_WIDTH, WINDOW_HEIGHT);

    if (NULL != window)
    {
        display(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}