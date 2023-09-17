#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "helpers.h"
#include "gpt.h"


const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const int wDead = 11;
const int wRepl = 1;
const int wBoost = 100;
const float pDeath = 0.2;


const int N = 50;
Eigen::Tensor<int, 2> lattice(0, N, N);

const int nSpecies = 5;



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
            int species = lattice(i, j);
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


Eigen::Tensor<int, 2> fillNewLattice(const Eigen::Tensor<int, 2>& lattice, const std::vector<Eigen::Tensor<int, 2>>& neighborsMatrix) {

    Eigen::Tensor<int, 3> cumClaims(N, N, nSpecies + 1);
    Eigen::Tensor<int, 3> claims(N, N, nSpecies + 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int replicationDirectionId = 0; replicationDirectionId < 4; ++replicationDirectionId) {
        Eigen::Tensor<int, 2> replicatorSpecies = neighborsMatrix[replicationDirectionId];
        Eigen::Tensor<int, 2> boosterSpeciesRequired = replicatorSpecies + 1;

        for (int i = 0; i < boosterSpeciesRequired.dimensions()[0]; i++) {
            for (int j = 0; j < boosterSpeciesRequired.dimensions()[1]; j++) {
                if (boosterSpeciesRequired(i, j) > nSpecies) {
                    boosterSpeciesRequired(i, j) = 1;
                }
            }
        }
        claims += onehotMult(replicatorSpecies, nSpecies, wRepl);
        // Eigen::TensorSum(claims, onehotMult(replicatorSpecies, nSpecies, wRepl));
        
        

        for (int boostDirectionId : boostLogic[replicationDirectionId]) {
            Eigen::Tensor<int, 2> boosterSpecies = neighborsMatrix[boostDirectionId];
            Eigen::Tensor<int, 2> correctBoosterSpecies = boosterSpecies * (boosterSpecies == boosterSpeciesRequired);
            claims += onehotMult(replicatorSpecies, nSpecies, wBoost);
        }
    }

    // Normalize claims and pick new cells
    Eigen::Tensor<float, 3> cumProb(N, N, nSpecies);
    
    cumProb = claims.cumsum(2);
    cumProb = cumProb / claims.sum(2);
    MatrixXd rr = MatrixXd::Random(N, N).array() * 0.5 + 0.5;

    Eigen::Tensor<int, 2> newLattice(0, N, N);

    // find max along the last (i.e. species) axis
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            
            for (int k = 1; k < nSpecies; k++) {
                if (rr(i, j) > cumProb(i, j, k)) {
                    newLattice(i, j) = k;
                    break;
                }
            }

            
        }
    }

    return newLattice;
}


void runSimulationStep() {


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<Eigen::Tensor<int, 2>> neighbors = computeNeighbours(lattice);

    Eigen::Tensor<int, 2> randomMatrix(N, N);
    randomMatrix.setRandom();
    randomMatrix = randomMatrix * 0.5 + 0.5; // Values between [0,1]
    Eigen::Tensor<int, 2> survived_mask = randomMatrix > pDeath;
    Eigen::Tensor<int, 2> died_mask = survived_mask == 0;

    lattice = lattice * survived_mask;


    Eigen::Tensor<int, 2> replication_mask(N, N);
    Eigen::array<int, 2> dims({ 0, 1 });
    replication_mask = lattice.convolve(replMatrix, dims);
    replication_mask.setConstant(0);
    // Fill replication_mask here, possibly with convolution or another approach.

    Eigen::Tensor<int, 2> replication_spots = (lattice == 0) * replication_mask;

    Eigen::Tensor<int, 2> new_lattice = fillNewLattice(lattice, neighbors);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (replication_spots(i, j) == 1) {
                lattice(i, j) = new_lattice(i, j);
            }
        }
    }
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
    replMatrix.setValues({ {0, 1, 0}, {1, 0, 1}, {0, 1, 0} });

    lattice = initLattice(lattice, nSpecies);

    GLFWwindow* window = initWindow(WINDOW_WIDTH, WINDOW_HEIGHT);

    if (NULL != window)
    {
        display(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}