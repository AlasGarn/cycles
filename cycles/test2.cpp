#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
//#include "helpers.h"
#include "gpt.h"


const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const int wDead = 11;
const int wRepl = 1;
const int wBoost = 100;
const float pDeath = 0.5;


const int N = 5;


const int nSpecies = 3;



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
void DrawMatrix(Eigen::Tensor<int, 2>& lattice) {
    int N = lattice.dimensions()[0];
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
            case 4: glColor3f(1.0f, 1.0f, 1.0f); break; // Species 3
            case 5: glColor3f(0.5f, 0.5f, 0.0f); break; // Species 3
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

Eigen::Tensor<int, 2> convolve2DWithWrap(const Eigen::Tensor<int,2>& input, const Eigen::Tensor<int, 2>& kernel) {
    int rows = input.dimensions()[0];
    int cols = input.dimensions()[1];

    int kRows = kernel.dimensions()[0];
    int kCols = kernel.dimensions()[1];

    Eigen::Tensor<int, 2> output(rows, cols);
    output.setZero();

    // Ensure kernel dimensions are odd for simplicity
    if (kRows % 2 == 0 || kCols % 2 == 0) {
        std::cerr << "Kernel dimensions should be odd for this implementation." << std::endl;
        return output;
    }


    int kHalfRows = kRows / 2;
    int kHalfCols = kCols / 2;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int m = -kHalfRows; m <= kHalfRows; ++m) {
                for (int n = -kHalfCols; n <= kHalfCols; ++n) {
                    int ii = (i + m + rows) % rows;
                    int jj = (j + n + cols) % cols;

                    output(i, j) += input(ii, jj) * kernel(m + kHalfRows, n + kHalfCols);
                }
            }
        }
    }

    return output;
}

Eigen::Tensor<int, 3> onehotMult(const Eigen::Tensor<int, 2>& mat, int nSpecies, int a) {
    int rows = mat.dimensions()[0];
    int cols = mat.dimensions()[1];

    // Create a Tensor of zeros initially
    Eigen::Tensor<int, 3> onehotTensor(rows, cols, nSpecies + 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int speciesIdInLattice = mat(i, j);
            for (int speciesId = 0; speciesId < nSpecies + 1; ++speciesId) {
                onehotTensor(i, j, speciesId) = 0;
                if (speciesId == speciesIdInLattice) {
                    onehotTensor(i, j, speciesId) = 1;
                }
            }
        }
    }

    return onehotTensor;
}

Eigen::Tensor<int, 2> flattenClaims(Eigen::Tensor<int, 3>& claims, int nSpecies, int N) {
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
    int Rsq = N;
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

Eigen::Tensor<int, 2> fillNewLattice(const Eigen::Tensor<int, 2>& lattice, const std::vector<Eigen::Tensor<int, 2>>& neighborsMatrix) {
    const std::vector<std::vector<int>> boostLogic = {
        {{4,5,1,3}}, // up
        {{4,6,0,2}}, // left
        {{6,7,1,3}}, // down
        {{5,7,0,2}}  // right
    };
    int N = lattice.dimensions()[0];
    Eigen::Tensor<int, 3> cumClaims(N, N, nSpecies + 1);
    cumClaims.setZero();
    Eigen::Tensor<int, 3> claims(N, N, nSpecies + 1);
    claims.setZero();
    
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

        
        for (int boostDirectionId : boostLogic[replicationDirectionId]) {
            Eigen::Tensor<int, 2> boosterSpecies = neighborsMatrix[boostDirectionId];
            Eigen::Tensor<bool, 2> boosterSpeciesRequiredBool;
            boosterSpeciesRequiredBool = boosterSpecies == boosterSpeciesRequired;
            Eigen::Tensor<int, 2> correctBoosterSpecies = boosterSpecies * boosterSpeciesRequiredBool.cast<int>();
            claims += onehotMult(replicatorSpecies, nSpecies, wBoost);
        }
 
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            claims(i, j, 0) = wDead;
        }
    }
    std::cout << "claims dims" << claims.dimensions() << std::endl;
    std::cout << "claims " << std::endl << claims << std::endl;

    // Normalize claims and pick new cells
    Eigen::Tensor<int, 3> cumSum = claims.cumsum(2);



    Eigen::array<int, 1> dims({ 2 });
    Eigen::Tensor<int, 2> claimsSum = claims.sum(dims);
    std::cout << "claimsSum dims" << claimsSum.dimensions() << std::endl;
    std::cout << "claimsSum " << std::endl << claimsSum << std::endl;

   /* broadcast then reshape
    Eigen::array<int, 2> bcast({1, nSpecies+1 });
    Eigen::Tensor<int, 2> claimsSumBroadcast = claimsSum.broadcast(bcast);
    std::cout << "claimsSumBroadcast" << claimsSumBroadcast.dimensions() << std::endl;

    std::cout << claimsSumBroadcast << std::endl;

    Eigen::array<int, 3> newShape{ { N, N, nSpecies + 1 } };
    Eigen::Tensor<int, 3> claimsSumReshaped(N, N, nSpecies + 1);
    claimsSumReshaped = claimsSumBroadcast.reshape(newShape);
      std::cout << "claimsSumBroadcast" << claimsSumBroadcast.dimensions() << std::endl;
          std::cout << claimsSumBroadcast << std::endl;
    */

    Eigen::Tensor<float, 3> cumProb(N, N, nSpecies+1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < nSpecies + 1; k++) {
                cumProb(i, j, k) = float(cumSum(i, j, k)) / claimsSum(i, j);
            }
        }
    }
    std::cout << "cumProb" << cumProb.dimensions() << std::endl;
    std::cout << cumProb << std::endl;
 
    Eigen::Tensor<float, 2> randomMatrix(N, N);
    randomMatrix.setRandom();
    randomMatrix = randomMatrix * 0.5f + 0.5f;// Values between [0,1]
    std::cout << "randomMatrix" << randomMatrix.dimensions() << std::endl;
    std::cout << randomMatrix << std::endl;
   
    Eigen::Tensor<int, 2> newLattice(N, N);
    newLattice.setZero();
    // find max along the last (i.e. species) axis
  
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            for (int k = 0; k < nSpecies+1; k++) {
                if (randomMatrix(i, j) < cumProb(i, j, k)) {
                    newLattice(i, j) = k;
                    break;
                }
            }
        }
    }
    std::cout << "newLattice" << std::endl << newLattice << std::endl;
    return newLattice;
}


Eigen::Tensor<int, 2> runSimulationStep(Eigen::Tensor<int, 2> lattice, Eigen::Tensor<int, 2> replMatrix) {

    Eigen::Tensor<float, 2> zeroMatrix(N, N);
    zeroMatrix.setZero();
    Eigen::Tensor<float, 2> randomMatrix(N, N);
    randomMatrix.setRandom();
    randomMatrix = randomMatrix * 0.5f + 0.5f;// Values between [0,1]
    std::vector<Eigen::Tensor<int, 2>> neighbors = computeNeighbours(lattice);
    std::cout << "neighbors"  << std::endl;
    for (int i = 0; i < neighbors.size(); i++) {
        std::cout << neighbors[i] << std::endl << std::endl;
    }
    

    Eigen::Tensor<int, 2> survivedIntMask = (randomMatrix > pDeath).cast<int>() * (lattice > 0).cast<int>();
    Eigen::Tensor<int, 2> diedIntMask = (randomMatrix <= pDeath).cast<int>() * (lattice > 0).cast<int>();

    lattice = lattice * survivedIntMask;

    Eigen::Tensor<int, 2> replicationIntMask(N , N);
    replicationIntMask = convolve2DWithWrap(lattice, replMatrix); // empty cells with replicating neighbours


    Eigen::Tensor<int, 2> replicationSpots = (replicationIntMask > 0).cast<int>() * diedIntMask;


    Eigen::Tensor<int, 2> new_lattice = (replicationSpots > 0).select(fillNewLattice(lattice, neighbors), lattice);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (replicationSpots(i, j) == 1) {
                lattice(i, j) = new_lattice(i, j);
            }
        }
    }
    std::cout << "replicationSpots" << replicationSpots.dimensions() << std::endl;
    std::cout << replicationSpots << std::endl;

    return lattice;
    
}

void display(GLFWwindow* window) {
    Eigen::Tensor<int, 2> replMatrix(3, 3);
    replMatrix.setValues({ {0, 1, 0}, {1, 0, 1}, {0, 1, 0} });

    Eigen::Tensor<int, 2> lattice(N, N);
    lattice = initLattice(lattice, nSpecies);
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
        glTranslatef(-WINDOW_WIDTH / 2, -WINDOW_HEIGHT / 2, -1000);
        DrawMatrix(lattice);
        lattice = runSimulationStep(lattice, replMatrix);

        // Update Screen
        glfwSwapBuffers(window);

        // Check for any input, or window movement
        glfwPollEvents();
    }
}

int main() {


    GLFWwindow* window = initWindow(WINDOW_WIDTH, WINDOW_HEIGHT);

    if (NULL != window)
    {
        display(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}