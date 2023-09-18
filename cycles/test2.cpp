#include <iostream>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
//#include "helpers.h"
#include "gpt.h"

// on linux compile with: 
// /usr/bin/gcc -fdiagnostics-color=always -g /home/k/cycles/cycles/test2.cpp -o /home/k/cycles/cycles/test2 --include-directory=/usr/include/eigen3  -lGL -lGLU -lglfw  -lm -lstdc++

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const double fpsLimit = 1.0 / 60.0;
double lastUpdateTime = 0;  // number of seconds since the last loop
double lastFrameTime = 0;   // number of seconds since the last frame



const int wDead = 37;
const int wRepl = 5;
const int wBoost = 400;
const float pDeath = 0.2;



const int DEBUG = 0;
const int N = 250;
const int nSpecies = 9;
// if (DEBUG == 1) {

// } else {
//     const int N = 350;
//     const int nSpecies = 9;
// }
int pbc(int x, int N){
    if (x < 0) {
        return int(N - x - 1);
    }
    if (x >= N) {
        return int(x - N);
    }
    return int(x);
}
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
    glfwWindowHint(GLFW_SAMPLES, 4); // 1x antialiasing

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
   // glEnable(GL_LIGHTING);
   // glEnable(GL_LIGHT0);
    glShadeModel(GL_SMOOTH);
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

    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            int species = lattice(i, j);
            // Set color based on species
            switch (species) {
                case 0: glColor4f(0.1f, 0.1f, 0.1f, 0.1f); break; // Dead
                case 1: glColor4f(0.0f, 0.0f, 0.5f, 0.0f); break;
                case 2: glColor4f(0.0f, 0.00196078431372549f, 1.0f, 0.125f); break;
                case 3: glColor4f(0.0f, 0.503921568627451f, 1.0f, 0.25f); break;
                case 4: glColor4f(0.08538899430740036f, 1.0f, 0.8823529411764706f, 0.375f); break;
                case 5: glColor4f(0.4901960784313725f, 1.0f, 0.4775458570524984f, 0.5f); break;
                case 6: glColor4f(0.8950031625553446f, 1.0f, 0.07273877292852626f, 0.625f); break;
                case 7: glColor4f(1.0f, 0.5816993464052289f, 0.0f, 0.75f); break;
                case 8: glColor4f(1.0f, 0.11692084241103862f, 0.0f, 0.875f); break;
                case 9: glColor4f(0.5f, 0.0f, 0.0f, 1.0f); break;

                default: glColor4f(1.0f, 1.0f, 1.0f, 1.0f);      // Others
            }


            float x = j * cellWidth, y = i * cellHeight;
            /*  heightA = (height(x-1, y-1) + height(x-1, y) + height(x, y-1) + height(x, y))/4
                heightB = (height(x-1, y) + height(x-1, y+1) + height(x, y) + height(x, y+1))/4
                heightC = (height(x, y) + height(x, y+1) + height(x+1, y) + height(x+1, y+1))/4
                heightD = (height(x, y-1) + height(x, y) + height(x+1, y-1) + height(x+1, y))/4
*/
            float zHeight = 0; // float(N) / 100;
            float z1 = zHeight * ( 
                lattice(i-1, j-1) +
                lattice(i-1, j) +
                lattice(i, j-1) + 
                species
                ) / 4;
            float z2 = zHeight * (
                lattice( i-1 ,  j ) +
                lattice( i-1 ,  j+1 ) +
                lattice( i ,  j+1 ) + 
                species
                ) / 4;
            float z3 = zHeight * (
                lattice( i ,  j+1 ) +
                lattice( i+1 ,  j ) +
                lattice( i+1 ,  j+1 ) + 
                species
                ) / 4;
            float z4 = zHeight * (
                lattice( i ,  j-1 ) +
                lattice( i+1 ,  j-1 ) +
                lattice( i+1 ,  j ) + 
                species
                ) / 4;
            glBegin(GL_QUADS);
            glVertex3f(x, y, z1); // D-L
            glVertex3f(x, y + cellHeight, z2); // U-L
            glVertex3f(x + cellWidth, y + cellHeight, z3); // U-R
            glVertex3f(x + cellWidth, y, z4); // D-R
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
                    onehotTensor(i, j, speciesId) = a;
                }
            }
        }
    }

    return onehotTensor;
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
    int Rsq = N*N/4 / 4;
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
    Eigen::Tensor<int, 2> up = roll(lattice, 1, 0);
    Eigen::Tensor<int, 2> down = roll(lattice, -1, 0);
    Eigen::Tensor<int, 2> right = roll(lattice, 0, -1);
    Eigen::Tensor<int, 2> left = roll(lattice, 0, 1);

    Eigen::Tensor<int, 2> up_right = roll(up, 0, -1);
    Eigen::Tensor<int, 2> up_left = roll(up, 0, 1);
    Eigen::Tensor<int, 2> down_right = roll(down, 0, -1);
    Eigen::Tensor<int, 2> down_left = roll(down, 0, 1);

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


        claims += onehotMult(replicatorSpecies, nSpecies, wRepl);
        if (DEBUG == 1) {
            std::cout << "claims after replication" << std::endl << claims << std::endl;
        }


        if (DEBUG == 1) {
            std::cout << "replicator species " << std::endl << replicatorSpecies << std::endl;
            std::cout << "claims" << std::endl << claims << std::endl;
        }



        Eigen::Tensor<int, 2> boosterSpeciesRequired = replicatorSpecies + 1; // boost comes from another species

        // replace the non-existant nSpecies+1 booster with the correct 1
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (boosterSpeciesRequired(i, j) > nSpecies) {
                    boosterSpeciesRequired(i, j) = 1;
                }
            }
        }
        for (int boostDirectionId : boostLogic[replicationDirectionId]) {
            Eigen::Tensor<int, 2> boosterSpecies = neighborsMatrix[boostDirectionId]; // potential boosters
            Eigen::Tensor<int, 2> whereAreCorrectBoosters = (boosterSpecies == boosterSpeciesRequired).cast<int>();
            Eigen::Tensor<int, 2> correctBoosterSpecies = replicatorSpecies * whereAreCorrectBoosters;
            claims += onehotMult(correctBoosterSpecies, nSpecies, wBoost);
            if (DEBUG == 1) {
                std::cout << "BoosterSpecies " << std::endl << boosterSpecies << std::endl;

                std::cout << "correctBoosterSpecies" << std::endl << correctBoosterSpecies << std::endl;
                std::cout << "claims" << std::endl << claims << std::endl;
            }
        }
 
    }
    

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            claims(i, j, 0) = wDead;
        }
    }
    if (DEBUG == 1) {
        std::cout << "claims after boosts" << std::endl << claims << std::endl;
    }
    // Normalize claims and pick new cells
    Eigen::Tensor<int, 3> cumSum = claims.cumsum(2);

    Eigen::array<int, 1> dims({ 2 });
    Eigen::Tensor<int, 2> claimsSum = claims.sum(dims);
    if (DEBUG == 1) {
        std::cout << "claimsSum " << std::endl << claimsSum << std::endl;
    }
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
    if (DEBUG == 1) {
        std::cout << "cumProb" << std::endl;
        std::cout << cumProb << std::endl;
    }
    Eigen::Tensor<float, 2> randomMatrix(N, N);
    randomMatrix.setRandom();
    //randomMatrix = randomMatrix * 0.5f + 0.5f;// Values between [0,1]

    if (DEBUG == 1) {
        std::cout << "randomMatrix" << std::endl;
        std::cout << randomMatrix << std::endl;
    }

    Eigen::Tensor<int, 2> newLattice(N, N);
    newLattice.setZero();
    // find max along the last (i.e. species) axis
  
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < nSpecies+1; k++) {
                if (cumProb(i, j, k) > randomMatrix(i, j)) {
                    newLattice(i, j) = k;
                    break;
                }
            }
        }
    }
    if (DEBUG == 1) {
        std::cout << "newLattice" << std::endl << newLattice << std::endl;
    }
    return newLattice;
}


Eigen::Tensor<int, 2> runSimulationStep(Eigen::Tensor<int, 2> lattice, Eigen::Tensor<int, 2> replMatrix) {

    if (DEBUG == 1) {
        std::cout << "lattice before step" << std::endl;
        std::cout << lattice << std::endl;
    }

    Eigen::Tensor<float, 2> zeroMatrix(N, N);
    zeroMatrix.setZero();
    Eigen::Tensor<float, 2> randomMatrix(N, N);
    randomMatrix.setRandom();
    //randomMatrix = randomMatrix * 0.5f + 0.5f;// Values between [0,1]
    Eigen::Tensor<bool, 2> tmpSurvivedIntMask = randomMatrix > pDeath;
    Eigen::Tensor<int, 2> survivedIntMask = (randomMatrix > pDeath).cast<int>() * (lattice > 0).cast<int>();
    Eigen::Tensor<int, 2> diedIntMask = (randomMatrix <= pDeath).cast<int>() * (lattice > 0).cast<int>();
    lattice = lattice * survivedIntMask;
    std::vector<Eigen::Tensor<int, 2>> neighbors = computeNeighbours(lattice);

    if (DEBUG == 1) {
        std::cout << "lattice after dying"  << std::endl;
        std::cout << lattice << std::endl;

        std::cout << "neighbors"  << std::endl;
        for (int i = 0; i < neighbors.size(); i++) {
            std::cout << neighbors[i] << std::endl << std::endl;
        }
    }

    Eigen::Tensor<int, 2> replicationIntMask(N , N);
    replicationIntMask = convolve2DWithWrap(lattice, replMatrix); // empty cells with replicating neighbours

    Eigen::Tensor<int, 2> replicationSpots = (replicationIntMask > 0).cast<int>() * (lattice == 0).cast<int>();


    Eigen::Tensor<int, 2> new_lattice = (replicationSpots > 0).select(
        fillNewLattice(lattice, neighbors), 
        lattice
        );
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (replicationSpots(i, j) == 1) {
                lattice(i, j) = new_lattice(i, j);
            }
        }
    }

    if (DEBUG == 1) {

 

        std::cout << "greater than death" << std::endl;
        std::cout << randomMatrix << std::endl << ">" << pDeath << std::endl;
        std::cout <<  tmpSurvivedIntMask << std::endl;
    
        std::cout << "survived mask" << std::endl;
        std::cout << survivedIntMask << std::endl;
        std::cout << "died mask" << std::endl;
        std::cout << diedIntMask << std::endl;
        std::cout << "replication mask" << std::endl;
        std::cout << replicationIntMask << std::endl;

        std::cout << "replicationSpots" << std::endl;
        std::cout << replicationSpots << std::endl;

        std::cout << "lattice after step" << std::endl;
        std::cout << lattice << std::endl;
    }


    return lattice;
    
}

void display(GLFWwindow* window) {
    Eigen::Tensor<int, 2> replMatrix(3, 3);
    replMatrix.setValues({ {0, 1, 0}, {1, 0, 1}, {0, 1, 0} });

    int filterSize = 7;
    Eigen::Tensor<int, 2> avgFilter(filterSize, filterSize);
    avgFilter.setConstant(1);
    avgFilter.setValues(
        {
            {0,1,1,1,1,1,0},
            {1,1,2,3,2,1,1},
            {1,2,4,5,4,2,1},
            {1,3,5,9,5,3,1},
            {1,2,4,5,4,2,1},
            {1,1,2,3,2,2,1},
            {0,1,1,1,1,1,0},
        }
    );

    // int filterSize = 3;
    // Eigen::Tensor<int, 2> avgFilter(filterSize, filterSize);
    // avgFilter.setValues(
    //     {
    //         {1,1,1},
    //         {1,5,1},
    //         {1,1,1},
    //     }
    // );

    Eigen::Tensor<int, 0> filterSum = avgFilter.sum();



    Eigen::Tensor<int, 2> lattice(N, N);
    Eigen::Tensor<int, 2> latticePrev(N, N);
    Eigen::Tensor<int, 2> latticePrevPrev(N, N);

    Eigen::Tensor<int, 2> latticeToDraw(N, N);
    lattice = initLattice(lattice, nSpecies);



    int step = 0;
    int maxSteps = 5;
    int frame = 0;




    while (!glfwWindowShouldClose(window))
    {
        double now = glfwGetTime();
        double deltaTime = now - lastUpdateTime;
        // Scale to window size
        GLint WINDOW_WIDTH, WINDOW_HEIGHT;
        glfwGetWindowSize(window, &WINDOW_WIDTH, &WINDOW_HEIGHT);
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

        // Draw stuff
        glClearColor(0.3, 0.3, 0.3, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //glMatrixMode(GL_PROJECTION_MATRIX);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glLoadIdentity();
        //gluPerspective(60, (double)WINDOW_WIDTH / (double)WINDOW_HEIGHT, 0.1, 2000);
        //glOrtho(-WINDOW_WIDTH  /2,  WINDOW_WIDTH /2, -WINDOW_HEIGHT /2, WINDOW_HEIGHT /2, 0.0, 2000);
        glOrtho(0,  WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0.0, 2000);

        glMatrixMode(GL_MODELVIEW_MATRIX);

        //glTranslatef(-WINDOW_WIDTH / 2, -WINDOW_HEIGHT / 2, -2*N);



        // Update Screen
        if ((now - lastFrameTime) >= fpsLimit) {

            lattice = runSimulationStep(lattice, replMatrix);


            latticeToDraw = ((latticePrevPrev != 0) && (lattice == 0)).select(latticePrevPrev, lattice);
            latticeToDraw = convolve2DWithWrap(latticeToDraw, avgFilter) / filterSum(0);
          //  latticeToDraw = ((latticePrev != 0) && (latticeToDraw == 0)).select(latticePrev, latticeToDraw);
            // Draw
           
            //DrawMatrix(latticeToDraw);
            DrawMatrix(lattice);

            frame++;
            glfwSwapBuffers(window);

            if (frame > 1) {
                latticePrevPrev = lattice;
            }
            latticePrev = latticeToDraw;

            lastFrameTime = now;
        }

        // set lastUpdateTime every iteration
        lastUpdateTime = now;

        // Check for any input, or window movement
        glfwPollEvents();
        if (DEBUG == 1) {
            if (step > maxSteps) {break;}
        }
        
        step++;
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