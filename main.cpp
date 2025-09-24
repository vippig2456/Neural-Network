#include <iostream>
#include "network-mk2.hpp"
int main() {
    double cost;
    int layers[3] = {8, 8, 1};
    network neural_net(3, layers, 0.01, leakyRelu, dLeakyRelu, quadraticCost, dQuadraticCost);
    double** inputs = new double*[256]; // an array for all of the different values for a 8 bit intiger
    double** targets = new double*[256]; // the targets for the array
    for(int i = 0; i < 256; i++){
        inputs[i] = new double[8];
        targets[i] = new double[1];
        targets[i][0] = i;
        for(int j = 0; j < 8; j++){
            inputs[i][j] = ((i & (1<<j)) != 0); // returns the bit at position j for the number i
        }
    }
    for(int i = 0; i < 50; i++){
        neural_net.trainingStep(targets[54], inputs[54]);
        std::cout << "Cost: " << neural_net.meanCost << "calculated cost: " << (neural_net.neurons[2][0]-targets[54][0])*(neural_net.neurons[2][0]-targets[54][0]) <<"\n";
    }
}