#include <iostream>
#include "network-mk2.hpp"

double learn(double input){
    if(input < 1000){
        return 0.0007;
    } 
    if(input < 10000){
        return 0.001;
    }
    return 0.01;
}

int main() {
    double cost;
    int layers[3] = {8, 8, 1};
    network neural_net(3, layers, leakyRelu, dLeakyRelu, quadraticCost, dQuadraticCost, learn);
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
        neural_net.trainingStep(targets, inputs, 256);
        std::cout << i+1 << " Cost: " << neural_net.meanCost <<"\n";
    }
}