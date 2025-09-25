#include <iostream>
#include "network-mk2.hpp"

double learn(double input){
    if(input < 1){
        return 0.0005;
    } 
    if(input < 10000){
        return 0.001;
    }
    return 0.01;
}

int main() {
    double cost;
    std::string file = "./8-bit-binary.dat";
    int layers[3] = {8, 8, 1};
    network neural_net(file, leakyRelu, dLeakyRelu, quadraticCost, dQuadraticCost, learn);
    std::cout << "length " << neural_net.length << "widths 1 " << neural_net.widths[2] << "\n";
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
    for(int i = 0; i < 1000; i++){
        neural_net.trainingStep(targets, inputs, 256);
        std::cout << i+1 << " Cost: " << neural_net.meanCost <<"\n";
    }
    neural_net.writeToFile();
}