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
    double inputs[8];
    std::string file = "./8-bit-binary.dat";
    int layers[3] = {8, 8, 1};
    network neural_net(file, leakyRelu, dLeakyRelu, quadraticCost, dQuadraticCost, learn);
    while(1){
        for(int i = 0; i < layers[0]; i++){
            std::cout << "Input neuron " << i << " value: ";
            std::cin >> inputs[i];
        }
        neural_net.compute(inputs);
        std::cout << "Output is: " << (int)(neural_net.neurons[2][0]*255+0.5) << '\n';
    }
}