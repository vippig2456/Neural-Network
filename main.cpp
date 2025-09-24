#include <iostream>
#include "network-mk2.hpp"
int main() {
    double cost;
    int layers[3] = {8, 8, 1};
    network neural_net(3, layers, 0.01, leakyRelu, dLeakyRelu, quadraticCost, dQuadraticCost);
    double input[8] = {0, 1, 1, 1, 0, 1, 0, 0};
    double targets[1] = {64};
    for (int i = 0; i < 50; i++) {
        neural_net.trainingStep(targets, input);
        cost = (targets[0]-neural_net.neurons[2][0])*(targets[0]-neural_net.neurons[2][0]);
        std::cout << "cost: " << cost << " output: " << neural_net.neurons[2][0] << '\n';
    }
}