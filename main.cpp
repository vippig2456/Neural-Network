#include <iostream>
#include "network-mk2.hpp"
int main() {
    int layers[3] = {8, 8, 1};
    network neural_net(3, layers, 0.01, 0.01);
    double input[8] = {0, 1, 1, 1, 0, 1, 0, 0};
    double targets[1] = {64};
    for (int i = 0; i < 1000; i++) {
        neural_net.trainingLoop(targets, input);
    }
    neural_net.computer(input);
    std::cout << neural_net.neurons[2][0];
    //for (int i = 0; i < 8; i++) {
    //    for (int g = 0; g < 8; g++) {
    //        std::cout << neural_net.weights[0][i][g] << '\n';
    //    }
    //}
}