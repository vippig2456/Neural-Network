#include <cmath>
#include <string>
#include <random>

// TODO: chnage all of the activation fuinctions in the code to the activation function pointer

double alpha = 0.01;

class network{
    public:
        double** neurons; // the values of the neurons
        double*** weights; 
        double** biases; // the biases
        int* widths; // the width of each layer
        int length; // the length of the network
        double** delta; // delta for backpropagation, note that there is no delta for the first layer
        double** z; // z values of all layers minus the first layer
        double*** nablaW; // derivative of the cost function with respect to each weight
        double eta; // learning rate

        double (*activation)(double); // the activation function pointer
        double (*activationDerivitive)(double); // the activation function derivitive function function pointer
        network(int l, int* lWidth, double _eta, double (*activationFunction)(double), double (*activationFunctionDerivitive)(double)){
            activation = activationFunction;
            activationDerivitive = activationFunctionDerivitive;

            delta = new double*[l - 1]; 
            for(int i = 0; i < l - 1; i++) { // cycling through lower layers
                delta[i] = new double[lWidth[i + 1]];
            }
            eta = _eta;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> normalDist(0, 1); // make a normal distribution weith a mean of 0 and sd of 1
            length = l;
            widths = new int[l];
            memcpy(widths, lWidth, l*4);
            z = new double*[length - 1];
            for(int i = 0; i < length-1; i++) {
                z[i] = new double[widths[i+1]];
            }
            nablaW = new double**[length-1];
            for (int i = 0; i < length-1; i++) {
                nablaW[i] = new double*[widths[i]];
                for (int k = 0; k < widths[i]; k++) {
                    nablaW[i][k] = new double[widths[i+1]];
                }
            }
            neurons = new double*[l];
            for(int i = 0; i < l; i++){
                neurons[i] = new double[lWidth[i]];
            }
            biases = new double*[l-1]; //REMEMBER WHEN INDEXING THAT BIASES HAVE ONE LESS LAYER THAN THE NEURONS...
            for (int i = 0; i < l-1; i ++) {
                biases[i] = new double[lWidth[i+1]];
                for(int j = 0; j < lWidth[i+1]; j++){
                    biases[i][j] = normalDist(gen);
                }
            }
            weights = new double**[l]; 
            for (int i = 0; i < l-1; i++) { //cycling through the layers
                weights[i] = new double*[lWidth[i]];
                for (int j = 0; j < lWidth[i]; j++) { //cycling through the lower layer
                    weights[i][j] = new double[lWidth[i+1]];
                    for (int h = 0; h < lWidth[i+1]; h++) { //cycling thourgh top layer
                        weights[i][j][h] = normalDist(gen); //assigning weights
                    } //weights[l][j][k] is the weight from the jth neuron in the  lth layer to the koth neuron in the l+1th layer
                }
            }
            return; 
        }
        
        double* computer(double* inputs){
            memcpy(neurons[0], inputs, widths[0]*8); //copy the inputs to the first layer of the neuron
            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i+1]; j++){
                    neurons[i+1][j] = 0;
                    for(int k = 0; k < widths[i]; k++){
                        neurons[i+1][j] += neurons[i][k]*weights[i][k][j];
                    }
                    neurons[i+1][j] += biases[i][j];
                    neurons[i+1][j] = activation(neurons[i+1][j]);
                }
            }
            return neurons[length-1];
        }

        void compute(double* inputs) { // computer but for training loop
            memcpy(neurons[0], inputs, widths[0]*8); //copy the inputs to the first layer of the neuron
            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i+1]; j++){
                    neurons[i+1][j] = 0;
                    for(int k = 0; k < widths[i]; k++){
                        neurons[i+1][j] += neurons[i][k]*weights[i][k][j];
                    }
                    neurons[i+1][j] += biases[i][j];
                    z[i][j] = neurons[i+1][j];
                    neurons[i+1][j] = activation(neurons[i+1][j]);
                }
            }
            return;
        }

        void backpropagation(double* targets) { // returns the derivatives of the cost function with repect to each weight and bias, given the activations and targets
            for(int k = 0; k < widths[length-1]; k++) {
                if (neurons[length-1][k] >= 0) {
                    delta[length-2][k] = 2*(neurons[length-1][k] - targets[k]);
                } else {
                    delta[length-2][k] = 0;
                }
            }
            for(int l = length - 3; l > -1; l--) { // cycling through lower layers
                delta[l] = new double[widths[l + 1]];
                for(int j = 0; j < widths[l+1]; j++) { // cycling through neurons for delta on the current layer
                    delta[l][j] = 0;
                    for (int k = 0; k < widths[l+2]; k++) {
                        if (z[l][j] >= 0) {
                            delta[l][j] += weights[l+1][k][j] * delta[l+1][k];
                        } else {
                            delta[l][j] = 0;
                        }
                    }
                }
            }
            for (int l = 0; l < length - 1; l++) {
                for (int k = 0; k < widths[l]; k++) {
                    for (int j = 0; j < widths[l+1]; j++) {
                        nablaW[l][k][j] = delta[l][j] * relu(z[l][k]);
                    }
                }
            }
            return;
        }

        void change_parameters() {
            for (int l = 0; l < length - 1; l++) {
                for (int j = 0; j < widths[l+1]; j++) {
                    biases[l][j] -= eta * delta[l][j];
                    for (int k = 0; k < widths[l]; k++) {
                        weights[l][k][j] -= eta * nablaW[l][k][j];
                    }
                }
            }
        }

        void trainingLoop(double* targets, double* inputs) {
            compute(inputs);
            backpropagation(targets);
            change_parameters();
            return;
        }

        ~network(){
            for(int i = 0; i < length; i++){
                delete[] neurons[i];
            }
            delete[] neurons;

            for(int i = 0; i < length-1; i++){
                delete[] biases[i];
            }
            delete[] biases;

            for(int i = 0; i < length-1; i++){
                delete[] z[i];
            }
            delete[] z;     

            for(int i = 0; i < length - 1; i++) {
                delete[] delta[i];
            }
            delete[] delta;


            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i]; j++){
                    delete[] weights[i][j];
                }
                delete weights[i];
            }
            delete[] weights;

            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i]; j++){
                    delete[] nablaW[i][j];
                }
                delete nablaW[i];
            }
            delete[] nablaW;

            delete[] widths;
            return;
        }
};

double relu(double input){
    if(input >= 0){
        return input;
    }
    return 0;
}

double dRelu(double input){
    if(input > 0){
        return 1;
    }
    return 0;
}

double leakyRelu(double input){
    if(input >= 0){
        return input;
    }
    return input*alpha;
}

double dLeakyRelu(double input){
    if(input > 0){
        return 1;
    }
    return alpha;
}

double sigmoid(double input){
    return 1/(1+pow(2.7182818, input));
}

double dSigmoid(double input){
    return sigmoid(input)*(1-sigmoid(input));
}

double softSign(double input){
    return input/(1+abs(input));
}

double dSoftSign(double input){
    return 1/((abs(input)+1)*(abs(input)+1));
}

double gaussian(double input){
    return pow(2.7182818, -(input*input));
}

double dGaussian(double input){
    return -2*input*pow(2.7182818, -(input*input));
}