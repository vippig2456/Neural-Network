#include <cmath>
#include <string>
#include <random>


double alpha = 0.01;

// TO DO: fix the runtime error with the deletion of nablaW

class network{
    public:
        double** neurons; // the values of the neurons
        double*** weights; 
        double** biases; // the biases
        int* widths; // the width of each layer
        int length; // the length of the network
        double** delta; // delta for backpropagation, note that there is no delta for the first layer
        double** z; // z values of all layers minus the first layer
        double*** nablaW; // Derivative of the cost function with respect to each weight
        double** nablaB; // Derivative of the cost function with respect to the biases
        double eta; // learning rate
        double meanCost; // the cost of the network

        double (*activation)(double); // the activation function pointer
        double (*activationDerivative)(double); // the activation function Derivative function function pointer
        double (*cost)(double* , double* , int ); // the function pointer for the cost function
        void (*costDerivative)(double*, double*, double*, int); // the function pointer for the Derivative of the cost function
        network(int l, int* lWidth, double _eta, double (*activationFunction)(double), double (*activationFunctionDerivative)(double), double (*costFunction)(double* , double* , int), void (*costFunctionDerivative)(double*, double*, double*, int)){
            activation = activationFunction;
            activationDerivative = activationFunctionDerivative;
            cost = costFunction;
            costDerivative = costFunctionDerivative;

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
            memcpy(widths, lWidth, l*sizeof(int));
            z = new double*[length - 1];
            for(int i = 0; i < length-1; i++) {
                z[i] = new double[widths[i+1]];
            }

            nablaB = new double*[l-1];
            for(int i = 0; i < l-1; i++){
                nablaB[i] = new double[widths[i+1]];
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
            weights = new double**[l-1]; 
            for (int i = 0; i < l-1; i++) { //cycling through the layers
                weights[i] = new double*[lWidth[i]];
                for (int j = 0; j < lWidth[i]; j++) { //cycling through the lower layer
                    weights[i][j] = new double[lWidth[i+1]];
                    for (int h = 0; h < lWidth[i+1]; h++) { //cycling thourgh top layer
                        weights[i][j][h] = normalDist(gen); //assigning weights
                    } //weights[l][j][k] is the weight from the jth neuron in the  lth layer to the kth neuron in the l+1th layer
                }
            }
            return; 
        }
        
        double* computer(double* inputs){
            memcpy(neurons[0], inputs, widths[0]*sizeof(double)); //copy the inputs to the first layer of the neuron
            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i+1]; j++){
                    neurons[i+1][j] = 0;
                    for(int k = 0; k < widths[i]; k++){
                        neurons[i+1][j] += neurons[i][k]*weights[i][k][j]; // sum weigthted neurons
                    }
                    neurons[i+1][j] += biases[i][j]; // apply biases
                    neurons[i+1][j] = activation(neurons[i+1][j]); // apply activations
                }
            }
            return neurons[length-1];
        }

        void compute(double* inputs) { // computer but for training loop
            memcpy(neurons[0], inputs, widths[0]*sizeof(double)); //copy the inputs to the first layer of the neuron
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

        void backpropagation(double* desiredOutput){
            costDerivative(desiredOutput, neurons[length-1], delta[length-2], widths[length-1]); // put the error Derivative from the cost function into the last layer of delta
            for(int i = 0; i < widths[length-1]; i++){
                delta[length-2][i] *= activationDerivative(z[length-2][i]); // multiply by the gradient of the z score for each varable
            }
            for(int i = length-3; i > -1; i--){
                for(int j = 0; j < widths[i+1]; j++){
                    delta[i][j] = 0; // set the current neurons error to 0 to sum up the weighted inputs
                    for(int k = 0; k < widths[i+2]; k++){
                        delta[i][j] += weights[i+1][j][k] * delta[i+1][k]; // sum the weigths times delta from the next layer
                    }
                    delta[i][j] *= activationDerivative(z[i][j]); // multiply by the gradient of the z score at the current neuron
                }
            }
            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i]; j++){
                    for(int k = 0; k < widths[i+1]; k++){
                        nablaW[i][j][k] += neurons[i][j]*delta[i][k]; // nablaW equals the activation of the jth neuron the of previous lay times the error of the kth neuron on the current layer.
                    }
                }
                for(int j = 0; j < widths[i+1]; j++){
                    nablaB[i][j] += delta[i][j]; // the gradient for the biases is the same as the error(the error being the gradient of the cost function(as the cost function should be minimised when its gradient is equal to 0))
                }
            }
            return;
        }

        void changeParameters() {
            for (int l = 0; l < length - 1; l++) {
                for (int j = 0; j < widths[l+1]; j++) {
                    biases[l][j] -= eta * delta[l][j];
                    for (int k = 0; k < widths[l]; k++) {
                        weights[l][k][j] -= eta * nablaW[l][k][j];
                    }
                }
            }
        }

        void trainingStep(double* targets, double* inputs) {
            meanCost = 0;
            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i]; j++){
                    for(int k = 0; k < widths[i+1]; k++){
                        nablaW[i][j][k] = 0;
                    }
                }
                for(int j = 0; j < widths[i+1]; j++){
                    nablaB[i][j] = 0;
                }
            }
            compute(inputs);
            backpropagation(targets);
            meanCost = cost(targets, neurons[length-1], widths[length-1]);
            changeParameters();
            return;
        }

        void trainingStep(double** targets, double** inputs, int numTargets) {
            meanCost = 0;
            for(int i = 0; i < length-1; i++){
                for(int j = 0; j < widths[i]; j++){
                    for(int k = 0; k < widths[i+1]; k++){
                        nablaW[i][j][k] = 0;
                    }
                }
                for(int j = 0; j < widths[i+1]; j++){
                    nablaB[i][j] = 0;
                }
            }
            for(int i = 0;i < numTargets; i++){
                compute(inputs[i]);
                backpropagation(targets[i]); // loop though the backpropagation to sum the bias and weight gradients
                meanCost += cost(targets[i], neurons[length-1], widths[length-1]);
            }
            for(int i = 0; i < length-1; i++){ // average the gradients of the weighs and biases across the different training exampels.
                for(int j = 0; j < widths[i]; j++){
                    nablaB[i][j] /= numTargets;
                    for(int k = 0; k < widths[i+1]; k++){
                        nablaW[i][j][k] /= numTargets;
                    }
                }
            }
            meanCost /= numTargets; // divide the sum of the costs to make it the mean cost
            changeParameters();
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
                delete[] nablaW[i];
            }
            delete[] nablaW;

            for(int i = 0; i < length - 1; i++) {
                delete[] nablaB[i];
            }
            delete[] nablaB;

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
    if(input >= 0){
        return 1;
    }
    return alpha;
}

double sigmoid(double input){
    return 1/(1+exp(-input));
}

double dSigmoid(double input){
    double s = sigmoid(input);
    return s*(1-s);
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

double quadraticCost(double* desiredOutput, double* activations, int arrayLength){
    double lengthSqrd = 0;
    for(int i = 0; i < arrayLength; i++){
        lengthSqrd += (desiredOutput[i] - activations[i])*(desiredOutput[i] - activations[i]); // the square of the difference bewtween the desired output(output) and the accual output(activations)
    }
    return lengthSqrd; // the length squared cancels out the square root needed to find the length of a vector leaving the sum of the squares(lengthSqrd)
}

void dQuadraticCost(double* desiredOutput, double* activations, double* output, int length){
    for(int i = 0; i < length; i++){
        output[i] = 2*(activations[i]-desiredOutput[i]);
    }
    return;
}