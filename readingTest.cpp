#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>

int main(){
    std::string line;
    std::fstream file("./8-bit-binary.dat");
    if(!file){
        std::cout << "file error";
    }
    std::getline(file, line, ' ');
    int length = std::stoi(line); // change the string to a int and store it in length the first number in the file is the length

    int* widths = new int[length];
    for(int i = 0; i < length; i++){
        std::getline(file, line, ' '); // get the next number(the width of that layer)
        widths[i] = std::stoi(line); // change the string to a int and store it as the layer width
    }
    std::getline(file, line, '\n');
    double*** weights = new double**[length-1];
    for(int i = 0; i < length-1; i++){
        weights[i] = new double*[widths[i]];
        for(int j = 0; j < widths[i]; j++){
            weights[i][j] = new double[widths[i+1]];
            for(int k = 0; k < widths[i+1]; k++){
                std::getline(file, line, ',');
                std::cout << line << ", ";
                weights[i][j][k] = std::stod(line);
            }
        }
    }
    std::getline(file, line, '\n');
    std::cout << "\n";
    double** biases = new double*[length-1];
    for(int i = 0; i < length-1; i++){
        biases[i] = new double[widths[i+1]]; 
        for(int j = 0; j < widths[i+1]; j++){
            std::getline(file, line, ',');
            std::cout << line << ", ";
            biases[i][j] = std::stod(line);
        }
    }
    return 0;
}