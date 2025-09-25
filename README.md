# Neural-Network
A bad neural network C++ header made just for fun 
## File Structure
The first three numbers in a file are followed by a space to use as a delimiter a '\n' is used after to signify a new thing.
The second line in the file is the weights with the weights being stored in a flattened array with the elements following this loop:
'''
for(int i = 0; i < length-1; i++){
    for(int j = 0; j < widths[i]; j++){
        for(int k = 0; k < widths[i+1]; k++){
            outputFile << weights[i][j][k] << ','; **output the weights in a flattened state**
        }
    }
}
'''