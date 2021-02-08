#include <array>
#include <ctime>
#include <cmath>
#include <iostream>
#include <random>

#define TRAIN_SAMPLES 2
#define INPUT_SIZE 2+1
#define HIDDEN_NEURONS 2
#define OUTPUT_NEURONS 1

using std::array;
using std::cout;
using std::endl;

// init random engine for uniformly distributed doubles in [-1, 1]
std::default_random_engine generator(static_cast<unsigned>(time(nullptr)));
static std::uniform_real_distribution<double> distribution(-1, 1);

// declaration of arrays that store intermediate and final results
array<array<double, INPUT_SIZE>, TRAIN_SAMPLES> train_samples;
array<array<double, INPUT_SIZE>, HIDDEN_NEURONS> l1_weights;
array<double, HIDDEN_NEURONS> l1_weighted_sums;
array<double, HIDDEN_NEURONS+1> l1_outputs;
array<array<double, HIDDEN_NEURONS+1>, OUTPUT_NEURONS> l2_weights;
array<double, OUTPUT_NEURONS> l2_weighted_sums;
array<double, OUTPUT_NEURONS> l2_outputs;

// generate random training set
void init_train_set(){

    for (int i=0; i<TRAIN_SAMPLES; i++){
        for (int j=0; j<INPUT_SIZE-1; j++){
            train_samples[i][j] = distribution(generator);
        }
        train_samples[i][INPUT_SIZE-1] = 1;  // bias coefficient
    }

}


// init neuron weights and biases
void init_network(){

    // layer 1 (hidden layer)
    for (int i=0; i<HIDDEN_NEURONS; i++){
        for (int j=0; j<INPUT_SIZE; j++){
            l1_weights[i][j] = distribution(generator);
        }
    }

    // layer 2 (output layer)
    for (int i=0; i<OUTPUT_NEURONS; i++){
        for (int j=0; j<=HIDDEN_NEURONS; j++){
            l2_weights[i][j] = distribution(generator);
        }
    }

    // level 2 bias coefficient (always equal to 1)
    l1_outputs[HIDDEN_NEURONS] = 1.0;

}

// activation function
double sigmoid(double x){

    return 1 / (1 + std::exp(-x));

}

void forward_pass(int sample){

    // zero-init/clear weighted sum arrays
    for (int i=0; i<HIDDEN_NEURONS; i++){
        l1_weighted_sums[i] = 0;
    }
    for (int i=0; i<OUTPUT_NEURONS; i++){
        l2_weighted_sums[i] = 0;
    }

    // layer 1 (hidden layer)
    for (int neuron=0; neuron<HIDDEN_NEURONS; neuron++){  // for each l1 neuron
        // calculate weighted sum
        for (int synapse=0; synapse<INPUT_SIZE; synapse++){
            l1_weighted_sums[neuron] += l1_weights[neuron][synapse] * train_samples[sample][synapse];
        }
        l1_outputs[neuron] = sigmoid(l1_weighted_sums[neuron]);  // neuron activation
    }

    // layer 2 (output layer)
    for (int neuron=0; neuron<OUTPUT_NEURONS; neuron++){  // for each l2 neuron
        // calculate weighted sum
        for (int synapse=0; synapse<=HIDDEN_NEURONS; synapse++){
            l2_weighted_sums[neuron] += l2_weights[neuron][synapse] * l1_outputs[synapse];
        }
        l2_outputs[neuron] = sigmoid(l2_weighted_sums[neuron]);  // neuron activation
    }

}

int main(){

    init_train_set();
    init_network();

    cout<<"Training set:"<<endl;
    for (int i=0; i<TRAIN_SAMPLES; i++){
        cout<<"[ ";
        for (int j=0; j<INPUT_SIZE; j++){
            cout<<train_samples[i][j]<<" ";
        }
        cout<<"]'"<<endl;
    }

    cout<<endl<<"L1 neuron weights and biases:"<<endl;
    for (int i=0; i<HIDDEN_NEURONS; i++){
        cout<<"[ ";
        for (int j=0; j<INPUT_SIZE; j++){
            cout<<l1_weights[i][j]<<" ";
        }
        cout<<"]'"<<endl;
    }

    cout<<endl<<"L2 neuron weights and biases:"<<endl;
    for (int i=0; i<OUTPUT_NEURONS; i++){
        cout<<"[ ";
        for (int j=0; j<=HIDDEN_NEURONS; j++){
            cout<<l2_weights[i][j]<<" ";
        }
        cout<<"]'"<<endl;
    }

    // forward pass for all samples
    for (int sample=0; sample<TRAIN_SAMPLES; sample++){
        forward_pass(sample);
        // check results
        cout<<endl;
        cout<<"L1 weighted sums: [ ";
        cout<<l1_weighted_sums[0]<<" "<<l1_weighted_sums[1]<<" ]'"<<endl;
        cout<<"L1 outputs: [ ";
        cout<<l1_outputs[0]<<" "<<l1_outputs[1]<<" ]'"<<endl;
        cout<<"L2 weighted sums: [ ";
        cout<<l2_weighted_sums[0]<<" ]"<<endl;
        cout<<"L2 outputs: [ ";
        cout<<l2_outputs[0]<<" ]"<<endl;
    }

    return 0;

}
