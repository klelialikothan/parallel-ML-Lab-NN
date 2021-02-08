#include <array>
#include <ctime>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

#define TRAIN_SAMPLES 1000
#define INPUT_SIZE 12+1
#define HIDDEN_NEURONS 100
#define OUTPUT_NEURONS 2

using std::array;
using std::cout;
using std::endl;

// init random engine for uniformly distributed doubles in [-1, 1] -> weights
std::default_random_engine w_gen(static_cast<unsigned>(time(nullptr)));
static std::uniform_real_distribution<double> w_distr(-1.0, 1.0);
// init random engine for uniformly distributed doubles in [-0.5, 0.5] -> biases
std::default_random_engine b_gen(static_cast<unsigned>(time(nullptr)));
static std::uniform_real_distribution<double> b_distr(-0.5, 0.5);
// init random engine for uniformly distributed doubles in [0.0, 100.0]
std::default_random_engine s_gen(static_cast<unsigned>(time(nullptr)));
static std::uniform_real_distribution<double> s_distr(0.0, 100.0);
// init random engine for uniformly distributed doubles in [400.0, 500.0]
std::default_random_engine l_gen(static_cast<unsigned>(time(nullptr)));
static std::uniform_real_distribution<double> l_distr(400.0, 500.0);

// arrays that store intermediate and final results
array<array<double, INPUT_SIZE>, TRAIN_SAMPLES> train_samples;
array<array<double, OUTPUT_NEURONS>, TRAIN_SAMPLES> train_labels;
array<array<double, INPUT_SIZE>, HIDDEN_NEURONS> l1_weights;
array<double, HIDDEN_NEURONS> l1_weighted_sums;
array<double, HIDDEN_NEURONS+1> l1_outputs;
array<double, HIDDEN_NEURONS> l1_deltas;
array<array<double, HIDDEN_NEURONS+1>, OUTPUT_NEURONS> l2_weights;
array<double, OUTPUT_NEURONS> l2_weighted_sums;
array<double, OUTPUT_NEURONS> l2_outputs;
array<double, OUTPUT_NEURONS> l2_errors;
array<double, OUTPUT_NEURONS> l2_deltas;

// misc variables
double alpha = 0.01;
double total_error;

// generate random training set
void init_train_set(){

    // class 0 samples
    for (int i=0; i<TRAIN_SAMPLES/2; i++){
        for (int j=0; j<INPUT_SIZE/2; j++){
            train_samples[i][j] = l_distr(l_gen);
        }
        for (int j=INPUT_SIZE/2; j<INPUT_SIZE-1; j++){
            train_samples[i][j] = s_distr(s_gen);
        }
        train_samples[i][INPUT_SIZE-1] = 1;  // bias coefficient
        train_labels[i][0] = 1.0;
        train_labels[i][1] = 0.0;
    }

    // class 1 samples
    for (int i=TRAIN_SAMPLES/2; i<TRAIN_SAMPLES; i++){
        for (int j=0; j<INPUT_SIZE/2; j++){
            train_samples[i][j] = s_distr(s_gen);
        }
        for (int j=INPUT_SIZE/2; j<INPUT_SIZE-1; j++){
            train_samples[i][j] = l_distr(l_gen);
        }
        train_samples[i][INPUT_SIZE-1] = 1;  // bias coefficient
        train_labels[i][0] = 0.0;
        train_labels[i][1] = 1.0;
    }

}


// init neuron weights and biases
void init_network(){

    // layer 1 (hidden layer)
    for (int i=0; i<HIDDEN_NEURONS; i++){
        for (int j=0; j<INPUT_SIZE-1; j++){
            l1_weights[i][j] = w_distr(w_gen);
        }
        l1_weights[i][INPUT_SIZE-1] = b_distr(b_gen);
    }

    // layer 2 (output layer)
    for (int i=0; i<OUTPUT_NEURONS; i++){
        for (int j=0; j<HIDDEN_NEURONS; j++){
            l2_weights[i][j] = w_distr(w_gen);
        }
        l2_weights[i][HIDDEN_NEURONS] = b_distr(b_gen);
    }

    // level 2 bias coefficient (always equal to 1)
    l1_outputs[HIDDEN_NEURONS] = 1.0;

    l1_deltas = {{0}};

}

// activation function
double sigmoid(double x){

    return 1 / (1 + std::exp(-x));

}

// error function
void calc_error(int sample){

    // calculate error for every level 2 neuron
    total_error = 0.0;
    #pragma omp for simd
    for (int i=0; i<OUTPUT_NEURONS; i++){
        l2_errors[i] = (l2_outputs[i] - train_labels[sample][i]);
        total_error += l2_errors[i] * l2_errors[i];
    }

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

    #pragma omp parallel for
    for (int neuron=0; neuron<HIDDEN_NEURONS; neuron++){  // for each l1 neuron
        // calculate weighted sum
        double w_sum = 0.0;
        #pragma omp simd reduction(+:w_sum)
        for (int synapse=0; synapse<INPUT_SIZE; synapse++){
            w_sum += l1_weights[neuron][synapse] * train_samples[sample][synapse];
        }
        l1_weighted_sums[neuron] = w_sum;
        l1_outputs[neuron] = sigmoid(l1_weighted_sums[neuron]);  // neuron activation
    }

    // layer 2 (output layer)
    for (int neuron=0; neuron<OUTPUT_NEURONS; neuron++){  // for each l2 neuron
        // calculate weighted sum
        #pragma omp for simd
        for (int synapse=0; synapse<=HIDDEN_NEURONS; synapse++){
            l2_weighted_sums[neuron] += l2_weights[neuron][synapse] * l1_outputs[synapse];
        }
        l2_outputs[neuron] = sigmoid(l2_weighted_sums[neuron]);  // neuron activation
    }

}

// error backpropagation
void backpropagation(int sample){

    // calculate errors for layer 2 neurons
    calc_error(sample);

    // layer 2 computations
    #pragma omp for simd
    for (int neuron=0; neuron<OUTPUT_NEURONS; neuron++){
        // calculate deltas for layer 2 neurons
        l2_deltas[neuron] = l2_outputs[neuron] * (1.0 - l2_outputs[neuron]);
        for (int synapse=0; synapse<HIDDEN_NEURONS; synapse++){
            // calculate delta sums for layer 1 adjustment
            l1_deltas[synapse] += l2_deltas[neuron] * l2_weights[neuron][synapse];
            // adjust layer 2 weights
            l2_weights[neuron][synapse] -= alpha * l2_deltas[neuron] * l2_errors[neuron] * l1_outputs[neuron];
        }
    }

    // complete calculation of layer 1 deltas
    #pragma omp parallel for simd
    for (int neuron=0; neuron<HIDDEN_NEURONS; neuron++){
        l1_deltas[neuron] *= l1_outputs[neuron] * (1 - l1_outputs[neuron]);
    }

    // adjust layer 1 weights
    #pragma omp parallel for simd collapse(2)
    for (int neuron=0; neuron<HIDDEN_NEURONS; neuron++){
        for (int synapse=0; synapse<INPUT_SIZE; synapse++){
            l1_weights[neuron][synapse] -= alpha * l1_deltas[neuron] * train_samples[sample][synapse];
        }
    }

}

double calc_accuracy(int n_samples){

    double accuracy = 0.0;
    int assigned_class = -1;
    int true_class = -1;
    double max_out = 0.0;
    for (int i=0; i<n_samples; i++){
        forward_pass(i);
        for (int j=0; j<INPUT_SIZE; j++){
            if (l2_outputs[j] > max_out){
                assigned_class = j;
            }
            if (train_labels[i][j] > true_class){
                true_class = j;
            }
        }
        if (assigned_class == true_class){
            accuracy += 1.0;
        }
        assigned_class = -1;
        true_class = -1;
        max_out = 0.0;
    }

    return 100 * accuracy / n_samples;

}

int main(){

    init_train_set();
    init_network();

    // baseline
    // cout<<"Initial accuracy: "<<calc_accuracy(TRAIN_SAMPLES/2)<<"%"<<endl;

    // train network
    // int epochs = 1;
    std::chrono::duration<double> avg_time;
    for (int times=0; times<100; times++){
        for (int sample=0; sample<TRAIN_SAMPLES; sample++){
            auto start = std::chrono::high_resolution_clock::now();
            forward_pass(sample);
            backpropagation(sample);
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = finish - start;
            avg_time += diff;
            // cout<<"Epoch #"<<epochs<<" total error: "<<total_error<<endl;
            // epochs++;
        }
        if (!(times % 10)){
            cout<<"Epoch #"<<times * TRAIN_SAMPLES<<" total error: "<<total_error<<endl;
        }
    }
    avg_time /= (100 * TRAIN_SAMPLES);

    // final accuracy
    // cout<<"Accuracy after 40000 epochs: "<<calc_accuracy(TRAIN_SAMPLES)<<"%"<<endl;
    cout<<"Average time for 1 sample: "<<std::chrono::duration<double,std::milli>(avg_time).count()<<" ms"<<endl;

    return 0;

}
