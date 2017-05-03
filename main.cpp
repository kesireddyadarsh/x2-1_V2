//
//  main.cpp
//  x2+1_v3
//
//  Created by adarsh kesireddy on 4/23/16.
//  Copyright © 2016 adarsh kesireddy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>

using namespace std;

struct connect{
    double weight;
};

static double random_global(double a) { return a* (rand() / double(RAND_MAX)); }

// This is for each Neuron

class Neuron;
typedef vector<Neuron> Layer;

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    vector<connect> z_outputWeights;
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    unsigned z_myIndex;
    double z_outputVal;
    void setOutputVal(double val) { z_outputVal = val; }
    double getOutputVal(void) const { return z_outputVal; }
    void feedForward(const Layer prevLayer);
    double transferFunction(double x);
};

//This creates connection with neurons.

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        z_outputWeights.back().weight = randomWeight();
    }
    z_myIndex = myIndex;
}


double Neuron::transferFunction(double x){
    return tanh(x);
}


void Neuron::feedForward(const Layer prevLayer){
    double sum = 0.0;
    bool debug_sum_flag = false;
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        if(debug_sum_flag == true){
            cout<<prevLayer[n].getOutputVal()<<endl;
            cout<<&prevLayer[n].z_outputWeights[z_myIndex];
            cout<<prevLayer[n].z_outputWeights[z_myIndex].weight;
        }
        sum += prevLayer[n].getOutputVal() * prevLayer[n].z_outputWeights[z_myIndex].weight;
        //cout<<"This is sum value"<<sum<<endl;
    }
    //cout<<"This is output passing to node::"<<sum<<endl;
    z_outputVal = Neuron::transferFunction(sum);
}

//This is single neural network

class Net{
public:
    Net(vector<unsigned> topology);
    void feedForward(vector<double> inputVal, vector<double> inputVal_scaled,  int numCases, int max_range, int min_range, double interval);
    vector<Layer> z_layer;
    double backProp();
    double z_error;
    double z_error_temp;
    vector<double> z_error_vector;
    void mutate();
    vector<double> temp_inputs;
    vector<double> temp_targets;
    vector<double> temp_ref;
    double scale(double val, int max_range, int min_range);
    vector<double> outputvalues;
};

Net::Net(vector<unsigned> topology){
    
    for(int  numLayers = 0; numLayers<topology.size(); numLayers++){
        //unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        
        unsigned numOutputs;
        if (numLayers == topology.size()-1) {
            numOutputs=0;
        }else{
            numOutputs= topology[numLayers+1];
        }
        
        if(numOutputs>11){
            cout<<"Stop it number outputs coming out"<<numOutputs<<endl;
            exit(10);
        }
        
        z_layer.push_back(Layer());
        
        for(int numNeurons = 0; numNeurons <= topology[numLayers]; numNeurons++){
            //cout<<"This is neuron number:"<<numNeurons<<endl;
            z_layer.back().push_back(Neuron(numOutputs, numNeurons));
        }
        
        z_layer.back().back().setOutputVal(1.0);
        
    }
}

void Net::mutate(){
    /*
     //popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight
     */
    for (int l =0 ; l < z_layer.size(); l++) {
        for (int n =0 ; n< z_layer[l].size(); n++) {
            for (int z=0 ; z< z_layer[l][n].z_outputWeights.size(); z++) {
                z_layer[l][n].z_outputWeights[z].weight += random_global(.5)-random_global(.5);
            }
        }
    }
}

double Net::scale(double val, int max_range, int min_range){
    val = val *(max_range-min_range); // val = val * (max-min);
    val = val + min_range; // val =val + (min)
    //cout<<"This is var in scale for output:: "<<val<<endl;
    return val;
}

void Net::feedForward(vector<double> inputVal, vector<double> inputVal_scaled,  int numCases, int max_range, int min_range, double interval){
    int cycle_inputs = 0 ;
    temp_inputs.clear();
    z_error_vector.clear();
    
    
    int cycle_target = 0 ;
    
    while (cycle_inputs<(inputVal.size()) ) {
        int push_vector_input = cycle_inputs;
        int push_vector_target = cycle_target;
        for ( int temp =0 ; temp<(inputVal_scaled.size()/numCases); temp++) {
            temp_inputs.push_back(inputVal_scaled.at(push_vector_input));
            temp_ref.push_back(inputVal.at(push_vector_input));
            push_vector_input++;
        }
        assert(temp_inputs.size() == z_layer[0].size()-1);
        for (unsigned i=0; i<temp_inputs.size(); ++i) {
            z_layer[0][i].setOutputVal(temp_inputs[i]);
        }
        for (unsigned layerNum = 1; layerNum < z_layer.size(); ++layerNum) {
            Layer &prevLayer = z_layer[layerNum - 1];
            for (unsigned n = 0; n < z_layer[layerNum].size() - 1; ++n) {
                z_layer[layerNum][n].feedForward(prevLayer);
            }
        }
        
        //Function for output
        double max_target = max_range*max_range+1;
        double min_target = min_range*min_range+1;
        
        double target = temp_ref[0]*temp_ref[0]+1;
        
        double fitness = 0.0;
        
        temp_inputs.clear();
        temp_ref.clear();
        Layer &outputLayer = z_layer.back();
        for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
            double temp = scale(outputLayer[n].getOutputVal(),max_target,min_target);
            outputvalues.push_back(temp);
            fitness = temp - target;
            fitness = (fitness > 0)?fitness:-fitness;
        }
        z_error_vector.push_back(fitness);
        cycle_inputs += (inputVal.size()/numCases);
    }
}


double Net::backProp(){
    z_error = 0.0;
    for (int temp = 0; temp< z_error_vector.size(); temp++) {
        //cout<<z_error_vector[temp]<<"\t";
        z_error += z_error_vector[temp];
    }
    return z_error;
}



//This is for population of neural network
class population{
public:
    population(int numNN,vector<unsigned> topology);
    vector<Net> popVector;
    void runNetwork(vector<double> inputVal, vector<double> inputVal_scaled, int numNN, int numCases, int max_range, int min_range, double interval);
    void sortError();
    void mutation(int numNN);
    void newerrorvector();
    void findindex();
    int returnIndex(int numNN);
    void repop(int numNN);
    vector<double> error_vector;
    
};

// variables used: indiNet -- object to Net
population::population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net indiNet(topology);
        popVector.push_back(indiNet);
    }
    
}

//Return index of higher

int population::returnIndex(int numNN){
    int temp = numNN;
    int number_1 = (rand() % temp);
    int number_2 = (rand() % temp);
    while (number_1 == number_2) {
        number_2 = (rand() % temp);
    }
    //double temp_1 =popVector[number_1].z_error;
    //double temp_2 =popVector[number_2].z_error;
    //cout<<"This is error in comparision::"<<popVector[number_1].z_error<<endl;
    //cout<<"This is error in comparision::"<<popVector[number_2].z_error<<endl;
    
    if (popVector[number_1].z_error<popVector[number_2].z_error) {
        return number_2;
    }else if (popVector[number_1].z_error>popVector[number_2].z_error){
        return number_1;
    }else{
        return NULL;
    }
}


void population::repop(int numNN){
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% popVector.size();
        popVector.push_back(popVector.at(R));
        popVector.back().mutate();
    }
}


void population::runNetwork(vector<double> inputVal, vector<double> inputVal_scaled, int numNN, int numCases, int max_range, int min_range, double interval){
    
    for (int temp=0 ; temp< numNN; temp++) {
        //Run neural network.
        popVector[temp].feedForward(inputVal, inputVal_scaled, numCases,  max_range,  min_range, interval);
        popVector[temp].backProp();
        error_vector.push_back(popVector[temp].z_error);
        cout<<popVector[temp].z_error<<"\t";
    }
    sort(error_vector.begin(), error_vector.end());
    for (int check_lowest_error = 0 ; check_lowest_error<error_vector.size(); check_lowest_error++) {
        if(error_vector.at(0)==popVector[check_lowest_error].z_error){
            cout<<" this"<<popVector[check_lowest_error].z_layer[2][0].z_outputVal<<endl;
            
            cout<<popVector[check_lowest_error].z_error<<endl;
            error_vector.clear();
            for (int temp_1 =0 ; temp_1<popVector[check_lowest_error].outputvalues.size(); temp_1++) {
                cout<<popVector.at(check_lowest_error).outputvalues.at(temp_1)<<"\t";
            }
            cout<<"completed"<<endl;
            for (int temp_2 =0; temp_2<popVector[temp_2].outputvalues.size(); temp_2++) {
                popVector.at(temp_2).outputvalues.clear();
            }
        }
        
    }
    for (int temp = 0 ; temp < numNN/2; temp++) {
        int temp_index = returnIndex(popVector.size());
        popVector.erase(popVector.begin()+temp_index);
    }
    //cout<<"This is size::"<<popVector.size()<<endl;
    repop(numNN);
}

double scale_input(double var,int max_range, int min_range){
    var = (var - min_range)/(max_range-min_range); //value - min/max-min
    return var;
}



//This is main function

int main(int argc, const char * argv[]) {
    // insert code here...
    //cout << "Hello, World!\n";
    srand(time(NULL));
    vector<double> inputVal_scaled;
    vector<double> inputVal;
    
    
    int numNN=100;
    int numCases = 0;
    int max_range = 5;
    int min_range = 0;
    double interval = 0.1;
    
    vector<unsigned> topology;
    topology.clear();
    inputVal.clear();
    topology.push_back(1);
    topology.push_back(4);
    topology.push_back(1);
    population mypop(numNN,topology);
    
    bool test_init = true;
    
    if (test_init==true) {
        for (int iterations = 0; iterations<10000; iterations++) {
            for (float number =0.0 ; number<=max_range; number=number+interval) {
                inputVal.push_back(number);
                inputVal_scaled.push_back(scale_input(number,max_range,min_range));
            }
            numCases = inputVal.size();
            mypop.runNetwork(inputVal, inputVal_scaled, numNN, numCases, max_range, min_range, interval);
            inputVal_scaled.clear();
            inputVal.clear();
            cout<<iterations<<endl;
        }
        //cout<<"This is completed errors"<<endl;
        
    }
    
    
    return 0;
}