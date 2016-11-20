#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <queue>

#include "mathGL.h"
#include "NNet.h"

NNode::NNode(unsigned int num)
{
    isSelected = false;
    numWeights = num;
    weights.assign(numWeights, 1.0); 
    deltaWeights.assign(numWeights, 0.0); 
    delta = 0.0;
}

NLayer::NLayer(unsigned int num, unsigned int weightsPerNode)
{
    numNodes = num;
    nodes.assign(numNodes, NNode(weightsPerNode));
}

NNet::NNet(std::vector<unsigned int> & numNPL)    // NPL = nodes per layer
{
    sumError = 0.0;
    iterations = 0;
    numLayers = numNPL.size(); // set total number of layers
    rho = 0.01;  // set the default learning rate

    // allocate and initialize layers 
    std::vector<unsigned int> numWPN;                           // num of weights per node
    numWPN.push_back(1);                                        // the input layer always has one weight per node
    numWPN.insert(end(numWPN), begin(numNPL), end(numNPL));     // the rest have num weights = num of nodes on previous layer
    for (int n = 0; n < numNPL.size(); n++)
    {
        layers.push_back(NLayer(numNPL[n], numWPN[n]));
    }

    // set iterators
    inputLayer       = layers.begin();
    firstHiddenLayer = layers.begin() + 1;
    lastHiddenLayer  = layers.end() - 2;
    outputLayer      = layers.end() - 1;

    // set positions
    int layerIndex = 0;
    for (auto &layer : layers)
    {
        int nodeIndex = 0;
        for (auto &node : layer.nodes)
        {
            node.pos = vec2(5.0*(layerIndex - 0.5*(numLayers-1)), 5.0*(nodeIndex - 0.5*(layer.numNodes-1)));
            nodeIndex++;        
        }
        layerIndex++;
    }

}

// 

unsigned int NNet::getNumLayers() const 
{
    return numLayers;
}

void NNet::resetIterations()
{
    iterations = 0.0;
}

std::vector<NLayer>::const_iterator NNet::getInputLayerIt() const
{
    return inputLayer;
}

std::vector<NLayer>::const_iterator NNet::getLayerEndIt() const
{
    return layers.end();
}

void NNet::setRho(float rate)
{
    rho = rate;
}

void NNet::updateWeights()
{
    for (auto layer = firstHiddenLayer; layer != layers.end(); layer++)
    {
        for (auto & node : layer->nodes)
        {
            for (int wIndex = 0; wIndex < node.numWeights; wIndex++)
            {
                node.weights[wIndex] += node.deltaWeights[wIndex];
            }
        }
    }
}

void NNet::setSelectedNode(int L, int N)
{
    if (L < 0 || L >= numLayers)
    {
        printf("The layer selection (L = %d) is out of range\n", L);
        return;
    }
    else if (N < 0 || N >= layers[L].numNodes)
    {
        printf("The node selection (L,N) = (%d, %d) is out of range\n", L, N);
        return;
    }
    
    bool & state = layers[L].nodes[N].isSelected;
    state = !state;
}


void NNet::backProp(std::vector<float> target)
{
    if (target.size() != outputLayer->numNodes)
    {
        printf("Warning! Failed to backprop: check that number of targets equals number of nodes in output layer \n");
    }
    else
    {
        if (iterations == 0)
        {
            sumError = 0.0; // reset sumError
        }
        iterations++;

        // computation for the output layer
        auto outNode = outputLayer->nodes.begin();
        for (float t : target)
        {   
            outNode->delta = (t - outNode->out)*exp(-0.7*(outNode->z)*(outNode->z)); // (o - t)*(1 - z^2)  this is using 1 - z^2 as the approx tanh'(z) 

            auto prevLayer = outputLayer -1;
            auto prevNode = prevLayer->nodes.begin();

            for (auto & deltaW : outNode->deltaWeights)
            {
                deltaW = rho*(outNode->delta)*(prevNode->out);
                prevNode++;
            }
            outNode++;
        }

        // computation for the hidden layers
        for (auto layer = lastHiddenLayer; layer != inputLayer; layer--)   // note that this loop goes "backwards" that is important
        {
            auto prevLayer = layer -1;
            auto nextLayer = layer +1;

            int wIndex = 0;
            for (auto & node : layer->nodes)
            {
                float sum = 0.0;
                for (auto & nextNode : nextLayer->nodes)
                {
                    sum += (nextNode.delta)*(nextNode.weights[wIndex]);
                }
                node.delta = exp(-0.7*(node.z)*(node.z))*sum;
                wIndex++;

                int dIndex = 0;
                for (auto & prevNode : prevLayer->nodes)
                {
                    node.deltaWeights[dIndex] = rho*(node.delta)*(prevNode.out);
                    dIndex++;
                }
            }
        }
    }

    // Average error in an error window of size 100 is tracked
    float error = 0.0;
    for (int t = 0; t < target.size(); t++)
    {
        error += (target[t] - outputLayer->nodes[t].out)*(target[t] - outputLayer->nodes[t].out)/target.size();
    }

    sumError += error;
    errorWindow.push(error); 
    if (iterations > 100)
    {
        sumError -= errorWindow.front();
        errorWindow.pop();
        avgError = sumError/float(errorWindow.size());
    }
}
void NNet::test()
{
    int layerID = 0, nodeID = 0;
    printf("testing new imp: \n");

    printf("size = %d\n", layers.size());
    for (auto layer = firstHiddenLayer; layer != layers.end(); ++layer)
    {
        auto prevLayer = std::prev(layer);
        for (NNode & node : layer->nodes)
        {
            printf("(L,N) = (%d,%d)\n" ,layerID, nodeID);
            float sum = 0.0;
            auto prevLayerNode = prevLayer->nodes.begin();
            for (float weight : node.weights)
            {
                sum += weight*(prevLayerNode->out);
                prevLayerNode++;
            }
            node.z = sum;
            node.out = tanh(sum);
            nodeID++;
        }
        layerID++;
    }

    printf("\n\n");
    layerID = 0;
    for (auto layer = lastHiddenLayer; layer != inputLayer; layer--)
    {
        printf("(L,N) = (%d,%d)\n" ,layerID, layer->nodes.size());
        layerID++;
    }
}

void NNet::forwardPropagate()
{
    for (auto layer = firstHiddenLayer; layer != layers.end(); ++layer)
    {
        auto prevLayer = std::prev(layer);
        for (NNode & node : layer->nodes)
        {
            float sum = 0.0;
            auto prevLayerNode = prevLayer->nodes.begin();
            for (float weight : node.weights)
            {
                sum += weight*(prevLayerNode->out);
                prevLayerNode++;
            }
            node.z = sum;
            node.out = tanh(sum);
        }
    }
}

void NNet::randWeights(float (*rng)(void) )
{
    for (auto layer = firstHiddenLayer; layer != layers.end(); layer++)
    {
        for (auto & node : layer->nodes)
        {
            for (auto & weight : node.weights)
            {
                weight = 0.5 + 0.5*rng();
            }
        }
    }
}

void NNet::setInputs(std::vector<float> & in)
{
    if ( inputLayer->numNodes == in.size() )
    {
        auto inNode = inputLayer->nodes.begin();
        for (auto input : in)
        {
            inNode->z = input;
            inNode->out = input; 
            inNode++;
        }
    }
    else
        printf("dude you messed up, have to have as many input vals as there are input nodes\n");
}

void NNet::print()
{
    if (silent)
        return;

    // Print Network status
    printf("-----------------------------------------------------------------------\n");
    printf("Iterations : %d \n", iterations);
    printf("Current avg error : %f \n", avgError);
    printf("Outputs = { ");
    for (int o = 0 ; o < outputLayer->numNodes-1; o++)
    {
        printf("%f, ", outputLayer->nodes[o].out);
    }
    printf("%f } \n\n\n", outputLayer->nodes[outputLayer->numNodes-1].out);

    int layerNum = 0;
    for (auto layer = inputLayer; layer != layers.end(); layer++)
    {
        printf("Layer %d : \t o  = ", layerNum);
        for (auto & node : layer->nodes)
        {
            printf("%f, ", node.out);
        }
        printf("\n");
        printf("\t\t z  = ");
        for (auto & node : layer->nodes)
        {
            printf("%f, ", node.z);
        }
        printf("\n\n");


        int nodeID = 1;      
        for (auto & node : layer->nodes)
        {

            printf("selected : %d \n", node.isSelected);
            printf("Node %d : " , nodeID);
            printf("\t w  = ");
            for (auto & weight : node.weights)
            {
                printf("%f, ", weight);
            }
            printf("\n");
            printf("d = %f \t Dw = ", node.delta);
            for (auto & deltaW : node.deltaWeights)
            {
                printf("%f, ", deltaW);
            }
            nodeID++;
            printf("\n");
            printf("\n");
        }

        layerNum++;
        printf("\n\n");
    }
}

void NNet::setSilent(bool b)
{
    silent = b;
}


float NNet::getAvgError()
{
    return avgError;
}