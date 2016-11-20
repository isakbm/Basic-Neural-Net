#define NNET_NODE 0
#define NNET_WEIGHT 1

struct NNode
{
    int numWeights;
    std::vector<float> weights;
    std::vector<float> deltaWeights;
    
    float delta;  // for error propagation
    float z;      // for storing the argument of the activation function
    float out;    // the output of the node

    vec2 pos;
    bool isSelected; 

    NNode() : numWeights(0) , isSelected(false) {};
    NNode(unsigned int nW);
};

struct NLayer
{
    int numNodes;
    std::vector<NNode> nodes;

    NLayer() : numNodes(0) {};
    NLayer(unsigned int nN, unsigned int wPN);
};

class NNet 
{
    private:

        unsigned int numLayers;

        std::vector<NLayer> layers;

        std::vector<NLayer>::iterator inputLayer;               // these iterators will be initialized to "point" to their descriptive named locations
        std::vector<NLayer>::iterator firstHiddenLayer;
        std::vector<NLayer>::iterator lastHiddenLayer;
        std::vector<NLayer>::iterator outputLayer;

        float rho;                  // learning rate;
        unsigned int iterations;    // learning steps
        float avgError; 
        float sumError;
        std::queue<float> errorWindow;
        bool silent = true;        // "should we shut up (aka not print state of network) ? "
        
    public:
        
        NNet() : numLayers(0), iterations(0), sumError(0) {};
        NNet(std::vector<unsigned int> & data);

        void randOuts();
        void forwardPropagate();
        void setInputs(std::vector<float> & in);
        void randWeights(float (*rng)(void) );
        void backProp(std::vector<float> target);
        void updateWeights();

        void print();
        void test();
        
        void deleteSelectedNodes();

        void setRho(float rate);
        void setSilent(bool b);
        void setSelectedNode(int L, int N);


        void resetIterations();


        float getAvgError();

        unsigned int getNumLayers() const ;

        std::vector<NLayer>::const_iterator getInputLayerIt() const;
        std::vector<NLayer>::const_iterator getLayerEndIt() const;
};

