#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <queue>

// Custom vector and matrix classes and operator overloading.
// Contains only what I deemed necessary at the time
// Could probably use GLM instead

#include <GL/glew.h>    // extension loading
#include <GLFW/glfw3.h> // window and input

#include "mathGL.h"
#include "NNet.h"


#define MAX_INT 2147483647.0 


GLFWwindow* window;
double resx = 1600,resy = 900;

int clickedButtons = 0;
int prevClickedButtons = 0;
enum buttonMaps   { FIRST_BUTTON=1, SECOND_BUTTON=2, THIRD_BUTTON=4, FOURTH_BUTTON=8, FIFTH_BUTTON=16, NO_BUTTON=0 };
enum modifierMaps { CTRL=2, SHIFT=1, ALT=4, META=8, NO_MODIFIER=0 };


// Camera stuff
vec3 pos = vec3(0, 0, 50.0);

vec3 f = vec3(0.0, 0.0, -1.0);         
vec3 u = vec3(0.0, 1.0, 0.0);         
vec3 r = vec3(1.0, 0.0, 0.0);         

// forward, up and right vector relative to camera 

float phi = 0*PI/180.0;     // azimuthal angle (latitude), from 0 to 2*PI
float theta = 0*PI/180.0;  // zenith/polar angle (longitude) from 0 to PI
float fov = 45.0;   


float time;

// Buffer object stuffs
GLuint programID;
GLuint programID2;
GLuint VertexArrayID;

GLuint quadVertexbuffer;
GLuint lineGraphVertexBuffer;
GLuint smallGraphVertexBuffer;

bool toggleNetUpdate = 1;

// Function forward declarations

char *readFile(const char*);
GLuint LoadShaders(const char*, const char*);
void initGLFWandGLEW();
void initGL();
void cleanGL();
void Draw();

void key_callback(GLFWwindow*, int, int, int, int);
void mousebutton_callback(GLFWwindow*, int, int, int);
void mousepos_callback(GLFWwindow*, double, double);
void mousewheel_callback(GLFWwindow*, double, double);
void windowsize_callback(GLFWwindow*, int, int);

static const GLfloat quadVertices[] =
{
    -1.0f, -1.0f, 0.0f,
     1.0f, -1.0f, 0.0f, 
    -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f, 0.0f,
};



#define LIN_GRAPH_SIZE 1000
#define SMALL_GRAPH_SIZE 150

GLfloat lineGraphVertices[3*LIN_GRAPH_SIZE];
GLfloat smallGraphVertices[3*SMALL_GRAPH_SIZE];

float mRand();
void NNetDraw(const NNet* net, mat4 proj, int mvp_loc, int GLSL_program);
void initLineGraphData(int size);

void NNetInteractionUpdate(NNet* net);
void drawLineGraph(mat4 proj, int GLSL_program);
void drawNetFeedbackPrediction(mat4 Proj, int GLSL_program);

std::vector<unsigned int> data = {3, 3, 5, 5, 1};
NNet testNet = NNet(data);

int main() {

    for (int i = 0; i < 1500; i++)
    {
        mRand();
    }

    initLineGraphData(LIN_GRAPH_SIZE);
    initGLFWandGLEW();
    initGL();

    testNet.setSilent(false);  // makes NNet::print() silent 
    testNet.randWeights(mRand);
    testNet.setRho(0.5);
    // testNet.print();



    while ( !glfwWindowShouldClose(window) )
    {   
        NNetInteractionUpdate(&testNet);
        Draw();

        prevClickedButtons = clickedButtons;
        glfwPollEvents();
    }

    cleanGL();

    glfwTerminate();

    return 0;
}
 
void Draw() {


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(VertexArrayID);

    // get location of the modelview matrix in the shaders
    int mvp_loc_1 = glGetUniformLocation(programID, "MVP");      
    int mvp_loc_2 = glGetUniformLocation(programID2, "MVP");        

    mat4 Projection(1.0);
    Projection.m22 = float(resx)/float(resy);
    Projection.m44 = 0.5*double(resx);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // printf("%f ", (Projection*View).M[i][j]);
        }
        // printf("\n");
    }

    // Update network
    if (toggleNetUpdate)
    {  
        // toggleNetUpdate = 0;

        // float x = int(1.0 + mRand());
        // float y = int(1.0 + mRand());
        // float out = int(x) && int(y);
        
        // printf("%d\n", int(5.0*(1.0 + mRand())));
       


        // Labeled data point
        // --------------------------------------------------------
        int sampleStart = int(5.0*(1.0 + mRand()));
        float x1 = sin(2.0*PI*(float(sampleStart) + 0)/10.0);
        float x2 = sin(2.0*PI*(float(sampleStart) + 1)/10.0);
        float x3 = sin(2.0*PI*(float(sampleStart) + 2)/10.0);
        float t  = sin(2.0*PI*(float(sampleStart) + 3)/10.0);
        // --------------------------------------------------------


        // Updating weights via back propagation
        // --------------------------------------------------------
        std::vector<float> vecIn  = {x1, x2, x3};
        std::vector<float> target = {t};
        testNet.setInputs(vecIn);
        testNet.forwardPropagate();
        testNet.backProp(target);
     	testNet.updateWeights();
        // --------------------------------------------------------

        // testNet.print();
       

        // Displays the error
        drawLineGraph(Projection, programID2);

        // Displays prediction for RNN
        drawNetFeedbackPrediction(Projection, programID2);


    }
    else
    {
        drawNetFeedbackPrediction(Projection, programID2);

    }

    NNetDraw(&testNet, Projection, mvp_loc_1, programID);

    // drawLineGraph(Projection, View, programID2);

    // Swap buffers
    glfwSwapBuffers(window);

}


void cleanGL() {
    glDeleteBuffers(1, &quadVertexbuffer);
    glDeleteBuffers(1, &lineGraphVertexBuffer);
    glDeleteBuffers(1, &smallGraphVertexBuffer);

    glDeleteVertexArrays(1, &VertexArrayID);
    glDeleteProgram(programID);
    glDeleteProgram(programID2);
}

void initGLFWandGLEW() {
    printf("Initializing OpenGL/GLFW\n"); 
    if (!glfwInit()) {
        printf("Could not initialize\n");
        exit(-1);
    }

    // Create window
    glfwWindowHint(GLFW_SAMPLES, 4);    // samples, for antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // shader version should match these
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // do not use deprecated functionality

    window = glfwCreateWindow(resx, resy, "GLSL template", 0, 0);
    if (!window) {
        printf("Could not open glfw window\n");
        glfwTerminate();
        exit(-2);
    }
    glfwMakeContextCurrent(window); 

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(-3);
    }

    // set callback functions
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mousebutton_callback);
    glfwSetScrollCallback(window, mousewheel_callback);
    glfwSetCursorPosCallback(window, mousepos_callback);
    glfwSetWindowSizeCallback(window, windowsize_callback);

    // VSync on
    // set background color and enable depth testing
    glfwSwapInterval(1);
    glClearColor(0.1f, 0.1f, 0.25f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
}

void initGL() {
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray( VertexArrayID);

    // Create and compile GLSL program from the shaders
    programID  = LoadShaders( "vertex_shader.vs", "fragment_shader.fs" );
    programID2 = LoadShaders( "vertex_shader.vs", "colorRed_frag_shader.fs" );

    // Quad buffer data
    glGenBuffers(1, &quadVertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, quadVertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // Line graph buffer data
    glGenBuffers(1, &lineGraphVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, lineGraphVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineGraphVertices), lineGraphVertices, GL_STATIC_DRAW);  // Might want to change to GL_DYNAMIC_DRAW?

    // small graph buffer data
    glGenBuffers(1, &smallGraphVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, smallGraphVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(smallGraphVertices), smallGraphVertices, GL_STATIC_DRAW);
}

void windowsize_callback(GLFWwindow * /*win*/, int width, int height) { 
    // called if the window size is changed
    resx = width;
    resy = height;

    glViewport(0, 0, resx, resy);
    
}

void key_callback(GLFWwindow* win, int key, int /*scancode*/, int action, int /*mods*/) {
    // called if a keyboard key is pressed or released
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    { 
        toggleNetUpdate = !toggleNetUpdate;
        printf("pressed space\n");
    }
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(win, GL_TRUE);
    }
    if (key == GLFW_KEY_DELETE  && action == GLFW_PRESS)
    { 
        printf("pressed DEL\n");
        testNet.deleteSelectedNodes();
        testNet.print();

        
    }

}

void mousebutton_callback(GLFWwindow* win, int button, int action, int /*mods*/) {
    // called if a mouse button is pressed or released
    // glfwGetCursorPos(win,&prevx,&prevy);

    if (action == GLFW_PRESS)
    {
        // printf("setting\n");
        // prevClickedButtons = clickedButtons;
        clickedButtons |= (1 << button);
    }
    else
    {
                // printf("setting\n");

        // prevClickedButtons = clickedButtons;
        clickedButtons &= ~(1 << button);
    }

    if (clickedButtons & FIRST_BUTTON) {

    } else if (clickedButtons & SECOND_BUTTON) {
        // printf("propagating forward \n");
        // testNet.forwardPropagate();

    } else if (clickedButtons & THIRD_BUTTON) {

    } else if (clickedButtons & FOURTH_BUTTON) {

    } else if (clickedButtons & FIFTH_BUTTON) {

    }
}

void mousepos_callback(GLFWwindow* win, double xpos, double ypos) {
    // called if the mouse is moved
    if (clickedButtons&FIRST_BUTTON) {
        // printf("mouse pos : (%f, %f) \n", xpos, ypos);

    } else if (clickedButtons & SECOND_BUTTON) {

    } else if (clickedButtons & THIRD_BUTTON) {

    } else if (clickedButtons & FOURTH_BUTTON) {

    } else if (clickedButtons & FIFTH_BUTTON) {

    }
}

void mousewheel_callback(GLFWwindow* win, double xoffset, double yoffset) {
    // called if the scroll wheel is moved
    // changing the "field of view", the 3D "equivalent" of zooming
    double zoomFactor = pow(0.95, yoffset);
    fov = MAX(10, MIN(200.0, fov*zoomFactor));
    printf("fov = %f\n", fov);

    int zoom_loc = glGetUniformLocation(programID, "zoom");

    glUniform1f(zoom_loc, fov);

}



// Shader utils stuff
char *readFile(const char *filename) {
    // Read content of "filename" and return it as a c-string.
    printf("Reading %s\n", filename);
    FILE *f = fopen(filename, "rb");

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    printf("Filesize = %d\n", int(fsize));

    char *string = (char*)malloc(fsize + 1);
    fread(string, fsize, 1, f);
    string[fsize] = '\0';
    fclose(f);

    return string;
}


void CompileShader(const char * file_path, GLuint ShaderID) {
    GLint Result = GL_FALSE;
    int InfoLogLength;

    char *ShaderCode = readFile(file_path);

    // Compile Shader
    printf("Compiling shader : %s\n", file_path);
    glShaderSource(ShaderID, 1, (const char**)&ShaderCode , NULL);
    glCompileShader(ShaderID);

    // Check Shader
    glGetShaderiv(ShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(ShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);

    if ( Result == GL_FALSE ){
        char ShaderErrorMessage[9999];
        glGetShaderInfoLog(ShaderID, InfoLogLength, NULL, ShaderErrorMessage);
        printf("%s", ShaderErrorMessage);
    }
}

GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){
    printf("Creating shaders\n");
    GLuint VertexShaderID   = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    CompileShader(vertex_file_path, VertexShaderID);
    CompileShader(fragment_file_path, FragmentShaderID);


    printf("Create and linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    GLint Result = GL_FALSE;
    int InfoLogLength;

    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);

    if ( InfoLogLength > 0 ){
        GLchar ProgramErrorMessage[9999];
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]); fflush(stdout);

    }

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}


float mRand()
{
    static unsigned int IBM = 123;
    IBM *= 16807;
    return float(IBM)/MAX_INT - 1.0;
}

void NNetDraw(const NNet* net, mat4 Proj, int mvp_loc, int GLSL_program)
{
    glEnableVertexAttribArray(0);

    glUseProgram(GLSL_program);

    glBindBuffer(GL_ARRAY_BUFFER, quadVertexbuffer);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);

    int nodeFill_loc   = glGetUniformLocation(GLSL_program, "nodeFill");
    int frag_scale_loc = glGetUniformLocation(GLSL_program, "scale");
    int isSelected_loc = glGetUniformLocation(GLSL_program, "isSelected");
    int object_loc     = glGetUniformLocation(GLSL_program, "object");

    int layerIndex = 1;

    mat4 Scale = scale((45.0/fov)*vec3(1.0,1.0,0.0));
    mat4 SP = Scale*Proj;

    for (auto layer = net->getInputLayerIt(); layer != net->getLayerEndIt(); layer++)
    {
        int nodeIndex = 0;

        glDisable(GL_DEPTH_TEST);
       
        unsigned int numLayers = net->getNumLayers();
        unsigned int numNodes  = layer->numNodes;

        for (NNode node : layer->nodes)
        {
            vec3 nodePos = vec3(node.pos.x, node.pos.y, 0.0);
            glUniform1i(object_loc, NNET_NODE);
            glUniform1f(frag_scale_loc, 20.0);
            glUniform1f(nodeFill_loc, node.out);
            glUniform1i(isSelected_loc, node.isSelected);
            mat4 SPT = SP*translate( nodePos );
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &SPT.M[0][0]); 
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            for (int i = 0; i < node.numWeights; i++)
            {
                glUniform1i(object_loc, NNET_WEIGHT);
                float y = 12.0*(0.5 + (i - 0.5*node.numWeights));
                float x = -30.0 + 0.016*y*y;
                vec3 weightPos = nodePos + vec3(x, y, 0.0);
                glUniform1f(frag_scale_loc, 5.0);
                glUniform1f(nodeFill_loc, node.weights[i]);
                SPT = SP*translate( weightPos );

                glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &SPT.M[0][0]); 

                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }
            nodeIndex++;
        }
        layerIndex++;
    }

    glDisableVertexAttribArray(0);
}

void NNetInteractionUpdate(NNet* net)
{
    double cx, cy;
    glfwGetCursorPos(window, &cx, &cy);
    
    float scale = (fov/45.0);
    vec2 cursorPos ( + scale*(cx - 0.5*resx) , - scale*(cy - 0.5*resy) );

    float dist = 1E20;

    // printf("cursor pos = (%f, %f) \n", cursorPos.x, cursorPos.y);

    int nearestLayerID;
    int nearestNodeID;
    int layerID = 0;
    for (auto layer = net->getInputLayerIt(); layer != net->getLayerEndIt(); layer++)
    {
        int nodeID = 0;
        for (NNode node : layer->nodes)
        {
            float tempDist = (cursorPos - node.pos).length();
            if (tempDist < dist)
            {
                nearestLayerID = layerID;   
                nearestNodeID  = nodeID;
                dist = tempDist;
            } 
            nodeID++;
        }
        layerID++;
    }
    // printf("dist = %f\n", dist);

    if ( dist < 19.0 && (clickedButtons & FIRST_BUTTON) && !(prevClickedButtons & FIRST_BUTTON) )
    {
        printf("Setting node (%d, %d)\n", nearestLayerID, nearestNodeID);
        net->setSelectedNode(nearestLayerID, nearestNodeID);
    }

}


void drawLineGraph(mat4 Proj, int GLSL_program)
{ 
    static int start_loc = 0;
    glUseProgram(GLSL_program);

    glDisable(GL_DEPTH_TEST);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, lineGraphVertexBuffer);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);

    // int frag_scale_loc = glGetUniformLocation(GLSL_program, "scale");
    // glUniform1f(frag_scale_loc, 40.0);

    mat4 Model;
    mat4 MVP;
    mat4 Scale = scale((45.0/fov)*20.0*vec3(1.0, 1.0, 0.0));

    int mvp_loc = glGetUniformLocation(GLSL_program, "MVP");
    int scale_loc = glGetUniformLocation(GLSL_program, "scale");

    vec3 posAdjustment = vec3(-10.0, 0.0, 0.0);

    // draw first segment
    vec3 posSignalSource = -(1.0/30.0)*start_loc*vec3(1.0, 0.0, 0.0) + (1.0/30.0)*LIN_GRAPH_SIZE*vec3(1.0, 0.0, 0.0);
    Model = translate(posSignalSource + posAdjustment);
    MVP = Scale*Proj*Model;
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &MVP.M[0][0]); 
    glUniform1f(scale_loc, 1.0);
    glDrawArrays(GL_LINE_STRIP, 0, start_loc);


    // draw second segment
    vec3 posTail = -(1.0/30.0)*start_loc*vec3(1.0, 0.0, 0.0);  // the factor of 1/30 stems from the initial size of the data window, look at the initLineGraph()
    Model = translate(posTail + posAdjustment); 
    MVP =  Scale*Proj*Model;
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &MVP.M[0][0]); 
    glUniform1f(scale_loc, 1.0);
    glDrawArrays(GL_LINE_STRIP, start_loc, LIN_GRAPH_SIZE - start_loc);


    // overwriting old data with new data point
    GLfloat newPointData[3];
    newPointData[0] = (start_loc - 500.0)/30.0;
    newPointData[1] = 100.0*testNet.getAvgError(); //mRand();
    newPointData[2] = 0.0;
    glBufferSubData(GL_ARRAY_BUFFER, start_loc*sizeof(newPointData), sizeof(newPointData), newPointData);


    glDisableVertexAttribArray(0);

    start_loc ++;
    start_loc = (start_loc < LIN_GRAPH_SIZE) ? start_loc : 0;
}

void drawNetFeedbackPrediction(mat4 Proj, int GLSL_program)
{
    int size = 20;
    glUseProgram(GLSL_program);

    glDisable(GL_DEPTH_TEST);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, smallGraphVertexBuffer);

    float x1 = sin(2.0*PI*(0.0)/10.0);
    float x2 = sin(2.0*PI*(1.0)/10.0);
    float x3 = sin(2.0*PI*(2.0)/10.0);

    std::vector<float> input;
    input.push_back(x1);
    input.push_back(x2);
    input.push_back(x3);

    for (int i = 0; i < SMALL_GRAPH_SIZE; i++)
    {
        auto output = testNet.inputOutput(input);
        input[0] = input[1];
        input[1] = input[2];
        input[2] = output[0];
        
        smallGraphVertices[3*i + 0] = (i - 0.5*SMALL_GRAPH_SIZE)/3.0;
        smallGraphVertices[3*i + 1] = output[0];
        smallGraphVertices[3*i + 2] = 0.0;
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(smallGraphVertices), smallGraphVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);

    // int frag_scale_loc = glGetUniformLocation(GLSL_program, "scale");
    // glUniform1f(frag_scale_loc, 40.0);

    mat4 Model;
    mat4 MVP;
    mat4 Scale = scale((45.0/fov)*20.0*vec3(1.0, 1.0, 0.0));

    int mvp_loc = glGetUniformLocation(GLSL_program, "MVP");
    int scale_loc = glGetUniformLocation(GLSL_program, "scale");

    // draw first segment
    vec3 posSignalSource = 0.25*SMALL_GRAPH_SIZE*vec3(1.0, 0.0, 0.0);
    Model = translate(posSignalSource);
    MVP = Scale*Proj*Model;
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &MVP.M[0][0]); 
    glUniform1f(scale_loc, 1.0);
    glDrawArrays(GL_LINE_STRIP, 0, SMALL_GRAPH_SIZE);


    glDisableVertexAttribArray(0);

   
}

void initLineGraphData(int size) 
{
    for (int i = 0; i < size; i++)
    {
        lineGraphVertices[3*i + 0] = (i - 500.0)/30.0;
        lineGraphVertices[3*i + 1] = 0.0;
        lineGraphVertices[3*i + 2] = 0.0;
    }
} 