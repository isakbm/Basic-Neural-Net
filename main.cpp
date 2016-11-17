#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <queue>

// Custom vector and matrix classes and operator overloading.
// Contains only what I deemed necessary at the time
// Could probably use GLM instead
#include "mathGL.h"

#include <GL/glew.h>    // extension loading
#include <GLFW/glfw3.h> // window and input

#include "NNet.h"


#define MAX_INT 2147483647.0 


GLFWwindow* window;
double resx = 1600,resy = 900;

int clickedButtons = 0;
enum buttonMaps { FIRST_BUTTON=1, SECOND_BUTTON=2, THIRD_BUTTON=4, FOURTH_BUTTON=8, FIFTH_BUTTON=16, NO_BUTTON=0 };
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
GLuint quadUVbuffer;

GLuint lineGraphVertexBuffer;

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

static const GLfloat quadUV[] =
{
     0.0f,  0.0f,
     1.0f,  0.0f, 
     0.0f,  1.0f,
     1.0f,  1.0f, 
};

#define LIN_GRAPH_SIZE 1000

GLfloat lineGraphVertices[3*LIN_GRAPH_SIZE];

float mRand();
void drawNNet(const NNet* net, mat4 proj, mat4 view, int mvp_loc);
void initLineGraphData(int size);

std::vector<unsigned int> data = {2,3,4,4,1};
NNet testNet = NNet(data);

int main() {

    initLineGraphData(LIN_GRAPH_SIZE);
    initGLFWandGLEW();
    initGL();

    testNet.setSilent(false);  // makes NNet::print() silent 

    for (int i = 0; i < 1500; i++)
    {
        mRand();
    }

    testNet.randWeights(mRand);
    testNet.setRho(0.5);

    testNet.print();


    while ( !glfwWindowShouldClose(window)) {   
        Draw();
        glfwPollEvents();
    }

    cleanGL();

    glfwTerminate();

    return 0;
}

void Draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Tell OpenGL which program to use (can switch between multiple)
    glBindVertexArray(VertexArrayID);

    // get location of the modelview matrix in the shaders
    int mvp_loc_1 = glGetUniformLocation(programID, "MVP");      

    // Create view and projection matrix, same for every cube drawn
    mat4 View = view(r, u, f, pos);
    mat4 Projection = projection(fov, resx/float(resy), 0.1, 1000.0);

    mat4 Model = translate(vec3(0.0, 0.0, 0.0));
    mat4 MVP = Projection*View*Model;

    // Update network
    if (toggleNetUpdate)
    {  
        // toggleNetUpdate = 0;

        float x = int(1.0 + mRand());
        float y = int(1.0 + mRand());
        float out = int(x) && int(y);
        // printf("(%f, %f) --> %f\n", x,y,out);
        std::vector<float> vecIn = {x, y};
        std::vector<float> target = {out};

        testNet.setInputs(vecIn);
        testNet.forwardPropagate();
        testNet.backProp(target);

        testNet.print();
        testNet.updateWeights();

    }

    drawNNet(&testNet, Projection, View, mvp_loc_1);

    // Swap buffers
    glfwSwapBuffers(window);

}

void cleanGL() {
    glDeleteBuffers(1, &quadVertexbuffer);
    glDeleteBuffers(1, &quadUVbuffer);
    glDeleteBuffers(1, &lineGraphVertexBuffer);

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
    glGenBuffers(1, &quadUVbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, quadUVbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadUV), quadUV, GL_STATIC_DRAW);   

    // Line graph buffer data
    glGenBuffers(1, &lineGraphVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, lineGraphVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineGraphVertices), lineGraphVertices, GL_STATIC_DRAW);  // Might want to change to GL_DYNAMIC_DRAW?
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
        
    }
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(win, GL_TRUE);
    }
}

void mousebutton_callback(GLFWwindow* win, int button, int action, int /*mods*/) {
    // called if a mouse button is pressed or released
    // glfwGetCursorPos(win,&prevx,&prevy);

    if (action == 1)
        clickedButtons |= (1 << button);
    else
        clickedButtons &= ~(1 << button);

    if (clickedButtons&FIRST_BUTTON) {
        
    } else if (clickedButtons&SECOND_BUTTON) {
        // printf("propagating forward \n");
        // testNet.forwardPropagate();

    } else if (clickedButtons&THIRD_BUTTON) {

    } else if (clickedButtons&FOURTH_BUTTON) {

    } else if (clickedButtons&FIFTH_BUTTON) {

    }
}

void mousepos_callback(GLFWwindow* win, double xpos, double ypos) {
    // called if the mouse is moved
    if (clickedButtons&FIRST_BUTTON) {
        // printf("mouse pos : (%f, %f) \n", xpos, ypos);

    } else if (clickedButtons&SECOND_BUTTON) {

    } else if (clickedButtons&THIRD_BUTTON) {

    } else if (clickedButtons&FOURTH_BUTTON) {

    } else if (clickedButtons&FIFTH_BUTTON) {

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

void drawNNet(const NNet* net, mat4 proj, mat4 view, int mvp_loc)
{
    glEnableVertexAttribArray(0);

    glUseProgram(programID);

    glBindBuffer(GL_ARRAY_BUFFER, quadVertexbuffer);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);

    int loc       = glGetUniformLocation(programID, "nodeFill");
    int scale_loc = glGetUniformLocation(programID, "scale");

    int layerIndex = 1;

    for (auto layer = net->getInputLayerIt(); layer != net->getLayerEndIt(); layer++)
    {
        int nodeIndex = 0;

        glDisable(GL_DEPTH_TEST);
       
        unsigned int numLayers = net->getNumLayers();
        unsigned int numNodes = layer->numNodes;

        for (NNode node : layer->nodes)
        {
            vec3 nodePos = vec3(5.0*(layerIndex - 0.5*numLayers), 2.5 + 5.0*(nodeIndex - 0.5*numNodes), 0.0);
            glUniform1f(scale_loc, 1.0);
            glUniform1f(loc, node.out);
            mat4 model = translate( nodePos ); 
            mat4 MVP = proj*view*model;
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &MVP.M[0][0]); 
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            for (int i = 0; i < node.numWeights; i++)
            {
                vec3 weightPos = nodePos + vec3(-1.5, 0.5 + (i - 0.5*node.numWeights), 0.0);
                glUniform1f(scale_loc, 4.0);
                glUniform1f(loc, node.weights[i]);
                model = translate( weightPos ); 
                MVP = proj*view*model;
                glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &MVP.M[0][0]); 
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }
            nodeIndex++;
        }
        layerIndex++;
    }

    glDisableVertexAttribArray(0);
}

void initLineGraphData(int size)
{
    for (int i = 0; i < size; i++)
    {
        lineGraphVertices[2*i + 0] = i/500.0;
        lineGraphVertices[2*i + 1] = 0.0;
        lineGraphVertices[2*i + 2] = 0.0;
    }
}