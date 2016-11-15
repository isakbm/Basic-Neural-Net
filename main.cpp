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

bool toggleNetUpdate = 1;

// Texture parameters

#define NUM_TEXTURES 2
GLuint *textures = new GLuint[NUM_TEXTURES];

int textureX = 20, textureY = 20;
#define TEX_SIZE 1200
unsigned char data1[TEX_SIZE];
unsigned char data2[TEX_SIZE];

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

float mRand();
void drawNNet(const NNet* net, mat4 proj, mat4 view, int mvp_loc);

std::vector<unsigned int> data = {2,2,1};
NNet testNet = NNet(data);

int main() {

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
    int MVP_loc = glGetUniformLocation(programID, "MVP");      

    // Create view and projection matrix, same for every cube drawn
    mat4 View = view(r, u, f, pos);
    mat4 Projection = projection(fov, resx/float(resy), 0.1, 1000.0);
    mat4 Model = translate(vec3(0.0, 0.0, 0.0));
    mat4 MVP = Projection*View*Model;
    // same RNG seed every framex
    // "minimal standard" LCG RNG
    unsigned int IBM = 123;
    for (int i = 0; i < 100; i++) IBM *= 16807;

    // setup texture
   
    // no mipmaps, no interpolations
 
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
   
    // Get ready to draw objects

    glUseProgram(programID);

    // Draw Third object

    glBindBuffer(GL_ARRAY_BUFFER, quadVertexbuffer);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
    // glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, quadUVbuffer);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,0,(void*)0);
    glBindTexture(GL_TEXTURE_2D, textures[0]);


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
        printf("target = ");
        for (auto t : target)
        {
            printf("%f, ");
        }
        printf("\n");

        testNet.print();
        testNet.updateWeights();

    }


    // Draw network
    // drawNetwork(testNet); TODO mofo



    drawNNet(&testNet, Projection, View, MVP_loc);

    // testNet.draw(Projection, View, MVP_loc);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    // Swap buffers
    glfwSwapBuffers(window);

}



void cleanGL() {
    glDeleteBuffers(1, &quadVertexbuffer);
    glDeleteBuffers(1, &quadUVbuffer);
    glDeleteTextures(2, textures);


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
    // Create and bind  VBO (Vertex Buffer Object). Vertex and element buffers are attached to this. 
    // Can have multiple VBO's
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

    // Create texture data1
    int ctr = 0;
    for (int j = 0; j < textureY; j++) {
        for (int i = 0; i < textureX; i++) {
            float x = i/float(textureX);
            float y = j/float(textureY);
            float arg = 255.0*( (i % 2 -0.5)*(j % 2 -0.5) > 0 ) ;
            data1[ctr++] = arg;
            data1[ctr++] = arg;
            data1[ctr++] = arg;
        }
    }

    // create texture data2
     ctr = 0;
    for (int j = 0; j < textureY; j++) {
        for (int i = 0; i < textureX; i++) {
            float x = i/float(textureX);
            float y = j/float(textureY);
            data2[ctr++] = 0*(0.5 + 0.5*sin(3*2*PI*x));
            data2[ctr++] = 0*(0.5 + 0.5*cos(3*2*PI*y));
            data2[ctr++] = 255*(0.5 + 0.5*sin(3*2*PI*x*y));
        }
    }

    // Generate textures
    glGenTextures(2, textures);

    // Construct first texture
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureX, textureY, 0, GL_RGB, GL_UNSIGNED_BYTE, data1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);      

    // Construct second texture
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureX, textureY, 0, GL_RGB, GL_UNSIGNED_BYTE, data2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);      

    glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(programID, "myTextureSampler"), 0);

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

        // double xpos, ypos;
        // glfwGetCursorPos(win , &xpos, &ypos);
        // printf("mouse pos : (%f, %f) \n", xpos, ypos);
        // printf("also updating deltas\n");
        // std::vector<float> target = {-1.0};
        // std::vector<float> input = {1.0,1.0};
        // testNet.setInputs(input);
        // testNet.forwardPropagate();
        // testNet.backProp(target);
        // testNet.updateWeights();
        
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
    fov = MAX(45, MIN(100.0, fov*zoomFactor));
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
    int layerIndex = 1;
    for (auto layer = net->getInputLayerIt(); layer != net->getLayerEndIt(); layer++)
    {
        // layer.draw(layerID, numLayers, proj, view, mvp_loc);

        int loc = glGetUniformLocation(programID, "nodeFill");
        int nodeIndex = 0;

        glDisable(GL_DEPTH_TEST);
       
        unsigned int numLayers = net->getNumLayers();
        unsigned int numNodes = layer->numNodes;

        for (NNode node : layer->nodes)
        {
            vec3 nodePos = vec3(2.5 + 5.0*(layerIndex - 0.5*numLayers), 2.5 + 5.0*(nodeIndex - 0.5*numNodes), 0.0);
            // Draw node
            // drawElement(float value, vec3 transl);
            glUniform1f(loc, node.out);
            mat4 model = translate( nodePos ); 
            mat4 MVP = proj*view*model;
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &MVP.M[0][0]); 
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            // Draw weights
            for (int i = 0; i < node.numWeights; i++)
            {
                vec3 weightPos = 2.0*nodePos + vec3(-2.0, 1.0 + 2.0*(i - 0.5*node.numWeights), -50.0);
                
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
}