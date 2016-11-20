#version 330 core

layout(location = 0) in vec3 vertexPosition;
// layout(location = 1) in vec3 vertexNormal;
// layout(location = 2) in vec2 vertexUVs;

uniform mat4 MVP;
uniform float scale;

// to fragment shader
out vec3 pos;
// out vec3 normal; 
// out vec2 uv; 

void main() {
	gl_Position =  MVP *  vec4(vertexPosition, 1.0/float(scale));
	pos = vertexPosition;
	// normal = vertexNormal;
	// uv = vertexUVs;
}

