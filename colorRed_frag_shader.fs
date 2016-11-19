#version 330 core

// interpolated from the vertex shader values. 
in vec3 pos;
// in vec3 normal;
// in vec2 uv;

uniform float time;

uniform float nodeFill;
uniform float scale;  // scale of drawn object
uniform float zoom;   // camera zoom (used to compete with aliasing)

uniform sampler2D myTextureSampler;

out layout(location = 0) vec3 color; 

void main() {
	color = vec3(1.0, 0.0, 0.0);
}

 