#version 330 core

#define NNET_NODE   0
#define NNET_WEIGHT 1

// interpolated from the vertex shader values. 
in vec3 pos;
// in vec3 normal;
// in vec2 uv;

// uniform sampler2D myTextureSampler;
uniform float nodeFill;



out layout(location = 0) vec3 color; 

uniform float zoom;     // camera zoom (used to compete with aliasing)
uniform bool isSelected; // is the object selected
uniform int object;


void main() {

	vec3 bgColor = vec3(0.1f, 0.1f, 0.25f);

	float x = pos.x;
	float y = pos.y;

	vec3 sumColor = vec3(0.0, 0.0, 0.0);
	 

	float r = x*x + y*y;
 
	float pref = abs(nodeFill)*(1.0 - r)  ;
	pref = (pref > 0.0 ) ? pref : 0.0;

 	// shade of a ring around object
	float zoomComp = (zoom != 0.0) ? 45.0/float(zoom) : 1.0;
	float shade; 
	if (object == NNET_NODE)
	{
		shade = (0.7  - zoomComp*40.0*(r - 0.85)*(r - 0.85));
	}
	else if (object	== NNET_WEIGHT)
	{
		shade = (0.7  - zoomComp*10.0*(r - 0.75)*(r - 0.75));
	}
	else
		shade = 0.0;
	shade = (shade > 0.0) ? shade : 0; 
		
	// color of the object
	vec3 normalColor, selectedColor; 
	if (nodeFill < 0.0)  
	{
		normalColor   = shade*vec3(1.0,1.0,1.0) + pref*vec3(0.3,0.3,1.0) + (1.0 - pref)*bgColor;
		selectedColor = shade*vec3(1.0,1.0,0.0) + pref*vec3(0.3,0.3,1.0) + (1.0 - pref)*bgColor;
		color = (isSelected) ? selectedColor : normalColor;
	}
	else
	{
		normalColor   = shade*vec3(1.0,1.0,1.0) + pref*vec3(1.0,0.3,0.3) + (1.0 - pref)*bgColor;
		selectedColor = shade*vec3(1.0,1.0,0.0) + pref*vec3(1.0,0.3,0.3) + (1.0 - pref)*bgColor;
		color = (isSelected) ? selectedColor : normalColor;
	}

}
 