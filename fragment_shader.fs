#version 330 core

// interpolated from the vertex shader values. 
in vec3 pos;
in vec3 normal;
in vec2 uv;

uniform sampler2D myTextureSampler;
uniform float nodeFill;

out layout(location = 0) vec3 color; 

uniform float scale;  // scale of drawn object
uniform float zoom;   // camera zoom (used to compete with aliasing)

void main() {

	vec3 bgColor = vec3(0.1f, 0.1f, 0.25f);

	float x = pos.x;
	float y = pos.y;

	vec3 sumColor = vec3(0.0, 0.0, 0.0);
	 

	float r = x*x + y*y;
 
	float pref = abs(nodeFill)*(1.0 - r)  ;
	pref = (pref > 0.0 ) ? pref : 0.0;

	float zoomComp = (zoom != 0.0) ? 45.0/float(zoom) : 1.0;

	// zoomComp = 1.0/10.0;

	float shade;
	if (scale == 1.0)
	{
		shade = (0.7  - zoomComp*40.0*(r - 0.85)*(r - 0.85));
	}
	else
	{
		shade = (0.7  - zoomComp*10.0*(r - 0.75)*(r - 0.75));
	}
	shade = (shade > 0.0) ? shade : 0; 
		
	
	if (nodeFill < 0.0) 
	{
		color = shade*vec3(1.0,1.0,1.0) + pref*vec3(0.3,0.3,1.0) + (1.0 - pref)*bgColor;
	}
	else
	{
		color = shade*vec3(1.0,1.0,1.0) + pref*vec3(1.0,0.3,0.3) + (1.0 - pref)*bgColor;
	}


}
 