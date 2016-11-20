#include <math.h>

#define MIN(a,b)   ((a) < (b) ? (a) : (b))
#define MAX(a,b)   ((a) > (b) ? (a) : (b))


// double PI = 4.0*atan(1.0);
static const double PI = 4.0*atan(1.0);
// #define PI 3.14159265358979323846264338327950288419

struct vec4;

// vec2 declaration
struct vec2 {
    float x, y;

    vec2();
    vec2(float x, float y);


    float length() const;
};

vec2 operator+(const vec2  &, const vec2  &);
vec2 operator+(const float &, const vec2  &);
vec2 operator-(const vec2  &, const vec2  &);
vec2 operator-(const float &, const vec2  &);
vec2 operator*(const vec2  &, const float &);
vec2 operator*(const float &, const vec2  &);


// Declare vec3 and corrensponding operator overloading and functions
// so that i don't have to include a separate library for it (glm.h)
struct vec3 {
    float x, y, z;
    vec3();
    vec3(const float &, const float &, const float &);
    vec3(const vec4 &);
    vec3& operator=(const vec3 &);
};

vec3 operator+(const vec3 &, const vec3 &);
vec3 operator-(const vec3 &, const vec3 &);
vec3 operator-(const vec3 &);
vec3 operator*(const vec3 &, const float &);
vec3 operator*(const float &, const vec3 &);

vec3 normalize(const vec3 &);
float dot(const vec3 &, const vec3 &);
vec3 cross(const vec3 &, const vec3 &);

// Declare vec4 and corrensponding operator overloading and functions
struct vec4 {
    float x, y, z, w;
    vec4();
    vec4(const float &, const float &, const float &, const float &);
    vec4(const vec3 &, const float &);
};

vec4 operator+(const vec4 &, const vec4 &);
vec4 operator-(const vec4 &, const vec4 &);
vec4 operator-(const vec4 &);
vec4 operator*(const vec4 &, const float &);
vec4 operator*(const float &, const vec4 &);

vec4 normalize(const vec4 &);
float dot(const vec4 &, const vec4 &);

// Declare mat4 and corrensponding operator overloading and functions
struct mat4 {
    union {
        struct {float m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44;};
        float m[16];
        float M[4][4];
    };

    mat4() {};
    mat4(const float &);
    mat4(const vec4 &, const vec4 &, const vec4 &, const vec4 &);
};

vec4 operator*(const mat4 &, const vec4 &);
mat4 operator*(const mat4 &, const mat4 &);

mat4 translate(const vec3 &);
mat4 scale(const vec3 &);
mat4 rotate(const vec3 &, const float &);
mat4 view(const vec3 &, const vec3 &, const vec3 &, const vec3 &);
mat4 lookAt(const vec3 &, const vec3 &, const vec3 &);
mat4 projection(const float &, const float &, const float &, const float &);
mat4 ortho(float x1, float x2, float y1, float y2, float z1, float z2);




