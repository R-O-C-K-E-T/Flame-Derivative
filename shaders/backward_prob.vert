#version 430

flat out uint lineage[LINEAGE_SIZE];
flat out float colour[FUNC_COUNT];

uniform vec2 lower_bound;
uniform vec2 upper_bound;
uniform uint offset;

struct Particle {
	vec2 pos;
	uint lineage[LINEAGE_SIZE];
	float colour[FUNC_COUNT];
	vec2 gradient[PARAM_COUNT]; // list of row major gradients
};

layout (std430, binding = 1) readonly buffer InputBuffer {
	Particle points_in[];
};

void main() {
    const Particle particle = points_in[gl_VertexID];
	
	vec2 pos = (particle.pos - lower_bound) / (upper_bound - lower_bound);
	
	lineage = particle.lineage;
	colour = particle.colour;

	gl_Position = vec4(2 * pos - 1.0, 0, 1);
}