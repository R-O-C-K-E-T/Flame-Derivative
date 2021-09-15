#version 430

out vec4 colour[min(MAX_ATTACHMENTS, (FUNC_COUNT + 3) / 4)];

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

	for (uint i = 0; i<min(colour.length * 4, FUNC_COUNT - offset); i++) {
		colour[i / 4][i % 4] = particle.colour[i + offset];
	}
	gl_Position = vec4(2 * pos - 1.0, 0, 1);
}