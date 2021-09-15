#version 430

layout(location = 0) out vec4 frag_colour[min(MAX_ATTACHMENTS, (FUNC_COUNT + 3) / 4)];

in vec4 colour[min(MAX_ATTACHMENTS, (FUNC_COUNT + 3) / 4)];

uniform float kernel_size;
uniform vec2 lower_bound;
uniform vec2 upper_bound;


float kernel(float dist) {
    if (dist < 0.5) {
        return 6 * dist * dist * (dist - 1) + 1;
    } else if (dist < 1.0) {
        float inverse = 1.0 - dist;
        return 2 * inverse * inverse * inverse;
    } else {
        return 0.0;
    }
}

float kernel(vec2 offset) {
    return kernel(length(offset));
}

float kernel_grad(float dist) {
    if (dist < 0.5) {
        return 6 * dist * (3 * dist - 2);
    } else if (dist < 1.0) {
        float inverse = 1.0 - dist;
        return -6 * inverse * inverse;
    } else {
        return 0.0;
    }
}

float kernel_grad(vec2 offset, vec2 direction) {
    float dist = length(offset);

    if (dist == 0.0) {
        return 0.0;
    }

    float dist_grad = dot(offset, direction) / dist;
    return kernel_grad(dist) * dist_grad;
}

void main() {
    const vec2 offset = (gl_PointCoord - vec2(0.5)) * 2.0;
    const float factor = kernel(offset);

    for (uint i = 0; i<colour.length; i++) {
        frag_colour[i] = colour[i] * factor;
    }
}