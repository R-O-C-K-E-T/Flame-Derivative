#version 430

layout(location = 0) out vec4 frag_colour[min(MAX_ATTACHMENTS, (PARAM_COUNT + 3) / 4)];

flat in vec2 gradient[min(4 * MAX_ATTACHMENTS, PARAM_COUNT)];
flat in float colour[FUNC_COUNT];

uniform sampler3D grad_in;
//uniform sampler2DArray grad_in;

uniform vec2 kernel_size;
uniform uint batch_offset;

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
    vec2 offset = (gl_PointCoord - vec2(0.5)) * 2.0;
    float dist = length(offset);
    
    if (dist == 0.0) discard;

    float factor = -kernel_grad(dist) / dist;

    ivec2 coord = ivec2(gl_FragCoord.xy);

    float loss_colour_grad = 0.0;
    for (int i = 0; i<FUNC_COUNT; i++) {
        loss_colour_grad += texelFetch(grad_in, ivec3(coord, batch_offset + i), 0).r * colour[i];
    }

    if (loss_colour_grad == 0.0) discard;

    factor *= loss_colour_grad;
    
    for (uint i = 0; i<gradient.length; i++) {
        frag_colour[i / 4][i % 4] = dot(offset, gradient[i] / kernel_size) * factor;
    }
}