#version 430

layout(location = 0) out vec4 frag_colour[min(MAX_ATTACHMENTS, (FUNC_COUNT*FUNC_COUNT + 3) / 4)];

flat in uint lineage[LINEAGE_SIZE];
flat in float colour[FUNC_COUNT];

uniform sampler3D grad_in;

uniform vec2 lower_bound;
uniform vec2 upper_bound;

uniform uint offset;
uniform uint batch_offset;

uniform float probabilities[FUNC_COUNT * FUNC_COUNT];


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

float kernel(vec2 v) {
    return kernel(length(v));
}


void main() {
    float factor = kernel((gl_PointCoord - vec2(0.5)) * 2.0);

    if (factor == 0.0) discard;

    vec2 coord = ivec2(gl_FragCoord.xy);

    float loss_colour_grad = 0.0;
    for (int i = 0; i<FUNC_COUNT; i++) {
        loss_colour_grad += texelFetch(grad_in, ivec3(coord, batch_offset + i), 0).r * colour[i];
    }

    if (loss_colour_grad == 0.0) discard;

    factor *= loss_colour_grad;

    for (uint i = 0; i<(frag_colour.length * 4); i++) {
        float influence = 0.0;
        for (uint j = 0; j<lineage.length; j++) {
            if (lineage[j] == i + offset) {
                influence += 1.0;
            }
        }
        frag_colour[i / 4][i % 4] = influence * factor / probabilities[i + offset];
    }
}