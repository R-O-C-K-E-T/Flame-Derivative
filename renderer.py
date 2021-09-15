import math, re, time, ctypes
from os import path

import numpy as np

from OpenGL.GL import *
from OpenGL.GL import shaders

import torch
from torch import nn

def flip_channels(image):
    return torch.movedim(image, -3, -1)

def gen_texture_2d():
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    return texture

def gen_texture_3d():
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_3D, texture)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    return texture

def tensor_from_texture_3d(texture, depth): # 4 channels
    glBindTexture(GL_TEXTURE_3D, texture)

    width  = int(glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_WIDTH))
    height = int(glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_HEIGHT))

    glBindTexture(GL_TEXTURE_3D, 0)
    
    result = np.empty((depth, height, width, 4), dtype=np.float32)
    glGetTextureSubImage(texture, 0, 0, 0, 0, width, height, depth, GL_RGBA, GL_FLOAT, result.nbytes, result.data)
    return torch.from_numpy(result)

def tensor_to_texture_3d(tensor, texture): # Single channel
    glBindTexture(GL_TEXTURE_3D, texture)
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, *tensor.shape[::-1], GL_RED, GL_FLOAT, tensor.cpu().contiguous().numpy())
    glBindTexture(GL_TEXTURE_3D, 0)

def compile_shader(text, shader_type, defines={}):
    def add_globals(string):
        header = ''
        for key, val in defines.items():
            if isinstance(val, bool):
                if val:
                    header += '#define {}\n'.format(key)
                else:
                    header += '#undef {}\n'.format(key)
            else:
                val = str(val).replace('\n', '\\\n')
                header += '#define {} {}\n'.format(key, val)

        index = string.index('\n')+1
        return string[:index] + header + string[index:]

    text = add_globals(text)

    #print(text)
    #print('\n\n\n\n')

    #return shaders.compileShader(text, type)
    try:
        return shaders.compileShader(text, shader_type)
    except RuntimeError as e:
        lines = text.split('\n')
        for cause in e.args[0].split('\\n'):
            print(cause)
            match = re.search('0\\(([0-9]+)\\)', cause)
            if match is None:
                continue
            line = int(match[1]) - 1
            print(*lines[line-1:line+2], sep='\n')
    raise RuntimeError("Compilation Failed")

def round_up(value, multiple):
    return math.ceil(value / multiple) * multiple

def ssbo_size(program, resource_index):
    return glGetProgramResourceiv(program, GL_SHADER_STORAGE_BLOCK, resource_index, 1, [GL_BUFFER_DATA_SIZE], 1)[1]

class TextureSummer:
    def __init__(self, local_size=1):
        defines = {
            'LOCAL_SIZE' : local_size,
        }
        shader = compile_shader(open(path.join('shaders', 'sum.comp')).read(), GL_COMPUTE_SHADER, defines)
        self.program = shaders.compileProgram(shader)
        glDeleteShader(shader)

        self.buffer = glGenBuffers(1)

    def __call__(self, texture, layers):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffer)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * 4 * layers, None, GL_DYNAMIC_DRAW)
        glUseProgram(self.program)

        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.buffer)

        glDispatchCompute(layers, 1, 1)

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glUseProgram(0)
        
        data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * 4 * layers)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        return torch.from_numpy(np.frombuffer(data, dtype=np.float32))

    def cleanup(self):
        glDeleteBuffers(1, [self.buffer])
        glDeleteProgram(self.program)


class FractalRenderer:
    def __init__(self, width, height, func_count, /, max_batch_size=16, total_iterations=(1 << 17), particle_count=2048, local_size=128, lineage_size=4, colour_factor=0.5):
        assert particle_count > 0 and particle_count % local_size == 0
        assert total_iterations % particle_count == 0
        self.width = width
        self.height = height
        self.func_count = func_count
        self.param_count = func_count * 6
        
        self.max_batch_size = max_batch_size
        self.particle_count = particle_count
        self.local_size = local_size
        self.iterations = total_iterations // particle_count
        self.lineage_size = lineage_size

        self.max_attachments = glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS)

        self.texture_summer = TextureSummer()

        defines = {
            'LOCAL_SIZE'      : self.local_size,
            'FUNC_COUNT'      : self.func_count,
            'PARAM_COUNT'     : self.param_count,
            'ITERATIONS'      : self.iterations,
            'MAX_ATTACHMENTS' : self.max_attachments,
            'LINEAGE_SIZE'    : self.lineage_size,
            'FACTOR'          : colour_factor,
        }

        comp_shader = compile_shader(open(path.join('shaders', 'iterate.comp')).read(), GL_COMPUTE_SHADER, defines)
        self.iterate_prog = shaders.compileProgram(comp_shader)
        glDeleteShader(comp_shader)

        self.cutoffs_uniform   = glGetUniformLocation(self.iterate_prog, 'cutoffs')
        self.functions_uniform = glGetUniformLocation(self.iterate_prog, 'functions')


        vert_shader = compile_shader(open(path.join('shaders', 'forward.vert')).read(), GL_VERTEX_SHADER, defines)
        frag_shader = compile_shader(open(path.join('shaders', 'forward.frag')).read(), GL_FRAGMENT_SHADER, defines)
        self.forward_prog = shaders.compileProgram(vert_shader, frag_shader)
        glDeleteShader(vert_shader)
        glDeleteShader(frag_shader)

        vert_shader = compile_shader(open(path.join('shaders', 'backward_param.vert')).read(), GL_VERTEX_SHADER, defines)
        frag_shader = compile_shader(open(path.join('shaders', 'backward_param.frag')).read(), GL_FRAGMENT_SHADER, defines)
        self.backward_param_prog = shaders.compileProgram(vert_shader, frag_shader, validate=False)
        glDeleteShader(vert_shader)
        glDeleteShader(frag_shader)

        vert_shader = compile_shader(open(path.join('shaders', 'backward_prob.vert')).read(), GL_VERTEX_SHADER, defines)
        frag_shader = compile_shader(open(path.join('shaders', 'backward_prob.frag')).read(), GL_FRAGMENT_SHADER, defines)
        self.backward_prob_prog = shaders.compileProgram(vert_shader, frag_shader)
        glDeleteShader(vert_shader)
        glDeleteShader(frag_shader)

        self.lower_bound_uniform = [glGetUniformLocation(prog, 'lower_bound') for prog in (self.forward_prog, self.backward_param_prog, self.backward_prob_prog)]
        self.upper_bound_uniform = [glGetUniformLocation(prog, 'upper_bound') for prog in (self.forward_prog, self.backward_param_prog, self.backward_prob_prog)]
        self.offset_uniform = [glGetUniformLocation(prog, 'offset') for prog in (self.forward_prog, self.backward_param_prog, self.backward_prob_prog)]
        self.kernel_size_uniform = glGetUniformLocation(self.backward_param_prog, 'kernel_size')
        self.grad_in_uniform = [glGetUniformLocation(prog, 'grad_in') for prog in (self.backward_param_prog, self.backward_prob_prog)]
        self.batch_offset_uniform = [glGetUniformLocation(prog, 'batch_offset') for prog in (self.backward_param_prog, self.backward_prob_prog)]
        self.probabilities_uniform = glGetUniformLocation(self.backward_prob_prog, 'probabilities')
        self.particle_offset_uniform = glGetUniformLocation(self.iterate_prog, 'offset')

        item_size_in_bytes = ssbo_size(self.iterate_prog, 1)
        assert item_size_in_bytes % 4 == 0
        self.item_size = item_size_in_bytes // 4

        self.initial_random_buf, self.point_buffer = glGenBuffers(2)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.initial_random_buf)
        initial_random = np.random.bytes(4 * 4 * self.particle_count)
        glBufferData(GL_SHADER_STORAGE_BUFFER, len(initial_random), initial_random, GL_DYNAMIC_DRAW)
        #glBufferStorage(GL_SHADER_STORAGE_BUFFER, len(initial_random), initial_random, 0)
        del initial_random

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.point_buffer)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * self.total_iterations * self.item_size * self.max_batch_size, None, GL_DYNAMIC_DRAW)
        #glBufferStorage(GL_SHADER_STORAGE_BUFFER, 4 * self.total_iterations * self.item_size * self.max_batch_size, None, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        layers = math.ceil(self.func_count / 4)    
        self.forward_texture = gen_texture_3d()
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, self.width, self.height, layers * self.max_batch_size, 0, GL_RGBA, GL_FLOAT, None)

        layers = math.ceil(self.param_count / 4)    
        self.backward_param_texture = gen_texture_3d()
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, self.width, self.height, layers * self.max_batch_size, 0, GL_RGBA, GL_FLOAT, None)

        layers = math.ceil(self.func_count * self.func_count / 4)
        self.backward_prob_texture = gen_texture_3d()
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, self.width, self.height, layers * self.max_batch_size, 0, GL_RGBA, GL_FLOAT, None)

        self.grad_in_texture = gen_texture_3d()
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, self.width, self.height, self.max_batch_size * self.func_count, 0, GL_RED, GL_FLOAT, None)

        self.current_invocation_id = 0

        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, params, probabilities, lower_bound, upper_bound, kernel_radius):
                result = self.forward(params, probabilities, lower_bound, upper_bound, kernel_radius)

                ctx.kernel_radius = kernel_radius
                ctx.save_for_backward(lower_bound, upper_bound, probabilities)

                self.current_invocation_id += 1
                ctx.invocation_id = self.current_invocation_id
                
                return result.to(device=params.device)

            @staticmethod
            def backward(ctx, grad_in):
                assert self.current_invocation_id == ctx.invocation_id
                lower_bound, upper_bound, probabilities = ctx.saved_tensors

                #glfw.make_context_current(window)

                batch_size = len(grad_in)
                
                tensor_to_texture_3d(grad_in.reshape((-1, self.height, self.width)), self.grad_in_texture)

                if ctx.needs_input_grad[0]:
                    param_out = self.backward_param(batch_size, lower_bound, upper_bound, ctx.kernel_radius).to(device=grad_in.device)
                else:
                    param_out = None
                
                if ctx.needs_input_grad[1]:
                    prob_out = self.backward_prob(batch_size, lower_bound, upper_bound, ctx.kernel_radius, probabilities).to(device=grad_in.device)
                else:
                    prob_out = None

                #glfw.make_context_current(None)

                return param_out, prob_out, None, None, None

        self.apply = Function.apply
    
    @property
    def total_iterations(self):
        return self.iterations * self.particle_count

    @property
    def baseline_exposure(self):
        return self.total_iterations / (self.width * self.height)

    def dump_points(self, points_buf):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, points_buf)
        data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * self.total_iterations * self.item_size)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return np.frombuffer(data, dtype=np.float32).reshape((self.total_iterations, self.item_size))

    def forward(self, params: torch.Tensor, probabilities: torch.Tensor, lower_bound=torch.tensor([-1.0, -1.0]), upper_bound=torch.tensor([1.0, 1.0]), kernel_radius=2.0):
        # kernel_radius in pixels
        assert params.shape[0] == probabilities.shape[0]
        assert params.shape[0] <= self.max_batch_size
        assert probabilities.shape[1:] == (self.func_count, self.func_count)
        assert params.shape[1:] == (self.func_count, 2, 3)

        batch_size = len(params)

        glUseProgram(self.iterate_prog)

        # Generate points with colours+gradients

        cutoffs = torch.cumsum(probabilities, -2)
        del probabilities

        assert torch.allclose(cutoffs[:, -1, :].cpu(), torch.tensor(1.0))
        cutoffs[:, -1, :] = 1.0

        cutoffs = cutoffs.transpose(-1, -2).reshape(-1, self.func_count*self.func_count).cpu().contiguous()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.initial_random_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.point_buffer)

        params = params.cpu().numpy()
        for i in range(len(params)):
            glUniform1fv(self.cutoffs_uniform, cutoffs.shape[1], cutoffs[i].numpy())
            glUniformMatrix3x2fv(self.functions_uniform, self.param_count, GL_TRUE, params[i])
            glUniform1ui(self.particle_offset_uniform, self.total_iterations * i)

            glDispatchCompute(self.particle_count // self.local_size, 1, 1)

        del params, cutoffs

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glUseProgram(self.forward_prog)

        # Draw points with kernel
        layers = math.ceil(self.func_count / 4)    
        glClearTexImage(self.forward_texture, 0, GL_RGBA, GL_FLOAT, (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0))

        framebuffer = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

        glUniform2fv(self.lower_bound_uniform[0], 1, lower_bound.cpu().numpy())
        glUniform2fv(self.upper_bound_uniform[0], 1, upper_bound.cpu().numpy())
        glPointSize(2 * kernel_radius)

        glViewport(0, 0, self.width, self.height)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.point_buffer)
        for i in range(batch_size):
            
            for offset in range(0, layers, self.max_attachments):
                glUniform1ui(self.offset_uniform[0], offset * 4)

                buffers = []
                for attachment in range(min(self.max_attachments, layers - offset)):
                    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachment, self.forward_texture, 0, attachment + offset + i*layers)
                    buffers.append(GL_COLOR_ATTACHMENT0 + attachment)
                glDrawBuffers(len(buffers), buffers)

                glDrawArrays(GL_POINTS, self.total_iterations * i, self.total_iterations)


        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [framebuffer])
        glUseProgram(0)

        colour = tensor_from_texture_3d(self.forward_texture, layers * batch_size)
        colour = torch.movedim(colour, -1, 1).reshape((-1, 4*layers, self.height, self.width))[:, :self.func_count, :, :]
        return colour
    
    def backward_param(self, batch_size, lower_bound, upper_bound, kernel_radius):
        glUseProgram(self.backward_param_prog)

        glUniform2fv(self.lower_bound_uniform[1], 1, lower_bound.cpu().numpy())
        glUniform2fv(self.upper_bound_uniform[1], 1, upper_bound.cpu().numpy())
        glPointSize(2 * kernel_radius)

        glViewport(0, 0, self.width, self.height)

        glUniform2fv(self.kernel_size_uniform, 1, (kernel_radius * (upper_bound.cpu() - lower_bound.cpu()) / torch.tensor([self.width, self.height])).numpy())
        

        layers = math.ceil(self.param_count / 4)    
        glClearTexImage(self.backward_param_texture, 0, GL_RGBA, GL_FLOAT, (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0))

        framebuffer = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

        glUniform1i(self.grad_in_uniform[0], 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.grad_in_texture)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.point_buffer)
        for i in range(batch_size):
            glUniform1ui(self.batch_offset_uniform[0], i * self.func_count)

            for offset in range(0, layers, self.max_attachments):
                glUniform1ui(self.offset_uniform[1], offset * 4)

                buffers = []
                for attachment in range(min(self.max_attachments, layers - offset)):
                    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+attachment, self.backward_param_texture, 0, attachment + offset + i * layers)
                    buffers.append(GL_COLOR_ATTACHMENT0 + attachment)
                glDrawBuffers(len(buffers), buffers)

                glDrawArrays(GL_POINTS, self.total_iterations*i, self.total_iterations)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [framebuffer])
        glUseProgram(0)

        grad_out = self.texture_summer(self.backward_param_texture, batch_size * layers)
        grad_out = grad_out.reshape((batch_size, -1))
        grad_out = grad_out[:, :self.param_count]
        grad_out = grad_out.reshape((batch_size, self.func_count, 2, 3))

        return grad_out

    def backward_prob(self, batch_size, lower_bound, upper_bound, kernel_radius, probabilities):
        glUseProgram(self.backward_prob_prog)

        glUniform2fv(self.lower_bound_uniform[2], 1, lower_bound.cpu().numpy())
        glUniform2fv(self.upper_bound_uniform[2], 1, upper_bound.cpu().numpy())
        glPointSize(2 * kernel_radius)

        framebuffer = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

        glViewport(0, 0, self.width, self.height)

        probabilities = probabilities.transpose(-1, -2).reshape(-1, self.func_count*self.func_count).cpu().contiguous().numpy()
        
        glUniform1i(self.grad_in_uniform[1], 0)
        
        layers = math.ceil(self.func_count * self.func_count / 4)    
        glClearTexImage(self.backward_prob_texture, 0, GL_RGBA, GL_FLOAT, (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0))
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.grad_in_texture)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.point_buffer)
        for i in range(batch_size):
            glUniform1ui(self.batch_offset_uniform[0], i * self.func_count)
            glUniform1fv(self.probabilities_uniform, self.func_count*self.func_count, probabilities[i])
            
            for offset in range(0, layers, self.max_attachments):
                glUniform1ui(self.offset_uniform[2], offset * 4)

                buffers = []
                for attachment in range(min(self.max_attachments, layers - offset)):
                    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+attachment, self.backward_prob_texture, 0, attachment + offset + i*layers)
                    buffers.append(GL_COLOR_ATTACHMENT0 + attachment)
                glDrawBuffers(len(buffers), buffers)

                glDrawArrays(GL_POINTS, i * self.total_iterations, self.total_iterations)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [framebuffer])
        glUseProgram(0)
        
        grad_out = self.texture_summer(self.backward_prob_texture, batch_size * layers)
        grad_out = grad_out.reshape((batch_size, -1))
        grad_out = grad_out[:, :(self.func_count*self.func_count)]
        grad_out = grad_out.reshape((batch_size, self.func_count, self.func_count)).transpose(-1, -2)

        return grad_out
        

    def cleanup(self):
        glDeleteBuffers(2, [self.initial_random_buf, self.point_buffer])
        glDeleteProgram(self.iterate_prog)
        glDeleteProgram(self.forward_prog)
        glDeleteProgram(self.backward_param_prog)
        glDeleteProgram(self.backward_prob_prog)
        glDeleteTextures(3, [self.forward_texture, self.backward_param_texture, self.backward_prob_texture])

def make_rotation(input):
    output = input.new_empty((*input.shape, 2, 2))
    cos = torch.cos(input)
    sin = torch.sin(input)
    output[..., 0, 0] = cos
    output[..., 0, 1] = -sin
    output[..., 1, 0] = sin
    output[..., 1, 1] = cos
    return output

def convert_affine(input):
    assert input.shape[-1] == 6

    sigmoided = torch.sigmoid(input[..., :4])

    transformed = input.new_zeros((*input.shape[:-1], 2, 2))
    transformed[..., 0, 0] = sigmoided[..., 0] * 0.8 + 0.05

    transformed = make_rotation(sigmoided[..., 1] * (math.pi * 0.4)) @ transformed
    # transformed[..., 1, 1] == 0
    transformed[..., 1, 1] = sigmoided[..., 2] * 0.8 + 0.05

    transformed = make_rotation(sigmoided[..., 3] * (math.pi * 2.0)) @ transformed

    output = input.new_empty((*input.shape[:-1], 2, 3))
    output[..., :, :2] = transformed
    output[..., :, 2] = input[..., 4:]
    return output

def postprocess_fractal(histogram, colour, colour_transform, background_colour, exposure, gamma):
    alpha = (torch.log(histogram + 1.0) / torch.log(exposure)).clamp(max=1.0)**gamma

    reduced_colour = torch.transpose((colour_transform[..., np.newaxis, np.newaxis, :, :] @ torch.transpose(colour, -3, -1)[..., np.newaxis])[..., 0], -3, -1)
    return reduced_colour * alpha[..., np.newaxis, :, :] + background_colour[..., :, np.newaxis, np.newaxis] * (1 - alpha)[..., np.newaxis, :, :]

def postprocess_fractal_alpha(histogram, colour, colour_transform, exposure, gamma):
    result = colour.new_empty((colour.shape[0], 4, *colour.shape[2:]))
    result[:, -1, :, :] = (torch.log(histogram + 1.0) / torch.log(exposure)).clamp(max=1.0)**gamma
    result[:, :3, :, :] = torch.transpose((colour_transform[..., np.newaxis, np.newaxis, :, :] @ torch.transpose(colour, -3, -1)[..., np.newaxis])[..., 0], -3, -1)
    return result

def mse_with_alpha(input_image, target_image, /, alpha_importance=0.5):
    input_alpha = input_image[:, -1, ...]
    target_alpha = target_image[:, -1, ...]
    return nn.functional.mse_loss(input_alpha, target_alpha) * alpha_importance + nn.functional.mse_loss(input_image[:, :-1, ...] * target_alpha, target_image[:, :-1, ...] * target_alpha) * (1 - alpha_importance)

def apply_background(image, background_colour):
    alpha = image[:, -1, np.newaxis, ...]
    return alpha * image[:, :-1, ...] + (1 - alpha) * background_colour[..., :, np.newaxis, np.newaxis]

def configure_opengl():
    glEnable(GL_BLEND)
    glBlendFunc(GL_ONE, GL_ONE)
    
    glEnable(GL_POINT_SPRITE)
    glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.features(x)


class FractalNetwork(nn.Module):
    def __init__(self, renderer: FractalRenderer):
        super().__init__()
        self.renderer = renderer
        self.func_count = renderer.func_count
        self.generator = nn.Sequential(
            VGGBlock(3, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, self.func_count * (6 + 3) + 3 + self.func_count * self.func_count),
        )
        self.kernel_size = 2.0

    def forward(self, x):
        x = self.generator(x)
        #x = torch.randn_like(x)

        affine_params, probability_params, colour_params = torch.split(x, [self.func_count * 6, self.func_count*self.func_count, (self.func_count+1) * 3], -1)



        affine_params = convert_affine(affine_params.reshape((-1, self.func_count, 6)))

        probability_params = torch.sigmoid(probability_params) + 0.001
        probability_params = probability_params.reshape((-1, self.func_count, self.func_count))
        probability_params /= probability_params.sum(-2)[..., np.newaxis, :]

    
        colour_params = torch.sigmoid(colour_params)
        #colour_params = torch.rand_like(colour_params)

        colour_transform  = colour_params[:, :-3].reshape((-1, 3, self.func_count))
        background_colour = colour_params[:, -3:]

        #fractal_device = torch.device('cpu')
        #affine_params = affine_params.to(device=fractal_device)
        #probability_params = probability_params.to(device=fractal_device)

        #colour = torch.empty((len(x), self.func_count, 32, 32), device=fractal_device)
        colour = self.renderer.apply(
            affine_params, 
            probability_params, 
            torch.tensor([-1.0, -1.0]), 
            torch.tensor([1.0, 1.0]), 
            self.kernel_size
        )

        #colour = colour.to(device=x.device)

        histogram = colour.sum(-3)
        colour = colour / (histogram[:, np.newaxis, ...] + 1e-5)

        image = postprocess_fractal(histogram, colour, colour_transform, background_colour, 50.0 * self.renderer.baseline_exposure, 2.0)
        return image

def total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def split_range(start, stop, segment_size):
    big_steps, rem = divmod(stop - start, segment_size)

    for i in range(big_steps):
        yield range(i * segment_size, (i + 1) * segment_size)
    yield range(stop - rem, stop)

def main():
    from matplotlib import pyplot as plt

    configure_opengl()

    func_count = 8
    batch_size = 32

    # 8 Functions GTX1060 6G -> particle_count=2048, local_size=32
    renderer = FractalRenderer(32, 32, func_count, max_batch_size=batch_size, particle_count=2048, local_size=32, total_iterations=1<<19)

    '''affine_params = torch.randn((batch_size, func_count * 6))
    probability_params = torch.randn((batch_size, func_count * func_count))

    def compute(affine_params, probability_params):
        affine_params = convert_affine(affine_params.reshape((-1, func_count, 6)))

        probability_params = torch.sigmoid(probability_params) + 0.001
        probability_params = probability_params.reshape((-1, func_count, func_count))
        probability_params /= probability_params.sum(-2)[..., np.newaxis, :]

        return renderer.apply(affine_params, probability_params, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), 2.0)

    baseline = compute(affine_params, probability_params)

    affine_params += torch.randn_like(affine_params) * 0.1
    probability_params += torch.randn_like(probability_params)

    affine_params.requires_grad_(True)
    probability_params.requires_grad_(True)


    optimiser = torch.optim.Adam([affine_params, probability_params], lr=0.01)
    for _ in range(500):
        optimiser.zero_grad()

        result = compute(affine_params, probability_params)

        #loss = nn.functional.mse_loss(result, baseline)

        losses = torch.zeros((len(result)))
        for i in range(len(result)):
            losses[i] = nn.functional.mse_loss(result[i], baseline[i])

        loss = losses.mean()
        #glfw.make_context_current(None)
        loss.backward()
        #glfw.make_context_current(window)

        print(losses.detach().cpu().numpy().tolist())
        #print(loss.detach().cpu().numpy().tolist())
        #print(affine_params.grad)

        optimiser.step()

    return'''

    '''for img, ax in zip(result[0], plt.subplots(1, func_count)[1]):
        ax.imshow(img)'''

    '''for row_a, row_b in zip(result, plt.subplots(batch_size, func_count)[1]):
        for img, ax in zip(row_a, row_b):
            ax.imshow(img.detach())
    plt.show()'''

    device = torch.device('cpu')
    #device = torch.device('cuda')

    model = FractalNetwork(renderer)
    model.to(device=device)

    # 1024
    #  1 -> 32.52
    #  8 -> 21.96, 22.34
    # 16 -> 22.00s
    # 32 -> 21.71s

    t0 = time.time()
    for _ in range(512 // batch_size):
        batch = torch.randn((batch_size, 3, 32, 32), device=device)

        output = model(batch)

        loss = output.sum()

        # The backward pass may be running on a different thread, hence we need to hand the context over
        #glfw.make_context_current(None)
        loss.backward()
        #glfw.make_context_current(window)
    t1 = time.time()
    print(t1 - t0)

if __name__ == '__main__':
    with offscreen_context() as window:
        main()
