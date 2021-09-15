import math, argparse, pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

import numpy as np

import torch

from torch import nn
from torchvision import transforms

from renderer import *

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
    output[..., :, 2] = (torch.sigmoid(input[..., 4:]) - 0.5) * 2
    return output


class Network(nn.Module):
    def __init__(self, renderer: FractalRenderer, use_alpha: bool):
        super().__init__()
        self.renderer = renderer
        self.func_count = renderer.func_count
        self.use_alpha = use_alpha

        self.prob_norm         = nn.parameter.Parameter(torch.randn((1, self.func_count, self.func_count)))
        self.affine_params     = nn.parameter.Parameter(convert_affine(torch.randn((1, self.func_count, 6))))
        self.colour_transform  = nn.parameter.Parameter(torch.randn((1, 3, self.func_count)))
        if not self.use_alpha:
            self.background_colour = nn.parameter.Parameter(torch.randn((1, 3)))
        #self.exposure = nn.parameter.Parameter(torch.tensor(10.0))

    def forward(self, exposure: float, kernel_radius: float=2, gamma: float=2):
        params = self.convert_parameters()
        probability_params, colour_transform = params[:2]

        colour = self.renderer.apply(self.affine_params, probability_params, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), kernel_radius)

        histogram = colour.sum(-3)
        colour = colour / (histogram[:, np.newaxis, ...] + 1e-5)

        normalised_exposure = torch.tensor(exposure * renderer.baseline_exposure / kernel_radius**2)
        if self.use_alpha:
            return postprocess_fractal_alpha(histogram, colour, colour_transform, normalised_exposure, gamma)
        else:
            background_colour = params[2]
            return postprocess_fractal(histogram, colour, colour_transform, background_colour, normalised_exposure, gamma)

    def convert_parameters(self):
        probability_params = torch.sigmoid(self.prob_norm) + 0.001
        probability_params = probability_params.reshape((-1, self.func_count, self.func_count))
        probability_params /= probability_params.sum(-2)[..., np.newaxis, :]

        colour_transform = torch.sigmoid(self.colour_transform)
        background_colour = torch.sigmoid(self.background_colour)

        return probability_params, colour_transform, background_colour

class TextureRenderer:
    def __init__(self, size):
        assert len(size) == 2
        self.size = size
        self.texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *size, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def cleanup(self):
        glDeleteTextures(1, [self.texture])
        self.texture = None

    def set_data(self, data):
        if torch.is_tensor(data):
            data = data.numpy()

        data = np.ascontiguousarray(data)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, *self.size, GL_RGB, GL_FLOAT, data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self, min_x, min_y, width, height):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
            
        glTexCoord2f(0, 0)
        glVertex2f(min_x, min_y)
        
        glTexCoord2f(0, 1)
        glVertex2f(min_x, min_y + height)
        
        glTexCoord2f(1, 1)
        glVertex2f(min_x + width, min_y + height)
        
        glTexCoord2f(1, 0)
        glVertex2f(min_x + width, min_y)

        glEnd()
        glDisable(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

    def __del__(self):
        if self.texture is not None:
            self.cleanup()


def render_high_quality(network, exposure, gamma):
    final_renderer = FractalRenderer(512, 512, network.func_count, max_batch_size=1, particle_count=2048, local_size=32, total_iterations=1 << 22, colour_factor=0.6)
    final_network = Network(final_renderer, network.use_alpha)
    final_network.load_state_dict(network.state_dict())
    with torch.no_grad():
        output = final_network(exposure, 1.5, gamma)
    final_renderer.cleanup()

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts images to fractals')
    parser.add_argument('image', type=argparse.FileType('rb'), help='Image')
    parser.add_argument('--use-alpha', action='store_true', help='Render without a background colour')
    parser.add_argument('--exposure', type=float, default=2.0, help='Amount of exposure to apply to the resulting image (more=fainter image)')
    parser.add_argument('--gamma', type=float, default=2.0, help='Controls the shape of the colour intensity curve')
    parser.add_argument('--function-count', type=int, default=16, help='Number of functions in the fractal')
    args = parser.parse_args()

    pygame.display.set_mode((1024, 1024), OPENGL)

    with args.image as f:
        image = Image.open(f)

        if args.use_alpha:
            image = image.convert('RGBA')
        else:
            image = image.convert('RGB')
    
    image = torch.tensor(np.array(image)) / 255
    image = image.movedim(-1, 0)[np.newaxis, ...]

    print(image.shape)

    target_image_texture = TextureRenderer((image.shape[3], image.shape[2]))
    target_image_texture.set_data(flip_channels(image[0]))

    output_final_texture = TextureRenderer((512, 512))

    resized_image_texture = None
    output_current_texture = None

    configure_opengl()


    #glClear(GL_COLOR_BUFFER_BIT)
    #target_image_texture.draw(1, 1)

    network = None

    # rounds = [
    #     (  8, 0.1,   300),
    #     ( 16, 0.05,  300),
    #     ( 24, 0.01,  400),
    #     ( 32, 0.01,  400),
    #     ( 48, 0.005, 300),
    #     ( 64, 0.005, 300),
    #     ( 96, 0.002, 300),
    #     (128, 0.002, 500)
    # ]
    rounds = [
        (  8, 0.03,   300),
        ( 16, 0.01,   500),
        ( 24, 0.0025, 500),
        ( 32, 0.0025, 600),
        ( 48, 0.001,  400),
        ( 64, 0.001,  300),
        ( 96, 0.0005, 300),
        (128, 0.0005, 250),
        (256, 0.0005, 150),
    ]
    for i, (size, lr, steps) in enumerate(rounds):
        #steps *= 5
        resized_img = transforms.Resize((size, size))(image)
        #resized_mask = transforms.Resize((size, size))(mask)

        total_iterations = size*size * 16
        particle_count = min(total_iterations // 64, 2048)
        local_size = min(particle_count // 4, 32)

        if size == 24: # Eh
            local_size = 18


        if resized_image_texture is not None:
            resized_image_texture.cleanup()
        if output_current_texture is not None:
            output_current_texture.cleanup()

        resized_image_texture = TextureRenderer((resized_img.shape[3], resized_img.shape[2]))
        resized_image_texture.set_data(flip_channels(resized_img[0]))
        output_current_texture = TextureRenderer((resized_img.shape[3], resized_img.shape[2]))
        

        renderer = FractalRenderer(size, size, args.function_count, max_batch_size=1, particle_count=particle_count, local_size=local_size, total_iterations=total_iterations, colour_factor=0.6)

        prev_network = network
        network = Network(renderer, args.use_alpha)
        if prev_network is not None:
            network.load_state_dict(prev_network.state_dict())
            del prev_network
        else:
            output_final_texture.set_data(flip_channels(render_high_quality(network, args.exposure, args.gamma)))
        
        optimiser = torch.optim.Adam(network.parameters(), lr=lr)

        for step in range(steps):
            optimiser.zero_grad()

            output = network(args.exposure, gamma=args.gamma)

            if network.use_alpha:
                image_loss = mse_with_alpha(output, resized_img)
            else:
                image_loss = nn.functional.mse_loss(output, resized_img)

            S = torch.svd(network.affine_params[..., :2]).S
            function_loss = torch.var(S, -1).mean() * 0.05

            probability_loss = torch.var(network.prob_norm, -1).mean() * 0.0 # * 0.0001

            loss = image_loss + function_loss + probability_loss

            if step % 10 == 0:
                print('{} {} {} -> {}'.format(image_loss, function_loss, probability_loss, loss))

            #glfw.make_context_current(None)
            loss.backward()
            #glfw.make_context_current(window)

            optimiser.step()

            output_current_texture.set_data(flip_channels(output.detach()[0]))

            glViewport(0, 0, 1024, 1024)
            glLoadIdentity()
            gluOrtho2D(0, 2, 2, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            resized_image_texture.draw(0, 0, 1, 1)
            output_current_texture.draw(1, 0, 1, 1)
            target_image_texture.draw(0, 1, 1, 1)
            output_final_texture.draw(1, 1, 1, 1)
            pygame.display.flip()

        '''
        import matplotlib.pyplot as plt
        probability_params = network.convert_parameters()[0]
        colour = renderer.apply(network.affine_params, probability_params, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), 2).detach()[0]
        fig, axes = plt.subplots(2, colour.shape[0] // 2)
        for ax, channel in zip(axes.reshape(-1), colour):
            ax.imshow(channel)
        plt.show()'''

        renderer.cleanup()
        
        print()
        print(network.convert_parameters(), sep='\n')
        output_final_texture.set_data(flip_channels(render_high_quality(network, args.exposure, args.gamma)))
        
        '''import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2)

        if args.use_alpha:
            reformat = lambda arr: flip_channels(apply_background(arr), torch.full((1, 3), 1.0)[0])
        else:
            reformat = lambda arr: flip_channels(arr[0])

        axes[0,0].imshow(reformat(resized_img))
        axes[0,1].imshow(reformat(output.detach()))

        axes[1,0].imshow(reformat(image))
        axes[1,1].imshow(reformat(output_b.detach()))

        plt.show()'''

    glViewport(0, 0, 1024, 1024)
    
    pygame.display.set_mode((1024, 1024), OPENGL|RESIZABLE)
    
    waiting = True
    clock = pygame.time.Clock()
    try:
        while waiting:
            for event in pygame.event.get():
                if event.type == QUIT:
                    waiting = False
                elif event.type == VIDEORESIZE:
                    pygame.display.set_mode((event.w, event.w), OPENGL|RESIZABLE)
            
            glLoadIdentity()
            gluOrtho2D(0, 2, 2, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            resized_image_texture.draw(0, 0, 1, 1)
            output_current_texture.draw(1, 0, 1, 1)
            target_image_texture.draw(0, 1, 1, 1)
            output_final_texture.draw(1, 1, 1, 1)

            clock.tick(60)
            pygame.display.flip()
    finally:
        target_image_texture.cleanup()
        output_final_texture.cleanup()
        resized_image_texture.cleanup()
        pygame.quit()

        

