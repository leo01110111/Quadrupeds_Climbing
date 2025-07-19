import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Config:
    def __init__(self, downsampled_scale, size, horizontal_scale, vertical_scale, noise_range, slope_range, noise_step, platform_width, p_ridge):
        self.downsampled_scale = downsampled_scale
        self.size = size
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.noise_range = noise_range
        self.slope_range = slope_range
        self.noise_step = noise_step
        self.platform_width = platform_width
        self.p_ridge = p_ridge

def pyramid_ridges_terrain(difficulty: float, cfg: Config) -> np.ndarray:
    # check parameters  
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale) #the downsampled must be in [hori, inf)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    height_field_downsampled = np.zeros((width_downsampled,length_downsampled))
    
    if cfg.p_ridge > 0:

        max_noise_height = cfg.noise_range[1] / np.cos(slope)  
        print(f"max_noise {cfg.noise_range[1]} max noise height {max_noise_height}")
        min_noise_height = cfg.noise_range[0] / np.cos(slope)  
        noise_step_height = cfg.noise_step / np.cos(slope)
        # -- height
        height_min = int(min_noise_height / cfg.vertical_scale) #notice how the int casting means that the vert scaling needs to be less than 1 or else we risk height being 0 which will later cause 0 div
        height_max = int(max_noise_height / cfg.vertical_scale) 
        height_step = int(noise_step_height / cfg.vertical_scale) #noise step needs to be greater than vert scale

        # create range of heights possible
        print(f"heightmin {height_min}, heightmax {height_max}, height step {height_step}")
        height_range = np.arange(height_min, height_max + height_step, height_step)
        print("hrightrange:", height_range)
        height_range = height_range[height_range != 0]
        height_range = np.append(height_range,0)
        #probability such that 0 is 1-p_ridge and the rest is p_ridge
        # Calculate the number of non-zero elements
        num_non_zero = len(height_range) - 1  # Since 0 is at the end

        # Initialize probability array
        probabilities = np.zeros(len(height_range))

        # Assign probability p_ridges / num_non_zero to each non-zero element
        if num_non_zero > 0:
            probabilities[:-1] = cfg.p_ridge / num_non_zero

        # Assign probability 1 - p_ridges to the zero element (last element)
        probabilities[-1] = 1 - cfg.p_ridge
        print("hrightrange:", height_range)
        print("probabilities", probabilities)
        # sample heights randomly from the range along a grid
        height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled),p=probabilities)
        #everything need to now have a size of wid downscale x len downscale
        # create interpolation function for the sampled heights

    x = np.linspace(0, cfg.size[0], width_downsampled)
    y = np.linspace(0, cfg.size[1], length_downsampled)
    
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    print(f"slope: {slope}, cfg size {cfg.size[0]}, vert scale {cfg.vertical_scale}")
    height_max = slope * (cfg.size[0] / 2) / cfg.vertical_scale #im still confused by this. We never normalize the units to meter so idk how the sim handles this but this is the same for other terrains so i guess i trust that.
    print("max height in m", height_max)
    # -- center of the terrain
    center_x = cfg.size[0] / 2 # by dividing the element count by 2 ur finding the index not length
    center_y = cfg.size[1] / 2
    print("centerx" ,center_x)
    print("max x ", np.max(x))
    xx, yy = np.meshgrid(x, y, sparse=True)
    print("xx before transform", xx)
    # offset the meshgrid to the center of the terrain
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y # the division of the center height (m) turns it so that the middle is 1\
    print("xx after transform", xx)
    # create a sloped surface
    hf_raw = np.zeros((width_downsampled, length_downsampled))
    hf_raw = height_max * xx * yy
    print("hf: ", hf_raw)
    #print("hf: ", hf_raw)
    #print(height_field_downsampled)

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / 2/ cfg.downsampled_scale)
    # get the height of the platform at the corner of the platform
    x_pf = width_downsampled // 2 - platform_width #// is integer divison
    y_pf = length_downsampled // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf] #tallest location of the height field
    print(f"zpf: {z_pf} hrraw max: {np.max(hf_raw)}")
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    print(f"Hf raw: ", np.shape(hf_raw))
    print(f"Noise field: ", np.shape(height_field_downsampled))

    hf_raw = height_field_downsampled + hf_raw
    func = interpolate.RectBivariateSpline(x, y, hf_raw)

    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) 

     # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    print(f"z sample: {z_upsampled}")
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


def main():
    # Instantiate Config
    cfg = Config(
        downsampled_scale=0.5, #[hori, inf)
        size=(8, 8),
        horizontal_scale=0.1, #(0 to inf)
        vertical_scale=0.005,   #(0,1]
        noise_range=(0.2, 0.2), #a range of heights that the program will randomly choose from. Its the height perpendicular to the slope
        slope_range=(0.0, 0.5),
        noise_step=0.006, #[vert scale, noise max]
        platform_width=2,
        p_ridge = 0.5 #[0,1]
    )

    # Generate height field
    hf = pyramid_ridges_terrain(1.0, cfg)

    # Create meshgrid for plotting
    x = np.linspace(0, cfg.size[0], hf.shape[0])
    y = np.linspace(0, cfg.size[1], hf.shape[1])
    X, Y = np.meshgrid(x, y)

    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, hf, cmap='viridis')

    # Set fixed z-axis limits (e.g., based on expected height range)
    z_min = 0  # Adjust based on your data (e.g., cfg.noise_range[0])
    z_max = 400   # Adjust based on your data (e.g., cfg.noise_range[1] or slope-based max)
    ax.set_zlim(z_min, z_max)

    # Add labels and colorbar
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Height (meters)')
    ax.set_title('Pyramid Ridges Terrain')
    fig.colorbar(surf, ax=ax, label='Height')

    # Show plot
    plt.show()

if __name__ == '__main__':
    main()