import numpy as np

def perspective_projection_matrix(horizontal_fov_degrees, vertical_fov_degrees, near, far):
    horizontal_fov_radians = np.deg2rad(horizontal_fov_degrees)
    vertical_fov_radians = np.deg2rad(vertical_fov_degrees)
    
    f_horizontal = 1 / np.tan(horizontal_fov_radians / 2)
    f_vertical = 1 / np.tan(vertical_fov_radians / 2)
    
    aspect_ratio = np.tan(horizontal_fov_radians / 2) / np.tan(vertical_fov_radians / 2)
    
    projection_matrix = np.array([
        [f_horizontal, 0, 0, 0],
        [0, f_vertical * aspect_ratio, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, 2 * far * near / (near - far), 0]
    ])
    return projection_matrix

def view_matrix(camera_position, target, up):
    forward = (target - camera_position)
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    
    rotation_matrix = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])
    
    translation_matrix = np.array([
        [1, 0, 0, -camera_position[0]],
        [0, 1, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1]
    ])
    
    view_matrix = rotation_matrix @ translation_matrix
    return view_matrix

def cast_ray_from_screen_pixel(screen_x, screen_y, screen_width, screen_height, horizontal_fov_degrees, vertical_fov_degrees, camera_position, target, up):
    aspect_ratio = screen_width / screen_height
    projection_matrix = perspective_projection_matrix(horizontal_fov_degrees, vertical_fov_degrees, near=0.1, far=1000)
    view_matrix_ = view_matrix(camera_position, target, up)
    
    # Convert screen space coordinates to NDC
    ndc_x = (2 * screen_x / screen_width) - 1
    ndc_y = 1 - (2 * screen_y / screen_height)
    
    # Convert NDC to view space
    view_space = np.linalg.inv(projection_matrix) @ np.array([ndc_x, ndc_y, 0, 1])
    
    # Convert view space to world space
    world_space = np.linalg.inv(view_matrix_) @ view_space
    
    # Calculate ray direction
    ray_direction = world_space[:3] / np.linalg.norm(world_space[:3])
    
    return ray_direction


if __name__ == "__main__":
  # Example usage
  screen_width = 800
  screen_height = 600
  horizontal_fov_degrees = 90
  vertical_fov_degrees = 60
  camera_position = np.array([0.0, 0.0, 0.0])
  target = np.array([0, 0, -1])  # Looking along negative z-axis
  up = np.array([0, 1, 0])

  # Cast ray from pixel (400, 300)
  ray_direction = cast_ray_from_screen_pixel(400, 100, screen_width, screen_height, horizontal_fov_degrees, vertical_fov_degrees, camera_position, target, up)
  print("Ray direction:", ray_direction)
