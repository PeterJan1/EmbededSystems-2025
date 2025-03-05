import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Function to extract translation and rotation from a homogeneous transformation matrix
def decompose_homogenous_matrix(T):
    translation = T[:3, 3]
    rotation = T [:3, :3]
    quaternion = R.from_matrix(rotation).as_quat()
    return translation, quaternion

# Create a function to create a cube
def create_cube(position, orientation, color):
    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=color)
    return p.createMultiBody(1, collision_shape_id, visual_shape_id, position, orientation)


if __name__ =="__main__":
    
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    # Load a plane (optional)
    p.loadURDF("plane.urdf")

    # Define initial position and orientation
    initial_position = [0, 0, 0.1]  # Cube starts at (0,0,0.1)
    initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation

    # Create the cube in the initial position
    create_cube(initial_position, initial_orientation, [1, 0, 0, 1])  # Red cube for initial position
    # Define rotation matrices
    theta_z = np.pi / 4  # 45-degree rotation around Z-axis
    theta_x = np.pi / 6  # 30-degree rotation around X-axis

    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0,               0,               1]
    ])

    R_x = np.array([
        [1, 0,              0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x),  np.cos(theta_x)]
    ])

    # Combine rotations: First rotate around Z, then around X
    R_combined = R_x @ R_z  # Matrix multiplication

    # Define the homogeneous transformation matrix
    T = np.eye(4)  # Identity matrix
    T[:3, :3] = R_combined  # Insert the combined rotation matrix
    T[:3, 3] = [0.5, 0.3, 0.4]  # Translation (X=0.5, Y=0.3, Z=0.1)

    # Extract position and orientation from transformation matrix
    new_position, new_orientation = decompose_homogenous_matrix(T)

    # Create the transformed cube
    create_cube(new_position, new_orientation, [0, 1, 0, 1])  # Green cube for transformed position
    

    # Run simulation to observe both cubes
    num_steps = 1000
    axis_length = 0.2
    for t in range(num_steps):
        
        
        p.addUserDebugLine(new_position, new_position + R_combined[:, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(new_position, new_position + R_combined[:, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(new_position, new_position + R_combined[:, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0)  
        p.stepSimulation()
        time.sleep(1)

    p.disconnect()
