

import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# Function to extract translation and rotation from a homogeneous transformation matrix
def decompose_homogeneous_matrix(T):
    translation = T[:3, 3]  # Extract XYZ position
    rotation_matrix = T[:3, :3]  # Extract rotation
    quaternion = R.from_matrix(rotation_matrix).as_quat() # Convert to Euler angles
    # Convert to quaternion
    return translation, quaternion  # Return position and quaternion orientation

if __name__ =="__main__":

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load ground plane
    p.loadURDF("plane.urdf")

    # Load 2-DOF arm from URDF
    robot_id = p.loadURDF("urdf/3dof_planar_robot.urdf", basePosition=[0, 0, 0], useFixedBase=True)

    # Joint indices in URDF (PyBullet assigns indices to joints)
    joint1_idx = 1  # First revolute joint
    joint2_idx = 2  # Second revolute joint
    joint3_idx = 3
    # Define joint angles (theta1, theta2) in radians
    theta1 = np.deg2rad(-45)  # Rotate joint 1 by 30 degrees
    theta2 = np.deg2rad(45.0)  # Rotate joint 2 by -45 degrees
    theta3 = np.deg2rad(10.0)

    # Set joint angles in PyBullet
    p.setJointMotorControl2(robot_id, joint1_idx, controlMode=p.POSITION_CONTROL, targetPosition=theta1)
    p.setJointMotorControl2(robot_id, joint2_idx, controlMode=p.POSITION_CONTROL, targetPosition=theta2)
    p.setJointMotorControl2(robot_id, joint3_idx, controlMode=p.POSITION_CONTROL, targetPosition=theta3)

    # Compute forward kinematics: Transformation Matrices
    L1 = 1.0  # Length of first link
    L2 = 1.0  # Length of second link
    L3 = 1.0  # Length of third link
    # Transformation from world to base
    T_W_Base = np.eye(4)
    T_W_Base[2,3] = 0.05
    
    # Transformation from base to link1
    T1 = np.array([
        [np.cos(theta1),    0,     np.sin(theta1),  L1 * np.sin(theta1)],
        [0,                 1,     0,               0],
        [-np.sin(theta1),   0,     np.cos(theta1),  L1 * np.cos(theta1)],
        [0,                 0,     0,               1]
    ])

    # Transformation from link1 to link2
    T2 = np.array([
        [np.cos(theta2),    0,     np.sin(theta2),  L2 * np.sin(theta2)],
        [0,                 1,     0,               0],
        [-np.sin(theta2),   0,     np.cos(theta2),  L2 * np.cos(theta2)],
        [0,                 0,     0,               1]
    ])
    T3 = np.array([
        [np.cos(theta3),    0,     np.sin(theta3),  L3 * np.sin(theta3)],
        [0,                 1,     0,               0],
        [-np.sin(theta3),   0,     np.cos(theta3),  L3 * np.cos(theta3)],
        [0,                 0,     0,               1]
    ])

    # Compute the transformation matrices relative to the world
    T_link1 = T_W_Base @ T1  # Link1 relative to world
    T_link2 = T_link1 @ T2  # Link2 relative to world
    T_link3 = T_link2 @ T3
    print(T3)


    # Extract position and orientation for each link
    link1_position, link1_orientation = decompose_homogeneous_matrix(T_link1)
    link2_position, link2_orientation = decompose_homogeneous_matrix(T_link2)
    link3_position, link3_orientation = decompose_homogeneous_matrix(T_link3)
    # ee_position, ee_orientation = decompose_homogeneous_matrix(T_ee)

    # Print results
    print("Link1 Position:", link1_position)
    print("Link2 Position:", link2_position)
    print("Link3 Position:", link3_position)

    # Create a sphere at the correct end-effector position
    sphere_collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.05)
    sphere_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_collision_shape, baseVisualShapeIndex=sphere_visual_shape, basePosition=link3_position)

    # Run simulation for visualization
    axis_length = 0.2
    for _ in range(10000):  # Run long enough to visualize movement
        p.addUserDebugLine(link1_position, link1_position + T_link1[:3, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link1_position, link1_position + T_link1[:3, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link1_position, link1_position + T_link1[:3, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0)  
        
        # link2
        p.addUserDebugLine(link2_position, link2_position + T_link2[:3, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link2_position, link2_position + T_link2[:3, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link2_position, link2_position + T_link2[:3, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0)  
        
        # link3
        p.addUserDebugLine(link3_position, link3_position + T_link3[:3, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link3_position, link3_position + T_link3[:3, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link3_position, link3_position + T_link3[:3, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0)  
        
        p.stepSimulation()
        time.sleep(1 / 240)  # Slow down for visualization

    # Keep the GUI open
    input("Press Enter to exit...")
    p.disconnect()


