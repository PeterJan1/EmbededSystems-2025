import pybullet as p
import pybullet_data
import numpy as np 
from scipy.spatial.transform import Rotation as R
import time


def decompose_homogenous_matrix(T):
    translation = T[:3, 3]
    rotation = T [:3, :3]
    quaternion = R.from_matrix(rotation).as_quat()
    return translation, quaternion

if __name__=="__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.loadURDF("plane.urdf")
    
    robot_id = p.loadURDF("urdf/2dof_planar_robot.urdf", basePosition=[0,0,0], useFixedBase=True)
    
    joint1_idx = 1
    joint2_idx = 2
    
    theta1 = np.deg2rad(0)
    theta2 = np.deg2rad(0)
    
    p.setJointMotorControl2(robot_id, joint1_idx, controlMode=p.POSITION_CONTROL, targetPosition=theta1)
    p.setJointMotorControl2(robot_id, joint2_idx, controlMode=p.POSITION_CONTROL, targetPosition=theta2)
    
    L1 = 1
    L2 = 1
    
    T_W_Base = np.eye(4)
    T_W_Base[2, 3] = 0.05
    
    T1 = np.array([[np.cos(theta1),  0,  np.sin(theta1),  L1*np.sin(theta1)],
                  [0,               1,  0,               0                ],
                  [-np.sin(theta1), 0, np.cos(theta1),   L1*np.cos(theta1)],
                  [0,               0, 0,                1                ]])
      
    T2 = np.array([[np.cos(theta2),  0,  np.sin(theta2),  L2*np.sin(theta2)],
                    [0,               1,  0,               0                ],
                    [-np.sin(theta2), 0, np.cos(theta2),   L2*np.cos(theta2)],
                    [0,               0, 0,                1                ]])
    
    T_link1 = T_W_Base @ T1
    T_link2 = T_link1 @ T2
    
    link1_position, link1_orientation = decompose_homogenous_matrix(T_link1)
    link2_position, link2_orientation = decompose_homogenous_matrix(T_link2)
    
    print("Link1 Position:", link1_position)
    print("Link2 Position:", link2_position)
    
    sphere_collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.05)
    sphere_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0,0,1,1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_collision_shape, baseVisualShapeIndex=sphere_visual_shape, basePosition=link2_position)
    
    num_steps = 1000
    axis_length = 0.2
    for t in range(num_steps):
        
        
        p.addUserDebugLine(link1_position, link1_position + T_link1[:3, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link1_position, link1_position + T_link1[:3, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link1_position, link1_position + T_link1[:3, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0)  
        
        p.addUserDebugLine(link2_position, link2_position + T_link2[:3, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link2_position, link2_position + T_link2[:3, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
        p.addUserDebugLine(link2_position, link2_position + T_link2[:3, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0) 
        p.stepSimulation()
        time.sleep(1)

    input("Press enter to exit")
    p.disconnect()