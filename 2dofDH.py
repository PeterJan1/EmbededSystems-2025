import pybullet as p
import pybullet_data
import numpy as np
import time
import os

def  dh_transformation(theta, a, d, alpha):
    R_y = np.array([
        [np.cos(theta),  0,  np.sin(theta), a*np.sin(theta)],
        [0,              1,  0,             0],
        [-np.sin(theta), 0, np.cos(theta), a*np.cos(theta)],
        [0,              0, 0,             1]
    ])
    return R_y

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    robot_id = p.loadURDF("urdf/2dof_planar_robot.urdf",basePosition=[0,0,0], useFixedBase=True)
    
    joint_angles = [np.pi/4, np.pi/6]
    
    dh_params = [
        [1, 0, 0],
        [1,0, 0]
    ] 
    
    num_dof = 2
    
    T=np.eye(4)
    T[2,3] = 0.05
    T_list = np.zeros((num_dof, 4, 4))
    
    for i in range(num_dof):
        T_i = dh_transformation(joint_angles[i],*dh_params[i])
        T = T@T_i
        T_list[i]=T
        
    fk_position_dh = T[:3,3]
    
    for i in range(num_dof):
        p.setJointMotorControl2(robot_id, i+1, p.POSITION_CONTROL, targetPosition=joint_angles[i])
        
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1/240)
    
    ee_state = p.getLinkState(robot_id, 3, computeForwardKinematics=True)
    fk_position_pybullet = np.array(ee_state[4])
    
    
    print(f"DH Computed : {fk_position_dh}")
    print(f"Pybullet computation: {fk_position_pybullet}")
    
    while True:
        p.stepSimulation()
        time.sleep(1/240)
         