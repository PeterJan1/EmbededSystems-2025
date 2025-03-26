import pybullet as p
import pybullet_data
import numpy as np
import time



def dh_transformation(theta, a, d, alpha):
    """Compute the DH transformation matrix."""
    return np.array([
        [np.cos(theta),                    -np.sin(theta),                          0,                      a],
        [np.sin(theta)*np.cos(alpha),      np.cos(theta) * np.cos(alpha),          -np.sin(alpha),         -d * np.sin(alpha)],
        [np.sin(theta)*np.sin(alpha),      np.cos(theta) * np.sin(alpha),           np.cos(alpha),          d*np.cos(alpha)],
        [0,                                 0,                                       0,                     1]
    ])


if __name__=="__main__":
    # Start PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load the Franka Emika Panda robot URDF
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    num_dof = 7
    # Define DH Parameters for Franka Emika Panda
    dh_params = [
        [ 0,        0.333,   0],  # Joint 1
        [ 0,        0,      -np.pi/2],        # Joint 2
        [ 0,        0.316,  np.pi/2],    # Joint 3
        [ 0.0825,   0,      np.pi/2],  # Joint 4
        [-0.0825,   0.384, -np.pi/2],  # Joint 5
        [ 0,        0,      np.pi/2],       # Joint 6
        [ 0.088,    0.107,  np.pi/2] # Joint 7
    ]

    # Set joint angles (random values within range)
    joint_angles = [0, -np.pi/4, np.pi/4, -np.pi/2, np.pi/2, np.pi/2, np.pi/3]

    # Compute DH-based Forward Kinematics
    T = np.eye(4)
    T_list = np.zeros((num_dof,4,4))
    for i in range(num_dof):
        T_i = dh_transformation(joint_angles[i], *dh_params[i])
        T = np.dot(T, T_i)
        T_list[i] = T   

    # Extract DH FK end-effector position
    fk_position_dh = T[:3, 3]

    # Apply the same joint angles in PyBullet
    for i in range(num_dof):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_angles[i])

    # Step simulation to update joint positions
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1/240)

    # Get end-effector position from PyBullet
    ee_state = p.getLinkState(robot_id, 8, computeForwardKinematics=True)
    fk_position_pybullet = np.array(ee_state[4])  # Extract world position


    # sphere_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 0, 1, 1])

    # for i in range(num_dof):
    #     p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_visual_shape, basePosition=T_list[i,:3,3],)

    # print(T_list[3,:3,3],)

    # # p.addUserDebugPoints([fk_position_dh], [[0,0,1]], pointSize=20)
    # p.addUserDebugPoints(T_list[:,:3,3], [[0,0,1]]*num_dof, pointSize=20)

    # Print results
    print(f"DH Computed End-Effector Position: {fk_position_dh}")
    print(f"PyBullet End-Effector Position: {fk_position_pybullet}")

    # Keep GUI open for visualization
    while True:
        p.stepSimulation()
        time.sleep(1/240)

