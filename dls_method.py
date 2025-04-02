import pybullet as p
import pybullet_data
import numpy as np
import time

# Damped Least Squares Inverse Kinematics for KUKA

def numerical_ik(robot_id, end_effector_index, target_pos, joint_indices, max_iters=100, threshold=1e-3, alpha=0.1, damping=0.1):
    """Numerical IK using Jacobian Transpose with Damped Least Squares"""
    for _ in range(max_iters):
        # Get current joint states
        joint_states = [p.getJointState(robot_id, i)[0] for i in joint_indices]

        # Compute current end-effector position
        link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
        current_pos = np.array(link_state[4])

        # Compute position error
        error = np.array(target_pos) - current_pos
        if np.linalg.norm(error) < threshold:
            break

        # Compute numerical Jacobian
        zero_vec = [0.0] * len(joint_indices)
        J_lin, _ = p.calculateJacobian(
            robot_id, end_effector_index,
            localPosition=[0, 0, 0],
            objPositions=joint_states,
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )
        J = np.array(J_lin)

        # Compute damped pseudo-inverse
        JT = J.T
        d_theta = alpha * JT @ np.linalg.inv(J @ JT + damping**2 * np.eye(3)) @ error

        # Update joint angles
        for i, idx in enumerate(joint_indices):
            p.resetJointState(robot_id, idx, joint_states[i] + d_theta[i])
            # time.sleep(1/10)
        p.stepSimulation()

if __name__ == "__main__":
    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    kuka_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
    sphere_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])

    end_effector_index = 6
    target_position = [0.7, 0.0, 0.5]
    movable_joints = [i for i in range(p.getNumJoints(kuka_id)) if p.getJointInfo(kuka_id, i)[2] != p.JOINT_FIXED]

    # Run numerical IK
    numerical_ik(kuka_id, end_effector_index, target_position, movable_joints)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_visual_shape, basePosition=target_position)

    # Get and print final end-effector position
    final_pos = p.getLinkState(kuka_id, end_effector_index)[4]
    print("Target:", np.round(target_position, 3))
    print("Achieved:", np.round(final_pos, 3))

    input("Press Enter to disconnect...")
    p.disconnect()
