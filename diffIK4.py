import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

# Visualise desired path
def marker(pos, colour):
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=colour)
    p.createMultiBody(baseVisualShapeIndex=shape, basePosition=pos)


if __name__ == '__main__':
    # Setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load robot
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
    ee_index = 6
    num_joints = p.getNumJoints(robot_id)
    joint_indices = list(range(num_joints))

    # Trajectory
    start = np.array([0.5, 0.0, 0.6])
    end   = np.array([0.6, 0.2, 0.9])
    num_steps = 100
    positions = np.linspace(start, end, num_steps)
    orientation = p.getQuaternionFromEuler([0, 0, 0])  # fixed


    for pos in positions:
        marker(pos, [1, 0, 0, 1])  # red for desired

    # Pre-positioning loop: move to first pose and wait until it's accurate
    first_pose = positions[0]
    tol = 1e-3  # 1 mm position accuracy

    for _ in range(1000):  # max 1000 steps
        joint_angles = p.calculateInverseKinematics(robot_id, ee_index, first_pose, orientation)

        for i, j in enumerate(joint_indices):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=joint_angles[i], force=500)

        p.stepSimulation()
        time.sleep(1/240.0)

        # Check if close enough
        state = p.getLinkState(robot_id, ee_index)
        actual_pos = np.array(state[4])
        err = np.linalg.norm(actual_pos - first_pose)

        if err < tol:
            print(f"Robot reached starting position (err: {err:.4f} m)")
            break

    # Track error
    pos_errors = []
    ori_errors = []

    for pos in positions:
        joint_angles = p.calculateInverseKinematics(robot_id, ee_index, pos, orientation)

        for i, j in enumerate(joint_indices):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=joint_angles[i], force=500)

        p.stepSimulation()
        time.sleep(1/240)

        state = p.getLinkState(robot_id, ee_index)
        actual_pos = np.array(state[4])
        actual_ori = state[5]
        marker(actual_pos, [0, 0, 1, 1])  # blue for actual

        # Errors
        pos_error = np.linalg.norm(pos - actual_pos)
        dot = np.dot(orientation, actual_ori)
        dot = np.clip(dot, -1.0, 1.0)
        ori_error_deg = 2 * np.arccos(abs(dot)) * (180 / np.pi)

        pos_errors.append(pos_error)
        ori_errors.append(ori_error_deg)

    # Plot errors
    input("Press Enter to view error plots...")
    p.disconnect()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(pos_errors, label="Position Error (m)")
    plt.title("End-Effector Position Error")
    plt.grid(True)
    plt.xlabel("Step")
    plt.ylabel("m")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ori_errors, label="Orientation Error (°)", color='orange')
    plt.title("End-Effector Orientation Error")
    plt.grid(True)
    plt.xlabel("Step")
    plt.ylabel("Degrees")
    plt.legend()

    plt.tight_layout()
    plt.show()
