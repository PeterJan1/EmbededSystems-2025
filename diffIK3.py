import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # Simulation setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setTimeStep(0.01)
    dt = 0.01

    # Load robot and plane
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    # KUKA joint setup
    joint_indices = list(range(7))
    ee_link_index = 6

    # Circle trajectory parameters
    radius = 0.15
    steps = 2000
    speed = 0.5#2 * np.pi / steps
    Kp = 20.0
    Ki = 5.0
    damping = 1

    # ------------------------------
    # 🔧 Initialise robot to match trajectory start
    initial_target = [0.5 + radius, 0.0, 0.5]
    ik_solution = p.calculateInverseKinematics(robot_id, ee_link_index, initial_target)

    for i, joint_index in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_index, ik_solution[i])
        p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=ik_solution[i])

    for _ in range(100):  # settle robot
        p.stepSimulation()
        time.sleep(0.01)

    # Circle centre is initial target
    center = np.array(initial_target)
    z_const = center[2]

    # Initial joint config
    q = np.array([p.getJointState(robot_id, i)[0] for i in joint_indices])

    # Logging
    ee_positions = []
    desired_positions = []
    integral_error = 0.0
    # ------------------------------
    # 🔁 Main control loop
    for step in range(steps):
        t = step * dt

        # Desired EE position and velocity on circle
        angle = speed * t
        # print(angle)
        xd = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
        xd[2] = z_const
        v_des = radius * speed * np.array([-np.sin(angle), np.cos(angle), 0])

        # Get actual EE position
        ee_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
        x_act = np.array(ee_state[4])
        pos_error = xd - x_act
        integral_error += pos_error * dt
        v_cmd = v_des + Kp * pos_error + Ki * integral_error

        # Compute Jacobian
        dq_zero = [0.0] * len(joint_indices)
        jac_t, _ = p.calculateJacobian(robot_id, ee_link_index, [0, 0, 0], q.tolist(), dq_zero, dq_zero)
        J = np.array(jac_t)

        # Damped pseudoinverse
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + damping**2 * np.eye(JJT.shape[0]))

        # Joint velocities and integration
        dq = J_pinv @ v_cmd
        dq=np.clip(dq,-5,5)
        q = q + dq * dt  # integrate manually

        # Send joint positions using POSITION_CONTROL
        for i, qi in zip(joint_indices, dq):
            p.setJointMotorControl2(robot_id, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=qi)
            

        # Log data
        ee_positions.append(x_act.copy())
        desired_positions.append(xd.copy())

        p.stepSimulation()
        time.sleep(dt)

    p.disconnect()

    ee_positions = np.array(ee_positions)
    desired_positions = np.array(desired_positions)

    plt.figure(figsize=(6, 6))
    plt.plot(ee_positions[:, 0], ee_positions[:, 1], label='Actual Path')
    plt.plot( desired_positions[:,0], desired_positions[:, 1], '--', label='Desired Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.title('End-Effector XY Trajectory')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

