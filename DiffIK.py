import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

def compute_differential_ik(robot_id, joint_indices, ee_link_index, desired_ee_velocity):
    joint_states = p.getJointStates(robot_id, joint_indices)
    q = [s[0] for s in joint_states]
    dq_zero = [0.0] * len(joint_indices)
    
    local_pos = [0, 0 , 0]
    
    jac_t, _=p.calculateJacobian(robot_id, ee_link_index, local_pos, q, dq_zero, dq_zero)
    J = np.array(jac_t)
    
    J_pinv = np.linalg.pinv(J)
    
    dq = J_pinv @ desired_ee_velocity
    
    return dq

if __name__ == '__main__':
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    
    plane_id = p.loadURDF("plane.urdf")
    kuka_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
    
    joint_indices = [i for i in range(7)]
    ee_link_index = 6 
    
    center_x = 0.5
    center_y = 0.0
    
    radius = 0.1
    
    dq_log = []
    desired_velocity_log = []
    actual_velocity_log = []
    time_log = []
    prev_ee_pos = None
    
    for step in range(1000):
        t = step * 0.01
        
        vx = -radius * np.sin(t)
        vy = radius * np.cos(t)
        vz = 0.0
        desired_velocity = np.array([vx, vy, vz])
        
        dq = compute_differential_ik(kuka_id, joint_indices, ee_link_index, desired_velocity)
        dq_log.append(dq.copy())
        desired_velocity_log.append(desired_velocity.copy())
        time_log.append(t)
        
        joint_states = p.getJointStates(kuka_id, joint_indices)
        q = [s[0] for s in joint_states]
        dq_zero = [0.0] * len(joint_indices)
        
        jac_t, _=p.calculateJacobian(kuka_id, ee_link_index, [0,0,0], q, dq_zero, dq_zero)
        J = np.array(jac_t)
        ee_velocity_actual = J @ dq
        actual_velocity_log.append(ee_velocity_actual)
        
        state = p.getLinkState(kuka_id, ee_link_index, computeForwardKinematics=True)
        ee_pos = state[4]
        
        if prev_ee_pos is not None:
            p.addUserDebugLine(prev_ee_pos, ee_pos, lineColorRGB=[1,0,0], lineWidth=2, lifeTime=0)
        
        prev_ee_pos= ee_pos
        
        for i, vel in zip (joint_indices, dq):
            p.setJointMotorControl2(kuka_id, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=vel)
        
        p.stepSimulation()
        time.sleep(0.01)
    p.disconnect()
    
    desired_velocity_log = np.array(desired_velocity_log)
    actual_velocity_log = np.array(actual_velocity_log)
    
    plt.figure(figsize=(10, 6))
    labels = ['vx', 'vy', 'vz']
    for i in range(3):
        plt.plot(time_log, desired_velocity_log[:, i], '--', label=f'Desired{labels[i]}')
        plt.plot(time_log, actual_velocity_log[:, i], '--', label=f'Actual{labels[i]}')
        
    plt.xlabel('Time (s)')
    plt.ylabel('End-Effector velocity (m/s)')
    plt.title('Desired vs Actual End-Effector Velocities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    