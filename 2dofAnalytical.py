import numpy as np
import matplotlib.pyplot as plt

l1 = 1.0
l2 = 1.0

def inverse_kinematics(x,y):
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    if np.abs(cos_theta2) > 1:
        raise ValueError("Target is out of reach")
    
    theta2 = np.arccos(cos_theta2)
    k1 = l1 + l2 * np.cos(theta2)
    k2 = l2 * np.sin(theta2)
    theta1 = np.arctan2(y,x) - np.arctan2(k2, k1)
    
    return theta1, theta2

def forward_kinematics(theta1, theta2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return (0, x1, x2), (0, y1, y2)

def plot_arm(x, y):
    try:
        theta1, theta2 = inverse_kinematics(x, y)
        x_coords, y_coords = forward_kinematics(theta1, theta2)
        
        plt.figure()
        plt.plot(x_coords, y_coords, '-o', linewidth=4)
        plt.plot(x, y, 'rx', label='Target')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.title("2 dof planar arm")
        plt.legend()
        plt.show()
    except ValueError as e:
        print(f"{e}: ({x:.2f}, {y:.2f})")
        
if __name__ =="__main__":
    plot_arm(1.5, 1.5)
    plot_arm(0.5, 1.5)
    plot_arm(2, 0)
    plot_arm(3, 1.5)
             
             
                
        
