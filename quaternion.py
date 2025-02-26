import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quat_to_rot(q):
    eta, x, y, z = q
    return np.array([[2*(eta**2 + x**2) -1, 2*(x*y - eta*z), 2*(x*z + eta*y)],
                     [2*(x*y + eta*z), 2*(eta**2 + y**2) - 1, 2*(x*y - eta*x)],
                     [2*(x*z - eta*y), 2*(y*z + eta*x), 2*(eta**2 + z**2) - 1]])

def plot_frame(ax, R, o, label, length=0.5):
    # end point for the xyz axes 
    x_axis = o + R.dot(np.array([length, 0 , 0]))
    y_axis = o + R.dot(np.array([0, length , 0]))
    z_axis = o + R.dot(np.array([0, 0 , length]))
    
    ax.quiver(o[0], o[1], o[2], x_axis[0]-o[0], x_axis[1] - o[1], x_axis[2] - o[2], color= 'r', arrow_length_ratio =0.1)
    ax.quiver(o[0], o[1], o[2], y_axis[0]-o[0], y_axis[1] - o[1], y_axis[2] - o[2], color= 'g', arrow_length_ratio =0.1)
    ax.quiver(o[0], o[1], o[2], z_axis[0]-o[0], z_axis[1] - o[1], z_axis[2] - o[2], color= 'b', arrow_length_ratio =0.1)
    
    ax.text(o[0], o[1], o[2], label, fontsize=12, color='k')
    
def Rz(alpha):
    c=np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c,-s, 0], 
                     [s, c, 0], 
                     [0, 0, 1]])

def Ry(beta):
    c=np.cos(beta)
    s = np.sin(beta)
    return np.array([[c,0, s], 
                     [0, 1, 0], 
                     [-s, 0, c]])

def Rx(gamma):
    c=np.cos(gamma)
    s = np.sin(gamma)
    return np.array([[1,0, 0], 
                     [0, c, -s], 
                     [0, s, c]])

def skew(r):
    rx, ry, rz = r
    return np.array([[0, -rz, ry],
                     [rz, 0, -rx],
                     [-ry, rx, 0]])
    
def axis_angle_rot(theta, r):
    r = np.array(r) / np.linalg.norm(r)
    K = skew(r)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K.dot(K))

def rpy_to_rot(roll, pitch, yaw):
    return Rz(roll).dot(Ry(pitch)).dot(Rx(yaw))

    
if __name__=="__main__":
    angle = np.deg2rad(45)
    q_example = [np.cos(angle/2), np.sin(angle/2), 0, 0]
    R_quat= quat_to_rot(q_example)
    print(R_quat)
    
    fig = plt.figure()
    
    # subplot 1
    ax = fig.add_subplot(111, projection='3d')
    plot_frame(ax, np.eye(3),np.array([0, 0, 0]), "World Frame")
    plot_frame(ax,R_quat, np.array([0, 0, 0]), "Quaternion frame")
  
    ax.set_title("Quaternion rotation")
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.legend()
 
    plt.show()
    