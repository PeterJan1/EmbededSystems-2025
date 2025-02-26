import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    # theta = np.deg2rad(45)
    
    # r_axis = [1, 1, 0]
    # R_axis_angle = axis_angle_rot(theta, r_axis) 
    # print("Rotation Matrix (Rodrigues): \n", R_axis_angle )
    roll = np.deg2rad(30)
    pitch = np.deg2rad(45)
    yaw = np.deg2rad(60)
    R_rpy = rpy_to_rot(roll, pitch, yaw)
    print(R_rpy)
    
    fig = plt.figure()
    
    # subplot 1
    ax = fig.add_subplot(111, projection='3d')
    plot_frame(ax, np.eye(3),np.array([0, 0, 0]), "World Frame")
    plot_frame(ax,R_rpy, np.array([0, 0, 0]), "RPY frame")
  
    ax.set_title("RPY rotation")
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.legend()
 
    plt.show()
    