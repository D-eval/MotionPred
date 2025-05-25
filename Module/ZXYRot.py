# 求向量u到目标向量v的旋转矩阵
import numpy as np

def get_first_rot_angle(u,v,first_rot_dim1_st_dim2_equ):
    c = first_rot_dim1_st_dim2_equ[0]
    a = (c + 1) % 3
    b = (c + 2) % 3

    ab = [a,b]
    u_ab = u[ab]
    v_ab = v[ab]
    
    w_ab = np.zeros(2)

    equ_dim = first_rot_dim1_st_dim2_equ[1]
    another_dim = 3 - c - equ_dim

    equ_dim_local = 0 if equ_dim == a else 1
    another_dim_local = 1 - equ_dim_local
        
    bias = v[equ_dim]
    u_ab_norm = np.linalg.norm(u_ab)
    
    w_ab[equ_dim_local] = v_ab[equ_dim_local]
    w_ab[another_dim_local] = np.sqrt(u_ab_norm**2 - bias**2) # 另一个维度的取值统一取正

    w_ab_normed = w_ab / np.linalg.norm(w_ab)
    u_ab_normed = u_ab / u_ab_norm

    u_axis = np.array([[u_ab_normed[0], u_ab_normed[1]],[-u_ab_normed[1], u_ab_normed[0]]])
    w_in_u_axis = u_axis @ w_ab_normed
    return w_in_u_axis[0], w_in_u_axis[1]


def get_rot_mat(cosine_theta, sine_theta, rot_dim):
    a = (rot_dim + 1) % 3
    b = (rot_dim + 2) % 3
    rot_mat = np.zeros((3,3))
    rot_mat[rot_dim, rot_dim] = 1
    rot_mat[a, a] = cosine_theta
    rot_mat[b, a] = sine_theta
    rot_mat[a, b] = -sine_theta
    rot_mat[b, b] = cosine_theta
    return rot_mat



def get_second_rot_angle(w,v,second_rot_dim):
    rot_dim = second_rot_dim
    a = (rot_dim + 1) % 3
    b = (rot_dim + 2) % 3
    w_ab = w[[a,b]]
    v_ab = v[[a,b]]
    w_ab_normed = w_ab / np.linalg.norm(w_ab)
    v_ab_normed = v_ab / np.linalg.norm(v_ab)

    w_axis = np.array([[w_ab_normed[0], w_ab_normed[1]],[-w_ab_normed[1], w_ab_normed[0]]])
    v_in_w_axis = w_axis @ v_ab_normed
    return v_in_w_axis[0], v_in_w_axis[1]


def get_all_rot_mat(u,v):
    # 顺序 z,x,y
    u_xy_norm = np.linalg.norm(u[0:2])
    if u_xy_norm >= np.abs(v[0]):
        first_rot_dim1_st_dim2_equ = (2,0) # 旋转轴dim1使得dim2分量相等
        second_rot_dim = 0
    elif u_xy_norm >= np.abs(v[1]):
        first_rot_dim1_st_dim2_equ = (2,1)
        second_rot_dim = 1
    else:
        first_rot_dim1_st_dim2_equ = (0,1)
        second_rot_dim = 1

    cosine_theta, sine_theta = get_first_rot_angle(u,v,first_rot_dim1_st_dim2_equ)

    rot_mat_1 = get_rot_mat(cosine_theta, sine_theta, first_rot_dim1_st_dim2_equ[0])
    w = rot_mat_1 @ u

    cosine_phi, sine_phi = get_second_rot_angle(w,v,second_rot_dim)
    rot_mat_2 = get_rot_mat(cosine_phi, sine_phi, second_rot_dim)
    return rot_mat_1, rot_mat_2


def get_zxy_angle(u,v):
    # 顺序 z,x,y
    u_xy_norm = np.linalg.norm(u[0:2])
    if u_xy_norm >= np.abs(v[0]):
        first_rot_dim1_st_dim2_equ = (2,0) # 旋转轴dim1使得dim2分量相等
        second_rot_dim = 0
    elif u_xy_norm >= np.abs(v[1]):
        first_rot_dim1_st_dim2_equ = (2,1)
        second_rot_dim = 1
    else:
        first_rot_dim1_st_dim2_equ = (0,1)
        second_rot_dim = 1
    cosine_theta, sine_theta = get_first_rot_angle(u,v,first_rot_dim1_st_dim2_equ)
    rot_mat_1 = get_rot_mat(cosine_theta, sine_theta, first_rot_dim1_st_dim2_equ[0])
    w = rot_mat_1 @ u
    cosine_phi, sine_phi = get_second_rot_angle(w,v,second_rot_dim)
    theta = np.arctan2(sine_theta, cosine_theta)
    phi = np.arctan2(sine_phi, cosine_phi)
    result = np.zeros(3)
    result[first_rot_dim1_st_dim2_equ[0]] = np.rad2deg(theta)
    result[second_rot_dim] = np.rad2deg(phi)
    return result



def valid_input(u,v):
    rot_mat_1, rot_mat_2 = get_all_rot_mat(u,v)
    v_pred = rot_mat_2 @ rot_mat_1 @ u
    print("目标:",v,"预测:",v_pred)
    # return rot_mat_1, rot_mat_2

'''
for i in range(10):
    u = np.random.randn(3)
    v = np.random.randn(3)
    l = np.random.rand(1)
    u = u / np.linalg.norm(u) * l[0] * 10
    v = v / np.linalg.norm(v) * l[0] * 10
    valid_input(u,v)

u = np.array([1, 2, 3])
v = np.array([3, 1, 2])

u = np.array([5, 5, 1])
v = np.array([1, 1, 49**0.5])
valid_input(u,v)
'''
