import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pose):
    """
    Converte uma pose UR [x, y, z, rx, ry, rz]
    para uma matriz de transformação homogênea 4x4.
    """
    x, y, z, rx, ry, rz = pose
    
    # O UR usa vetores de rotação (Rodrigues)
    rot_vector = np.array([rx, ry, rz])
    rotation_matrix = R.from_rotvec(rot_vector).as_matrix()
    
    # Monta a matriz 4x4
    T = np.identity(4)
    T[0:3, 0:3] = rotation_matrix
    T[0:3, 3] = [x, y, z]
    return T

def matrix_to_pose(T):
    """
    Converte uma matriz de transformação homogênea 4x4 
    de volta para o formato de pose UR [x, y, z, rx, ry, rz].
    """
    position = T[0:3, 3]
    rotation_matrix = T[0:3, 0:3]
    rot_vector = R.from_matrix(rotation_matrix).as_rotvec()
    return np.hstack((position, rot_vector))

def get_transformation_between_frames(pose_f1_base, pose_f2_base):
    """
    Calcula a matriz T_F1_F2 que transforma pontos do Frame 1 para o Frame 2.
    """
    T_B_F1 = pose_to_matrix(pose_f1_base)
    T_B_F2 = pose_to_matrix(pose_f2_base)
    
    # Inversa de T_B_F1 (equivalente à transformação de F1 para a Base)
    T_F1_B = np.linalg.inv(T_B_F1)
    
    # T_F1_F2 = T_F1_B * T_B_F2
    T_F1_F2 = np.dot(T_F1_B, T_B_F2)
    return T_F1_F2

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # Observação: Se os valores copiados do Polyscope estiverem em mm, divida x, y, z por 1000.
    # Exemplo convertendo mm -> metros:
    pose_plano_1 = [-0.08680, -0.31972, 0.12380, 0.138, -3.072, 0.004] # em metros e radianos
    pose_plano_2 = [0.17183, 0.00336, -0.06058, 0.108, -0.004, -3.230]  # em metros e radianos
    
    # 1. Calcula a matriz de transformação 4x4 de Plano1 para Plano2
    T_F1_F2 = get_transformation_between_frames(pose_plano_1, pose_plano_2)
    
    # 2. Converte o resultado para o formato de pose do UR
    relative_pose = matrix_to_pose(T_F1_F2)
    
    print("\n--- Matriz de Transformação Homogênea 4x4 (T_F1 -> F2) ---")
    print(np.round(T_F1_F2, 4))
    
    print("\n--- Pose Relativa [x, y, z (m), rx, ry, rz (rad)] ---")
    print(np.round(relative_pose, 4))