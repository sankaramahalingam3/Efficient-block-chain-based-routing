import math
import numpy as np
from Global_Vars import Global_Vars
from check_obj import check_obj


def objfun(soln=None):

    n = Global_Vars.n
    S = Global_Vars.S_in
    Source = Global_Vars.Source
    Dest = Global_Vars.Dest
    Packet_loss = Global_Vars.Packet_loss
    Path = np.unique(soln)
    short_path = np.array([Source, Path, Dest])
    rmax = Global_Vars.rmax
    for rm in range(rmax):
        r = rm
        v = rm + 1
        # Operation for epoch
        G = []
        cl = []
        p = Global_Vars.p
        if (np.mod(r, np.round(1 / p)) == 0):
            for i in range(n + 1):
                g = 0
                G.append(g)
                c = 0
                cl.append(c)

            S.S_in_G = G
            S.cl = cl
    y2 = [S.S_in_xd, S.S_in_yd, S.S_in_G, S.S_in_type, S.S_in_E, S.S_in_ENERGY, S.cl]
    xd = np.reshape(y2[0], (-1))
    yd = np.reshape(y2[1], (-1))
    G = np.reshape(y2[2], (-1))
    type_ = np.reshape(y2[3], (-1))
    E = np.reshape(y2[4], (-1))
    Energy = np.reshape(y2[5], (-1))
    cl = np.reshape(y2[6], (-1))
    distance_Ch_node = np.zeros((len(y2[2])-1, 0))#CH.shape[1 - 1]))
    loca_data = np.asarray([xd, yd])
    cluster_center = np.asarray([loca_data[0][0], loca_data[1][0]])
    ## Distance Constraints
    # Finding distance matrix of inter Cluster
    for i in range(loca_data[0].shape[0]-1):
        for j in range(0):
            distance_Ch_node[i, j] = math.dist(cluster_center[:, j], np.transpose(loca_data[:, i]))
    a = np.asarray(cluster_center)
    b = np.asarray(cluster_center)
    Idist = np.zeros((len(a)))
    distn = np.zeros((len(b)))
    for l in range(a.shape[1 - 1]):
        for m in range(b.shape[1 - 1]):
            distn[m] = math.sqrt((a[l])**2 + (b[m])**2)
        Idist[l] = sum(distn)
    Idistannce = sum(Idist) / 10
    count = np.zeros((soln.shape[1 - 1]))
    # Finding Residual-energy
    E_mat = E
    CH_E = E_mat[:len(soln)]
    E_CH_min_value, E_CH_index_val = np.min(CH_E), np.amin(CH_E)
    E_node_min_value, E_node_index_val = np.min(E_mat), np.amin(E_mat)
    phi = 20.72
    Penalty = 0.5
    tou_value = - phi * (E_CH_min_value / (np.abs(E_node_min_value - E_CH_min_value) + 1e-10))
    f_energy_b = 1 - np.exp(tou_value)
    f_energy = np.abs(f_energy_b / np.exp(sum(count)))
    # Path Loss
    A = 10
    B = 25
    Ndf = 25
    path_loss = A * np.log10(Idist) + B * np.log10(Idist) + Ndf
    Path_loss = abs(path_loss)
    # Packet Delivery Ratio
    Total_Packet = 1000
    PKT_Loss_Ratio = sum(Packet_loss(short_path)) / Total_Packet
    Packet_Delivery_Ratio = (Total_Packet - PKT_Loss_Ratio) / Total_Packet
    # Delay
    Delay = 1 / sum(Packet_loss(short_path))
    # Throughput
    Throughput = Packet_Delivery_Ratio * 100
    out = (1/Throughput) + f_energy + Delay + Path_loss + Penalty

    return out

