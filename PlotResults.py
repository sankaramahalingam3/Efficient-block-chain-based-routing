import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def Plot_Results():
    Energy = np.load('Enery_Consumption.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, Energy[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="HHO")
    plt.plot(Nodes, Energy[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="BCO")
    plt.plot(Nodes, Energy[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="MAO")
    plt.plot(Nodes, Energy[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="SFO")
    plt.plot(Nodes, Energy[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Energy Consumption(J)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Energy-Consumption-alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, Energy[5, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="TB-BP")
    plt.plot(Nodes, Energy[6, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="QL-BP")
    plt.plot(Nodes, Energy[7, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="RLBC")
    plt.plot(Nodes, Energy[8, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, Energy[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Energy Consumption(J)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Energy-Consumption-method.png"
    plt.savefig(path1)
    plt.show()

    Path_loss = np.load('Path_loss.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, Path_loss[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="HHO")
    plt.plot(Nodes, Path_loss[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="BCO")
    plt.plot(Nodes, Path_loss[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="MAO")
    plt.plot(Nodes, Path_loss[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="SFO")
    plt.plot(Nodes, Path_loss[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Path Loss(dB)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Path-Loss-Alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, Path_loss[5, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="TB-BP")
    plt.plot(Nodes, Path_loss[6, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="QL-BP")
    plt.plot(Nodes, Path_loss[7, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="RLBC")
    plt.plot(Nodes, Path_loss[8, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, Path_loss[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Path Loss(dB)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Path-Loss-Method.png"
    plt.savefig(path1)
    plt.show()

    Delay = np.load('Delay.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, Delay[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="HHO")
    plt.plot(Nodes, Delay[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="BCO")
    plt.plot(Nodes, Delay[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="MAO")
    plt.plot(Nodes, Delay[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="SFO")
    plt.plot(Nodes, Delay[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Delay(Sec)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Delay-Alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, Delay[5, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="TB-BP")
    plt.plot(Nodes, Delay[6, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="QL-BP")
    plt.plot(Nodes, Delay[7, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="RLBC")
    plt.plot(Nodes, Delay[8, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, Delay[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Delay(Sec)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Delay-Method.png"
    plt.savefig(path1)
    plt.show()

    Throughput = np.load('Throughput.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, Throughput[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4',
             markersize=14,
             label="HHO")
    plt.plot(Nodes, Throughput[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e',
             markersize=14,
             label="BCO")
    plt.plot(Nodes, Throughput[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c',
             markersize=14,
             label="MAO")
    plt.plot(Nodes, Throughput[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728',
             markersize=14,
             label="SFO")
    plt.plot(Nodes, Throughput[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Throughput(Kbps)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Throughput-Alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, Throughput[5, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4',
             markersize=14,
             label="TB-BP")
    plt.plot(Nodes, Throughput[6, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e',
             markersize=14,
             label="QL-BP")
    plt.plot(Nodes, Throughput[7, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c',
             markersize=14,
             label="RLBC")
    plt.plot(Nodes, Throughput[8, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728',
             markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, Throughput[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Throughput(Kbps)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Throughput-Method.png"
    plt.savefig(path1)
    plt.show()



    PDR = np.load('Packet_Delivery_Ratio.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, PDR[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="HHO")
    plt.plot(Nodes, PDR[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="BCO")
    plt.plot(Nodes, PDR[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="MAO")
    plt.plot(Nodes, PDR[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="SFO")
    plt.plot(Nodes, PDR[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Packet Delivery Ratio(%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Packet-Delivery-Ratio-Alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, PDR[5, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="TP-BP")
    plt.plot(Nodes, PDR[6, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="QL-BP")
    plt.plot(Nodes, PDR[7, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="RLBC")
    plt.plot(Nodes, PDR[8, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, PDR[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Packet Delivery Ratio(%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Packet-Delivery-Ratio-Method.png"
    plt.savefig(path1)
    plt.show()

    Distance = np.load('Distance.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, Distance[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="HHO")
    plt.plot(Nodes, Distance[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="BCO")
    plt.plot(Nodes, Distance[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="MAO")
    plt.plot(Nodes, Distance[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="SFO")
    plt.plot(Nodes, Distance[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Distance(m)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Distance-Alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, Distance[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="TP-BP")
    plt.plot(Nodes, Distance[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="QL-BP")
    plt.plot(Nodes, Distance[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="RLBC")
    plt.plot(Nodes, Distance[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, Distance[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Distance(m)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Distance-Method.png"
    plt.savefig(path1)
    plt.show()


    Trust = np.load('Trust.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, Trust[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="HHO")
    plt.plot(Nodes, Trust[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="BCO")
    plt.plot(Nodes, Trust[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="MAO")
    plt.plot(Nodes, Trust[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="SFO")
    plt.plot(Nodes, Trust[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Trust')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Trust-Alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, Trust[5, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4', markersize=14,
             label="TP-BP")
    plt.plot(Nodes, Trust[6, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e', markersize=14,
             label="QL-BP")
    plt.plot(Nodes, Trust[7, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c', markersize=14,
             label="RLBC")
    plt.plot(Nodes, Trust[8, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728', markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, Trust[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Trust')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/Trust-Method.png"
    plt.savefig(path1)
    plt.show()

    ResponseTime = np.load('ResponseTime.npy', allow_pickle=True)
    Nodes = [50, 100, 150, 200]

    X = np.arange(4)
    plt.plot(Nodes, ResponseTime[0, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4',
             markersize=14,
             label="HHO")
    plt.plot(Nodes, ResponseTime[1, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e',
             markersize=14,
             label="BCO")
    plt.plot(Nodes, ResponseTime[2, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c',
             markersize=14,
             label="MAO")
    plt.plot(Nodes, ResponseTime[3, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728',
             markersize=14,
             label="SFO")
    plt.plot(Nodes, ResponseTime[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Response Time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/ResponseTime-Alg.png"
    plt.savefig(path1)
    plt.show()

    X = np.arange(4)
    plt.plot(Nodes, ResponseTime[5, :], color='#1f77b4', linewidth=3, marker='h', markerfacecolor='#1f77b4',
             markersize=14,
             label="TP-BP")
    plt.plot(Nodes, ResponseTime[6, :], color='#ff7f0e', linewidth=3, marker='h', markerfacecolor='#ff7f0e',
             markersize=14,
             label="QL-BP")
    plt.plot(Nodes, ResponseTime[7, :], color='#2ca02c', linewidth=3, marker='h', markerfacecolor='#2ca02c',
             markersize=14,
             label="RLBC")
    plt.plot(Nodes, ResponseTime[8, :], color='#d62728', linewidth=3, marker='h', markerfacecolor='#d62728',
             markersize=14,
             label="PoA-DL")
    plt.plot(Nodes, ResponseTime[4, :], color='k', linewidth=3, marker='h', markerfacecolor='k', markersize=14,
             label="HMASO")
    labels = ['50', '100', '150', '200']
    plt.xticks(Nodes, labels)

    plt.xlabel('No of Nodes')
    plt.ylabel('Response Time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    path1 = "./Results/ResponseTime-Method.png"
    plt.savefig(path1)
    plt.show()



def plot_results_conv():
    # matplotlib.use('TkAgg')
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'HHO', 'BCO', 'SFO', 'MAO', 'MO-SFO']
    nodes = ['50', '100', '150', '200']
    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
            # a = 1
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Node -', nodes[i], '- Statistical Report',
              '--------------------------------------------------')

        print(Table)

        length = np.arange(100)
        Conv_Graph = Fitness[i]
        # Conv_Graph = np.reshape(BestFit[i], (8, 20))
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label='HHO')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label='BCO')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='cyan',
                 markersize=12,
                 label='SFO')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label='MAO')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='HMASO')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/No-of-Nodes-%s.png" % (nodes[i]))
        plt.show()


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


if __name__ == '__main__':
    plot_results_conv()
    Plot_Results()
