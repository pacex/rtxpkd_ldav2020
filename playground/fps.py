import matplotlib.pyplot as plt
import numpy as np

nc = []
ack0 = []
ack1 = []
acinterp = []

filegroups = [["fps_logs/corona2_nc.txt", "fps_logs/corona2_ac_k0.txt", "fps_logs/corona2_ac_k1.txt", "fps_logs/corona2_ac_k0_interp.txt", "COVID-19"],
              ["fps_logs/laser4_nc.txt", "fps_logs/laser4_ac_k0.txt", "fps_logs/laser4_ac_k1.txt", "fps_logs/laser4_ac_k0_interp.txt", "Laser Ablation"]]

for filegroup in filegroups:

    nc.clear()
    ack0.clear()
    ack1.clear()
    acinterp.clear()

    with open(filegroup[0], "r") as f:
        for line in f.readlines():
            nc.append(float(line))

    with open(filegroup[1], "r") as f:
        for line in f.readlines():
            ack0.append(float(line))

    with open(filegroup[2], "r") as f:
        for line in f.readlines():
            ack1.append(float(line))

    with open(filegroup[3], "r") as f:
        for line in f.readlines():
            acinterp.append(float(line))

    minLength = min(len(nc), len(ack0), len(ack1), len(acinterp), 150)
    nc = nc[:minLength]
    ack0 = ack0[:minLength]
    ack1 = ack1[:minLength]
    acinterp = acinterp[:minLength]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(nc, '', label="no culling")
    ax.plot(ack0, '', label="culling, 1x1, NI")
    ax.plot(ack1, '', label="culling, 3x3, NI")
    ax.plot(acinterp, '', label="culling, 1x1, LI")
    plt.xlabel("Frame #", fontsize=18, weight='bold')
    plt.ylabel("FPS", fontsize=18, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    if (filegroup[4] == "COVID-19"):
        ax.set_yticks(np.arange(0, 12, 1))
        ax.set_yticks(np.arange(0, 12, 0.5), minor=True)
        ax.set_xticks(np.arange(0, 160, 20))
        ax.set_xticks(np.arange(0, 160, 10), minor=True)
    else:
        ax.set_yticks(np.arange(0, 34, 4))
        ax.set_yticks(np.arange(0, 34, 2), minor=True)
        ax.set_xticks(np.arange(0, 160, 20))
        ax.set_xticks(np.arange(0, 160, 10), minor=True)
    plt.title("Accumulation Performance: " + filegroup[4], fontsize=16, weight='bold')
    plt.grid()
    plt.legend(fontsize=18)
    plt.show(block=(filegroup[4]=="Laser Ablation"))