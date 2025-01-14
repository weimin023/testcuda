import seaborn as sns
import matplotlib.pyplot as plt   

sns.set_theme(style="whitegrid")
sns.set_theme(style="ticks")

x_sticks = ["16*16*16", "64*64*64", "128*128*128", "256*256*256", "512*512*512", "1024*1024*1024"]
data1 = [0.0124, 0.038, 0.07, 0.31, 2.40, 18.91]
data1 = [x * 1e3 for x in data1]

data2 = [0.0128, 0.0137, 0.02, 0.04, 0.25, 1.87]
data2 = [x * 1e3 for x in data2]

data3 = [0.0106, 0.0137, 0.02, 0.02, 0.1, 0.62]
data3 = [x * 1e3 for x in data3]

data4 = [0.0114, 0.011, 0.01, 0.02, 0.06, 0.32]
data4 = [x * 1e3 for x in data4]

sns.lineplot(x=x_sticks, y=data1, label="CUDA Core (Naive)", marker="o")
sns.lineplot(x=x_sticks, y=data2, label="CUDA Core (Shared Mem)", marker="s")
sns.lineplot(x=x_sticks, y=data3, label="CUDA Core (Shared Mem + Threads Reuse)", marker="^")
sns.lineplot(x=x_sticks, y=data4, label="Tensor Core (Naive)", marker="D")

plt.xlabel("Data Size (M*K*N)", fontsize=20)
plt.ylabel("Runtime (us)", fontsize=20)
plt.suptitle("Performance Comparison of Matrix Multiplication", fontsize=20, y=0.95)
plt.title("Using CUDA Cores and Tensor Cores", fontsize=15)

plt.xticks(rotation=45, fontsize=15)
plt.show()