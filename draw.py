import matplotlib.pyplot as plt

# 从 txt 文件中加载数据
def load_results_from_txt(filename):
    iterations = []
    accuracies = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.split()
            iterations.append(int(data[0]))
            accuracies.append(float(data[1]))
    return iterations, accuracies

# 加载算法1的数据
iterations_algorithm1, accuracies_algorithm1 = load_results_from_txt('log/fedavg_mnist_cnn_50_C0.1_acc.txt')

# 加载算法2的数据
iterations_algorithm2, accuracies_algorithm2 = load_results_from_txt('log/inner_agg_mnist_cnn_50_C0.1_acc.txt')

# 绘制准确率图像
plt.plot(iterations_algorithm1, accuracies_algorithm1, label='fedavg')
plt.plot(iterations_algorithm2, accuracies_algorithm2, label='inner product aggregation')

# 添加标题和标签
plt.title('Accuracy Comparison in CIFAR-10')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()

# 显示图像
plt.show()
