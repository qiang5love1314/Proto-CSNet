import numpy as np
import matplotlib.pyplot as plt

# 方法名称
methods = ['Siamese\nCNN', 'Siamese\nResNet', 'Proto\nCNN', 'Proto\nResNet', 'Proto ACmix\n+CNN','Proto ACmix\n+ResNet'
,'Proto\nCSNet']

#bedroom
# # 训练时间（秒）
# train_times = [185.84, 121.99, 133.34, 179.61, 147.43, 313.16, 41.81]
# # 测试时间（秒）
# test_times = [0.64, 0.48, 7.02, 11.73, 8.30, 12.48, 2.24]

# #meeting room
# 训练时间（秒）
train_times = [184.77, 122.03, 150.67, 221.13, 185.11, 314.08, 42.08]
# 测试时间（秒）
test_times = [0.58, 0.48, 6.82, 12.04, 8.60, 12.43, 2.30]



# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(8.5, 5))

# 绘制训练时间的折线图
ax1.plot(methods, train_times, color='royalblue',marker='p', label='Training time (s)')
ax1.set_ylabel('Training time (s)', size=14)
# ax1.tick_params(axis='y', labelcolor='royalblue')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=14)  # 调整x轴标签字体大小

# 在训练时间折线图上添加数值标签
for i, txt in enumerate(train_times):
    ax1.text(i, txt, f'{txt:.2f}', ha='center', va='bottom', color='royalblue', fontsize=10)  # 调整数值标签字体大小

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制测试时间的折线图
ax2.plot(methods, test_times, color='#ff7f0e', marker='p', label='Testing time (s)')  # 使用更深的灰色
ax2.set_ylabel('Testing time (s)', size=12)
# ax2.tick_params(axis='y', labelcolor='#ff7f0e')
ax2.set_yticks(np.arange(0, max(test_times) + 2, 2))  # 调整y轴刻度

# 在测试时间折线图上添加数值标签
for i, txt in enumerate(test_times):
    ax2.text(i, txt, f'{txt:.2f}', ha='center', va='bottom', color='#ff7f0e', fontsize=10)  # 调整数值标签字体大小

# 添加图例
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=9)  # 调整图例字体大小
# 添加网格
ax1.grid(color="grey", linestyle=':', linewidth=0.5)

plt.tight_layout()
# # 显示图形
# plt.title('Training and Testing Time Comparison', fontsize=16)  # 调整标题字体大小
plt.show()