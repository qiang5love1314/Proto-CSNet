import numpy as np
import matplotlib.pyplot as plt

stepLength = ['Siamese\nCNN', 'Siamese\nResNet', 'Proto\nCNN', 'Proto\nResNet', 'Proto ACmix\n+CNN','Proto\nACmix+ResNet'
,'Proto\nCSNet']

test_acc=[0.8143,0.8095,0.8397,0.9345,0.8576,0.8285,0.9670]
# test_acc=[0.7476,0.7143,0.6588,0.8730,0.7364,0.6555,0.8433]

plt.figure(figsize=(8.5, 5))  # 增加图形宽度
# plt.plot(stepLength, test_acc, color='royalblue', marker='p', label='test_acc')
plt.plot(stepLength, test_acc, color='royalblue', marker='p')

# 显示每个点的值
for i, txt in enumerate(test_acc):
    plt.text(i, txt, f'{txt:.4f}', ha='center', va='bottom',fontsize=11)  # 使用索引i作为x位置，txt作为y位置和要显示的文本

# plt.xlabel('Iteration Numbers', size=15)
plt.ylabel('Testing accuracy', size=14)
plt.grid(color="grey", linestyle=':', linewidth=0.5)
plt.tick_params(labelsize=12)
# plt.legend(loc='lower right')

# plt.yticks(np.arange(1, 2 + 1, 0.5))
# plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/TimeCompare_meet.pdf', bbox_inches='tight')

# 保存图形为SVG格式
# plt.savefig('plot_bed.pdf', bbox_inches='tight')
# plt.savefig('plot_meet.pdf', bbox_inches='tight')
# plt.savefig('plot_meet.svg', format='svg', dpi=1200)
plt.tight_layout()
plt.show()
