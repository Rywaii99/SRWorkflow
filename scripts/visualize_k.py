import numpy as np
import matplotlib.pyplot as plt

# 参数定义
img_size = 256  # 假设图像大小为256x256
k_list = [3, 5, 7]
nyquist = 0.5   # Nyquist频率 (cycles/pixel)

# 计算截止频率
cutoff_freqs = [nyquist / k for k in k_list]

# 生成频率轴
freq_axis = np.linspace(0, nyquist, 1000)

# 绘制响应曲线
plt.figure(figsize=(10, 4))
for k, cutoff in zip(k_list, cutoff_freqs):
    plt.plot(freq_axis, (freq_axis <= cutoff).astype(float),
             label=f'k={k}, fc={cutoff:.3f}')

plt.title("Ideal Low-Pass Filter Frequency Response")
plt.xlabel("Frequency (cycles/pixel)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()