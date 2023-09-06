本项目基于经典DeepLabv3+网络模型进行改进。

（1）编码区，DeepLabv3+中采用Xception作为主干网络提取特征，但Xception所带来的参数量和训练速度的问题仍有待优化。本文采用一种基于深度可分离卷积的轻量级网络MobileNetv2作为特征提取网络，对其进行改进，在降低参数量，减少计算开销的同时提升模型的特征提取效率，使更适合语义分割任务，从而提取出浅层特征和深层特征。值得注意的是，经典DeepLabv3+网络模型中解码器部分只取一层低水平特征层，过于简单，本文从MobileNetv2网络中提取出第4层、第7层两条浅层特征，分别施加NAM注意力机制，以增强低层的语义信息。

深层特征在ASPP模块中得到特征增强，但空洞卷积是离散的采样，较大的膨胀率易忽略连续点之间的依赖关系，存在网格效应，易造成局部信息的丢失，影响预测结果。本文中使用HDC模块对空洞卷积进行替换，通过一系列的空洞卷积覆盖底层特征层的方形区域，且方形区域中间无孔洞或缺失的边缘，以此改善网格效应带来的问题。除此之外，本文采用带状池化模块替代原模型中使用的全局平均池化模块，避免在距离较远的位置建立不必要的连接，分别通过垂直池化和水平池化构建通道间的依赖关系，从不同的空间维度收集信息。对堆叠压缩后的高层特征图，也施加轻量高效的NAM注意力机制，帮助图像提高分割精度。

（2）解码区，将经过NAM注意力的第7层特征进行上采样，变成与第4层同样大小的特征层并进行融合，调整通道数后添加ResNet50模块，进一步获得丰富的低层目标特征信息。而后同于原始模型，将深层特征与浅层特征进行拼接。最后，进行一次3×3的卷积和4倍上采样后，将图像恢复到原始图像大小。

![image-20230906203342818](C:\Users\23672\AppData\Roaming\Typora\typora-user-images\image-20230906203342818.png)

​                               
