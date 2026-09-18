# 硬件流程的无硬件测试

这里测试 `algorithm_Demo/` 中可脱离设备验证的逻辑，包括 Layer-4 解码、输出过滤、
录制配置、击球时机和回放窗口，以及 `layer4_weights.py` 的 Layer-4 weights
（下标 0 与旧手写权重逐元素一致、每套 weights 的 padding/几何契约）。测试通过
模拟 `samna` 对象运行，不会连接真实设备。
