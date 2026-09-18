# 测试

测试分为两个目录：

- `offline/`：圆检测、解码、播放器和置信度测试；
- `hardware/`：硬件 Demo 的配置、过滤和回放逻辑测试，使用模拟对象，不要求连接 Speck2f。

从项目根目录运行全部测试：

```bash
python -m unittest discover -s tests -t . -p "test_*.py" -v
```
