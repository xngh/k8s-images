# NDC深度学习任务平台v1

/root/jupyter_workspace/dataset：通过NFS外挂了lijlun3-1080Ti-Server(IP:172.18.232.123)这台服务器的/home/ndc-server/dataset目录，用于存放数据集。

### 使用：

- 基于基础镜像pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime。
- 在删除容器的前需要先下载代码到本地来保存。
- 创建任务时声明的GPU使用上限仅供调度器调度时使用，实际的使用显存可以大于这个值。（所以还是存在挤爆显存的可能。）
