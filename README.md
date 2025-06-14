# own-chat-ai

这是一个基于 Python 的聊天 AI 项目。
![alt text](image.png)
## 运行环境

- 本地运行环境为 Python 3.10

## 目录结构

- `main.py`：主程序入口
- `requirements.txt`：依赖包列表
- `static/`：静态资源文件夹
- `templates/`：模板文件夹
    - `index.html`：主页面模板
    - `index2.html`：备用页面模板

## 快速开始

1. 安装依赖：
   ```powershell
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
2. 运行项目：
   ```powershell
   python main.py
   ```

## 访问说明

- 访问主页面请请求：http://127.0.0.1:8000/
- 访问 index2.html 请请求：http://127.0.0.1:8000/page2
- 如需新增页面，请在 py 文件中添加对应路由。

## Windows 关闭本地已开启端口的方法

如果端口 8000 已被占用，可以通过以下方式关闭对应进程：

1. 打开命令行，输入：
   ```powershell
   netstat -aon | findstr :8000
   ```
   找到占用 8000 端口的 PID。
2. 打开任务管理器，找到对应 PID 的进程，结束该进程即可释放端口。

## 本地知识库相关接口说明

1. 初始化向量数据库请求：
   http://127.0.0.1:8000/knowledgeInit?docs_folder=origin_data
   - 也可通过网页右上角“更新数据库源”按钮进行初始化。
2. 聊天接口：
   - 普通对话请求地址：/chat
   - 使用本地向量数据库进行查询请求地址：/chatWithKnowledge

## 许可证

MIT License
