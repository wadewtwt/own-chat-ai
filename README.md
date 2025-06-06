# own-chat-ai

这是一个基于 Python 的聊天 AI 项目。

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
   pip install -r requirements.txt
   ```
2. 运行项目：
   ```powershell
   python main.py
   ```

## 访问说明

- 访问主页面请请求：http://127.0.0.1:8000/
- 访问 index2.html 请请求：http://127.0.0.1:8000/page2
- 如需新增页面，请在 py 文件中添加对应路由。

## 许可证

MIT License
