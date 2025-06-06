from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.llms import Ollama
from typing import List, Dict
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 挂载静态文件夹（JS、CSS）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置模板
templates = Jinja2Templates(directory="templates")

# 初始化Ollama模型
llm = Ollama(model="qwen:7b-chat")

class Query(BaseModel):
    question: str

async def generate_response(question: str):
    """异步生成响应"""
    try:
        # 使用Ollama模型生成回答
        response = llm.stream(question)
        for chunk in response:
            if chunk:
                # 将每个响应块格式化为SSE格式
                yield f"data: {json.dumps({'content': chunk})}\n\n"
                await asyncio.sleep(0.1)  # 添加小延迟使流更平滑
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """返回主页"""
    return templates.TemplateResponse("index.html", {"request": request})

# 新增一个路由访问 index2.html
@app.get("/page2", response_class=HTMLResponse)
async def get_page2(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})


@app.post("/chat")
async def chat(query: Query):
    """处理聊天请求并返回SSE流"""
    return StreamingResponse(
        generate_response(query.question),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)