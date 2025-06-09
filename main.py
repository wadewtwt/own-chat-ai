from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.llms import Ollama
from fastapi.templating import Jinja2Templates
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from typing import List, Dict
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import traceback

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

# 初始化Ollama模型和嵌入
llm = Ollama(model="qwen:7b-chat")
embeddings = OllamaEmbeddings(model="qwen:7b-chat")


# 初始化知识库（首次运行或更新文档时调用）
@app.get("/knowledgeInit")
async def initialize_knowledge_base(docs_folder: str):
    try:
        vectordb = init_vector_db(docs_folder)
        return {"status": "success", "message": "知识库初始化完成"}
    except Exception as e:
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

# 初始化向量数据库
def init_vector_db(docs_folder: str):
    if not os.path.exists(docs_folder):
        raise FileNotFoundError(f"指定的文件夹不存在: {docs_folder}")
    # 检查chroma_db文件夹是否存在，如果不存在则创建
    chroma_db_path = "./chroma_db"
    if not os.path.exists(chroma_db_path):
        os.makedirs(chroma_db_path)
    loader = DirectoryLoader(
        docs_folder,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=chroma_db_path
    )
    # Chroma 新版不再需要 persist 方法，直接返回 vectordb
    return vectordb

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

@app.post("/chatWithKnowledge")
async def chat_with_knowledge(query: Query):
    """处理带知识库的聊天请求并返回SSE流"""
    try:
        # 加载已存在的向量数据库（不重新初始化）
        vectordb = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
        retriever = vectordb.as_retriever()
        docs = retriever.get_relevant_documents(query.question)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"已知信息如下：\n{context}\n请根据上述内容回答：{query.question}"
        def stream_response():
            response = llm.stream(prompt)
            for chunk in response:
                if chunk:
                    yield f"data: {{\"content\": {json.dumps(chunk)} }}\n\n"
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)