import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"

from fastapi import FastAPI, HTTPException, Request
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from typing import List, Dict
import os
import traceback
import requests
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging
import asyncio
from pydantic import BaseModel


app = FastAPI(title="企业知识库AI服务111")

# 设置 CORS（虽然对SSE不完全生效，但保留作为兜底）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化Ollama模型和嵌入
llm = Ollama(model="qwen:7b-chat")
embeddings = OllamaEmbeddings(model="qwen:7b-chat")

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/knowledge/stream-query3")
async def stream_query_knowledge_base3(request: QuestionRequest):
    async def event_stream():
        # 1. 检索相关文档
        vectordb = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
        retriever = vectordb.as_retriever()
        docs = retriever.get_relevant_documents(request.question)
        context = "\n".join([doc.page_content for doc in docs])
        # 2. 构造带上下文的prompt
        prompt = f"已知信息如下：\n{context}\n请根据上述内容回答：{request.question}"
        # 3. 调用Ollama流式接口
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen:7b-chat",
            "prompt": prompt,
            "stream": True
        }
        with requests.post(url, json=payload, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        import json
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            yield f"data: {data['response']}\n\n"
                    except Exception:
                        continue

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    return StreamingResponse(event_stream(), media_type="text/event-stream")



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
    vectordb.persist()
    return vectordb

# 初始化知识库（首次运行或更新文档时调用）
@app.post("/api/knowledge/init")
async def initialize_knowledge_base(docs_folder: str):
    try:
        vectordb = init_vector_db(docs_folder)
        return {"status": "success", "message": "知识库初始化完成"}
    except Exception as e:
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

# 问答接口
@app.post("/api/knowledge/query")
async def query_knowledge_base(question: str):
    try:
        vectordb = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
        retriever = vectordb.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)