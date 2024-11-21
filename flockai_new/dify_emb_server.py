from fastapi import FastAPI, Body
from typing import List
from pydantic import BaseModel
import sys
sys.path.append(r"/home/kuaipan/flockai_new/server")
from embeddings_api import embed_texts_endpoint,embed_texts
import uvicorn
import logging

EMBEDDING_MODEL = "bge-large-zh"
app = FastAPI()

# class BaseResponse(BaseModel):
#     result: List[float]

def response_processer(result:BaseModel,model):
    result = result.dict()
    response = {"usage": {"total_tokens":0},
        "error": {"code":result["code"],"message": result["msg"]},"data": []}
    if "acge_" in model:
        # 针对acge_text_embedding模型
        result["data"] = result["data"].tolist()
    if len(result["data"])>0:
        for emb_item in result["data"]:
            response["data"].append({"embedding": emb_item})
    return response

@app.post("/embeddings", tags=["Other"], summary="将文本向量化，支持本地模型和在线模型")
async def new_embed_texts_endpoint(
    input: str = Body(..., description="要嵌入的文本列表"),
    model: str = Body(..., description="使用的嵌入模型，除了本地部署的Embedding模型，也支持在线API提供的嵌入服务。")
):
    # embed_texts_endpoint函数调用embed_texts函数，在这里直接调用embed_texts
    # result = embed_texts_endpoint(texts, embed_model, to_query)
    if not isinstance(input, list):
        inputs = [input]
    result = embed_texts(inputs, model, to_query=False)
    response = response_processer(result,model)
    response["usage"]["total_tokens"] = len(inputs[0])
    logging.info(f"请求文本为:{input}")
    logging.info("函数调用成功，结果")
    return response

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8593)
