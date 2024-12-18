from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     MODEL_PATH)
from configs import logger, log_verbose
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
import json
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_summary_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
from langchain.prompts import PromptTemplate

async def summary(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                  ge=0,
                                  le=2
                              ),
                              history: List[History] = Body(
                                  [],
                                  description="历史对话",
                                  examples=[[
                                      {"role": "user",
                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                      {"role": "assistant",
                                       "content": "虎头虎脑"}]]
                              ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM生成Token数量，默认None代表模型最大值"
                              ),
                              prompt_name: str = Body(
                                  "default",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def summary_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        firstcallback = AsyncIteratorCallbackHandler()
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        firstmodel = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[firstcallback],
        )
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        logger.info("query")
        logger.info(query)
        kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
        filenames = kb.list_files()
        logger.info(filenames)
        sumtext=""
        if len(filenames) > 0:
            topdoclist = await run_in_threadpool(search_docs,
                                       query="",
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=1,
                                       score_threshold=score_threshold,
                                       file_name=filenames[0])
            if len(topdoclist) > 0:
                logger.info("retrieving names...")
                for inum, doc in enumerate(topdoclist):
                    logger.info(doc.page_content)
                    sumpromptstr = "下面是一篇财报的第一段, 帮我提取出文章的名字和公司名字: {context}"
                    sumprompt= PromptTemplate(input_variables=["context"],template=sumpromptstr)
                    #logger.info(sumprompt.format(context = doc.page_content))
                    sumchain = LLMChain(prompt=sumprompt, llm=firstmodel)
                    sumtask = asyncio.create_task(wrap_done(
                        sumchain.acall({"context": doc.page_content}),
                        firstcallback.done),
                    )

                    async for token in firstcallback.aiter():
                        sumtext += token
                    await sumtask

                    logger.info("retrieved name:")
                    logger.info(sumtext.replace('\n', ' '))
                    break

        newquery = '本文是'+sumtext.replace('\n', ' ')+' '+query
        docs = await run_in_threadpool(search_docs,
                                       query=newquery,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k,
                                       score_threshold=score_threshold,
                                       file_name="")
        logger.info("num of searched docs")
        logger.info(len(docs))
        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL,"BAAI/bge-reranker-large")
            print("-----------------model path------------------")
            print(reranker_model_path)
            logger.info("model path")
            logger.info(reranker_model_path)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)
        context = "\n".join([doc.page_content for doc in docs])

        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template("summary", "empty")
        else:
            prompt_template = get_prompt_template("summary", prompt_name)
        logger.info("prompt")
        logger.info(prompt_name)
        logger.info(prompt_template)

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": newquery}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            logger.info("doc:")
            logger.info(base_url)
            logger.info("url:")
            logger.info(url)
            logger.info("出处")
            logger.info(text)
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            logger.info("dumps")
            logger.info(answer)
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(summary_iterator(query, top_k, history,model_name,prompt_name))


