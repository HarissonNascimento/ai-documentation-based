import weaviate
import os
from ingest import get_embeddings_model
from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from prompts import (
    RESPONSE_TEMPLATE,
    REPHRASE_TEMPLATE
)
from constants import (
    WEAVIATE_URL,
    WEAVIATE_DOCS_INDEX_NAME
)

from pydantic import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap
)
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai import ChatOpenAI
from langchain.globals import set_verbose

set_verbose(True)


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


class ChatResponse(BaseModel):
    response: str
    chat_history: Optional[List[Dict[str, str]]]


def get_retriever() -> BaseRetriever:
    client = weaviate.Client(
        url=WEAVIATE_URL
    )
    weaviate_client = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))


def create_retriever_chain(
        llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
            CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
                RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                )
                | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}' source='{doc.metadata['source']}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
            {
                "question": RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                ),
                "chat_history": RunnableLambda(serialize_history).with_config(
                    run_name="SerializeHistory"
                ),
            }
            | _context
            | response_synthesizer
    )


chat_history = []

llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    streaming=True,
    temperature=0,
    verbose=True
)

retriever = get_retriever()

answer_chain = create_chain(
    llm,
    retriever,
)


def invoke_ai(chat_request: ChatRequest) -> ChatResponse:
    response = answer_chain.invoke({'question': chat_request.question, 'chat_history': chat_request.chat_history})
    chat_history.append({'human': chat_request.question, 'ai': response})
    return ChatResponse(response=response, chat_history=chat_history)


# def call_ai(input: str):
#    response = answer_chain.invoke({'question': input, 'chat_history': chat_history})
#    chat_history.append({'human': input, 'ai': response})
#    return response
