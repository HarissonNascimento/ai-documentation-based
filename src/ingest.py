import weaviate
import os
import re
import logging
from typing import List
from bs4 import BeautifulSoup as Soup
from pydantic import BaseModel
from constants import (
    PREFIX_URL,
    WEAVIATE_DOCS_INDEX_NAME,
    WEAVIATE_URL,
    RECORD_MANAGER_DB_URL
)

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.weaviate import Weaviate
from langchain.indexes import index, SQLRecordManager
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

url = f'{PREFIX_URL}/docs/'


class IngestResponse(BaseModel):
    index_stats: str


def simple_extractor(html: str) -> str:
    soup = Soup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text.encode('latin1').decode('utf-8')).strip()


def load_api_docs(): 
    return RecursiveUrlLoader(
        url=url,
        max_depth=10,
        extractor=simple_extractor,
        #use_async=True,
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
).load()


def load_docs_info(docs: List[Document]):
    logger.info(f"Document example: {docs[0]}")
    sources = []
    for source in docs:
        directory = source.metadata['source'].replace(PREFIX_URL, "")
        sources.append(directory)

    logger.info(f'URLs found: {sources}')


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


def ingest_docs() -> IngestResponse:
    docs = load_api_docs()
    load_docs_info(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    client = weaviate.Client(
        url=WEAVIATE_URL
    )
    embedding = get_embeddings_model()
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    docs_from_documentation = text_splitter.split_documents(docs)

    for doc in docs_from_documentation:
        if "source" not in doc.metadata:
            doc.metadata['source'] = ""
        if "title" not in doc.metadata:
            doc.metadata['title'] = ""

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )

    record_manager.create_schema()
    
    indexing_stats = index(
        docs_from_documentation,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    return IngestResponse(index_stats=f"Index Stats: {indexing_stats}")