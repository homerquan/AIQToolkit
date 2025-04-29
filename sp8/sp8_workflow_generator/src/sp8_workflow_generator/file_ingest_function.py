import logging
import os

from langchain_core.embeddings import Embeddings

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PDFFileIngestToolConfig(FunctionBaseConfig, name="pdf_file_ingest_tool"):
    ingest_glob: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5"


@register_function(config_type=PDFFileIngestToolConfig)
async def pdf_file_ingest_tool(config: PDFFileIngestToolConfig, builder: Builder):

    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings: Embeddings = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    logger.info("Ingesting PDF documents matching: %s", config.ingest_glob)
    (ingest_dir, ingest_glob) = os.path.split(config.ingest_glob)
    logger.info("Ingesting PDF documents from: %s", ingest_dir)
    loader = DirectoryLoader(
        ingest_dir,
        glob=ingest_glob,
        loader_cls=PyPDFLoader,
        use_multithreading=True,  # Speed up if there are many PDFs
    )

    docs = [document async for document in loader.alazy_load()]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size)
    documents = text_splitter.split_documents(docs)
    vector = await FAISS.afrom_documents(documents, embeddings)

    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "pdf_file_ingest",
        config.description,
    )

    async def _inner(query: str) -> str:
        return await retriever_tool.arun(query)

    yield FunctionInfo.from_fn(_inner, description=config.description)