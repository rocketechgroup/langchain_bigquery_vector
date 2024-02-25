import os
from typing import List, Any, Optional, Dict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from bigquery_vector import create_vectors as bq_create_vectors, search_by_text as bq_search_by_text

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = 'europe-west2'
DATASET = "demo_langchain_dataset"
TABLE = "doc_and_vectors"

LLM = ChatOpenAI()


class BigQueryRetriever(BaseRetriever):
    project_id: str
    region: str
    dataset: str
    table: str
    filter: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.project_id = kwargs['project_id']
        self.region = kwargs['region']
        self.dataset = kwargs['dataset']
        self.table = kwargs['table']
        self.filter = kwargs['filter']

    def get_relevant_documents(self, query: str) -> List[Document]:
        return bq_search_by_text(
            project_id=self.project_id,
            region=self.region,
            dataset=self.dataset,
            table=self.table,
            filter=self.filter,
            query=query
        )


def ingest_vectors(url, context, truncate_all=False):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    metadata = [{"length": len(d.page_content), "context": context} for d in documents]

    bq_create_vectors(
        project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE,
        metadata=metadata, texts=[d.page_content for d in documents], truncate=truncate_all
    )


def search(query, filter):
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(LLM, prompt)
    retriever = BigQueryRetriever(project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE, filter=filter)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]


def ingest():
    ingest_vectors(context="langsmith_pricing", url="https://docs.smith.langchain.com/pricing", truncate_all=True)
    ingest_vectors(context="cloud_workstations",
                   url="https://practical-gcp.dev/scaling-development-teams-with-cloud-workstations/")
    ingest_vectors(context="bigframes",
                   url="https://practical-gcp.dev/serverless-distributed-processing-with-bigframes/")
    ingest_vectors(context="dataplex",
                   url="https://practical-gcp.dev/automated-data-profiling-and-quality-scan-via-dataplex/")


# ingest()
# answer = search(query="do you know anything about bigframes", filter={"context": "bigframes"})
# print(answer)
