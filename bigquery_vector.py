import os
import json

from google.cloud import bigquery
# from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores.utils import DistanceStrategy

from bigquery_vector_search import BigQueryVectorSearchLocal


class Properties:
    def __init__(self, project_id, region, dataset, table):
        self.project_id = project_id
        self.region = region
        self.dataset = dataset
        self.table = table


class VectorStoreFactory:
    def __init__(self, properties: Properties):
        self.embedding = VertexAIEmbeddings(
            model_name="textembedding-gecko@latest", project=properties.project_id
        )
        self.properties = properties

    def create_store(self):
        store = BigQueryVectorSearchLocal(
            project_id=self.properties.project_id,
            dataset_name=self.properties.dataset,
            table_name=self.properties.table,
            location=self.properties.region,
            embedding=self.embedding,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
        )

        return store


def create_vectors(project_id, region, dataset, table, metadata, texts: [str], truncate=False):
    properties = Properties(project_id, region, dataset, table)
    store = VectorStoreFactory(properties).create_store()
    client = bigquery.Client(project=properties.project_id, location=properties.region)
    client.create_dataset(dataset=dataset, exists_ok=True)

    if truncate:
        query_job = client.query(f"""
        TRUNCATE TABLE `{properties.project_id}.{properties.dataset}.{properties.table}`
        """)
        query_job.result()

    store.add_texts(metadatas=metadata, texts=texts)


def search_by_text(project_id, region, dataset, table, query, filter):
    properties = Properties(project_id, region, dataset, table)
    store = VectorStoreFactory(properties).create_store()
    return store.similarity_search(query=query, filter=filter)
