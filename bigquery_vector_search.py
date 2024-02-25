import json
import sys
from typing import Optional, List, Dict, Any, Tuple

from langchain_community.vectorstores import bigquery_vector_search as bigquery_vector_search_module
from langchain_community.vectorstores.bigquery_vector_search import BigQueryVectorSearch
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document


# Reason for subclassing:
# - fixing the bug self.metadata_field is a dict if the data is retrieved from the JSON data type from BQ
class BigQueryVectorSearchLocal(BigQueryVectorSearch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _search_with_score_and_embeddings_by_vector(
            self,
            embedding: List[float],
            k: int = bigquery_vector_search_module.DEFAULT_TOP_K,
            filter: Optional[Dict[str, Any]] = None,
            brute_force: bool = False,
            fraction_lists_to_search: Optional[float] = None,
    ) -> List[Tuple[Document, List[float], float]]:
        from google.cloud import bigquery

        # Create an index if no index exists.
        if not self._have_index and not self._creating_index:
            self._initialize_vector_index()
        # Prepare filter
        filter_expr = "TRUE"
        if filter:
            filter_expressions = []
            for i in filter.items():
                if isinstance(i[1], float):
                    expr = (
                        "ABS(CAST(JSON_VALUE("
                        f"base.`{self.metadata_field}`,'$.{i[0]}') "
                        f"AS FLOAT64) - {i[1]}) "
                        f"<= {sys.float_info.epsilon}"
                    )
                else:
                    val = str(i[1]).replace('"', '\\"')
                    expr = (
                        f"JSON_VALUE(base.`{self.metadata_field}`,'$.{i[0]}')"
                        f' = "{val}"'
                    )
                filter_expressions.append(expr)
            filter_expression_str = " AND ".join(filter_expressions)
            filter_expr += f" AND ({filter_expression_str})"
        # Configure and run a query job.
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("v", "FLOAT64", embedding),
            ],
            use_query_cache=False,
            priority=bigquery.QueryPriority.BATCH,
        )
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            distance_type = "EUCLIDEAN"
        elif self.distance_strategy == DistanceStrategy.COSINE:
            distance_type = "COSINE"
        # Default to EUCLIDEAN_DISTANCE
        else:
            distance_type = "EUCLIDEAN"
        if brute_force:
            options_string = ",options => '{\"use_brute_force\":true}'"
        elif fraction_lists_to_search:
            if fraction_lists_to_search == 0 or fraction_lists_to_search >= 1.0:
                raise ValueError(
                    "`fraction_lists_to_search` must be between " "0.0 and 1.0"
                )
            options_string = (
                ',options => \'{"fraction_lists_to_search":'
                f"{fraction_lists_to_search}}}'"
            )
        else:
            options_string = ""
        query = f"""
            SELECT
                base.*,
                distance AS _vector_search_distance
            FROM VECTOR_SEARCH(
                TABLE `{self.full_table_id}`,
                "{self.text_embedding_field}",
                (SELECT @v AS {self.text_embedding_field}),
                distance_type => "{distance_type}",
                top_k => {k}
                {options_string}
            )
            WHERE {filter_expr}
            LIMIT {k}
        """
        document_tuples: List[Tuple[Document, List[float], float]] = []
        # TODO(vladkol): Use jobCreationMode=JOB_CREATION_OPTIONAL when available.
        job = self.bq_client.query(
            query, job_config=job_config, api_method=bigquery.enums.QueryApiMethod.QUERY
        )
        # Process job results.
        for row in job:
            metadata = row[self.metadata_field]
            if metadata:
                # Fixing the bug self.metadata_field is a dict if the data is retrieved from the JSON data type from BQ
                try:
                    metadata = json.loads(metadata)
                except TypeError:
                    pass
            else:
                metadata = {}
            metadata["__id"] = row[self.doc_id_field]
            metadata["__job_id"] = job.job_id
            doc = Document(page_content=row[self.content_field], metadata=metadata)
            document_tuples.append(
                (doc, row[self.text_embedding_field], row["_vector_search_distance"])
            )
        return document_tuples
