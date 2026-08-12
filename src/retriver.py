import os
import time

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

from src.embeddings import embedding
from src.logger import get_logger


load_dotenv()

logger = get_logger(__name__)


def get_retriver():
    """
    Retrieve data from Vector store.
    """

    logger.info("Starting retriever initialization")

    logger.info("Initializing embedding model")
    embedding_model = embedding()

    logger.info("Connecting to Qdrant collection: Hr_support")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="Hr_support",
    )

    logger.info("Successfully connected to Qdrant")

    retriver = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

    logger.info(
        "Retriever initialized successfully | search_type=similarity | top_k=5"
    )

    return retriver


if __name__ == "__main__":

    logger.info("Starting retriever test")

    retriver = get_retriver()

    query = "What is leave policy for the company?"

    logger.info("Starting document retrieval")
    logger.info("Query: %s", query)

    start = time.perf_counter()

    docs = retriver.invoke(query)

    retrieval_time = time.perf_counter() - start

    logger.info(
        "Document retrieval completed | documents=%s | time=%.2fs",
        len(docs),
        retrieval_time
    )

    print(f"Retrieval time: {retrieval_time:.2f}s")

    print("=" * 60)
    print(f"Retrieved {len(docs)} Documents")
    print("=" * 60)

    for i, doc in enumerate(docs, start=1):

        logger.info(
            "Retrieved document %s | source=%s",
            i,
            doc.metadata.get("source")
        )

        print(f"\nDocument {i}")
        print("-" * 40)

        print(doc.page_content[:500])

        print("\nMetadata:")
        print(doc.metadata)

    logger.info("Retriever test completed successfully")