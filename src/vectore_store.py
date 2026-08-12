import os
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.loader import loade_document
from src.Splitter import pdy_splitter
from src.embeddings import embedding
from src.logger import get_logger


load_dotenv()

logger = get_logger(__name__)


def create_vector_store():

    logger.info("Starting vector store creation")

   
    logger.info("Loading documents...")
    start = time.perf_counter()

    document = loade_document()

    logger.info(
        "Documents loaded successfully: %s pages in %.2f seconds",
        len(document),
        time.perf_counter() - start
    )

    
    logger.info("Splitting documents...")
    start = time.perf_counter()

    chunks = pdy_splitter(document)

    logger.info(
        "Document splitting completed: %s chunks created in %.2f seconds",
        len(chunks),
        time.perf_counter() - start
    )

    # Generate embeddings
    logger.info("Initializing embedding model...")
    embedding_model = embedding()

    texts = [chunk.page_content for chunk in chunks]

    logger.info(
        "Generating embeddings for %s chunks...",
        len(texts)
    )

    start = time.perf_counter()

    vectors = embedding_model.embed_documents(texts)

    logger.info(
        "Embedding generation completed: %s vectors in %.2f seconds",
        len(vectors),
        time.perf_counter() - start
    )

    logger.info(
        "Vector dimension: %s",
        len(vectors[0])
    )

  
    logger.info("Connecting to Qdrant Cloud...")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )

    logger.info("Connected to Qdrant successfully")

    logger.info(
        "Available Qdrant collections: %s",
        client.get_collections()
    )

    
    logger.info("Creating Qdrant points...")

    points = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):

        point = PointStruct(
            id=i,
            vector=vector,
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            },
        )

        points.append(point)

    logger.info(
        "Created %s Qdrant points",
        len(points)
    )

    
    logger.info("Uploading vectors to Qdrant collection: Hr_support")

    start = time.perf_counter()

    client.upsert(
        collection_name="Hr_support",
        points=points
    )

    logger.info(
        "Vector upload completed in %.2f seconds",
        time.perf_counter() - start
    )

   
    collection_info = client.get_collection("Hr_support")

    logger.info(
        "Points currently in Hr_support: %s",
        collection_info.points_count
    )

    logger.info("Vector store creation completed successfully")

    return client


if __name__ == "__main__":

    create_vector_store()

    logger.info("Vector database created successfully!")