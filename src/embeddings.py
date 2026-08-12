from dotenv import load_dotenv
from src.loader import load_document
from src.Splitter import pdy_splitter
from langchain_jina import LateChunkEmbeddings
from src.logger import get_logger
import os

load_dotenv()

logger = get_logger(__name__)


def embedding():
    """
    Generate Embeddings
    """

    logger.info("Starting embedding model initialization")

    model_name = "jina-embeddings-v5-text-small"

    logger.info("Loading embedding model: %s", model_name)

    embedding_model = LateChunkEmbeddings(
        model=model_name, jina_api_key=os.getenv("JINA_API_KEY")
    )

    logger.info("Embedding model initialized successfully")

    return embedding_model


if __name__ == "__main__":

    logger.info("Starting embedding pipeline")

    logger.info("Loading documents")
    loader = load_document()

    logger.info("Documents loaded successfully: %s documents", len(loader))

    logger.info("Splitting documents into chunks")
    splitter = pdy_splitter(loader)

    logger.info("Document splitting completed: %s chunks created", len(splitter))

    first_chunk = splitter[0].page_content

    logger.info("Selected first chunk for embedding test")

    logger.info("First chunk length: %s characters", len(first_chunk))

    embedding_model = embedding()

    logger.info("Generating embedding for the first chunk")

    vector = embedding_model.embed_query(first_chunk)

    logger.info("Embedding generation completed successfully")

    logger.info("Generated vector dimension: %s", len(vector))

    print("Chunk:")
    print(first_chunk)

    print("\nVector dimensions:")
    print(len(vector))

    print("\nFirst 10 values:")
    print(vector[:10])

    logger.info("Embedding pipeline completed successfully")
