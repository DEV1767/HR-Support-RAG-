from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from src.logger import get_logger

logger = get_logger(__name__)


def load_document():
    """Load PDF documents."""

    logger.info("Starting document loading")

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    logger.info("Project base directory: %s", base_dir)
    logger.info("Looking for PDF documents in: %s", data_dir)

    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        return []

    logger.info("Data directory found")

    loader = PyPDFDirectoryLoader(str(data_dir))

    logger.info("PDF loader initialized")

    documents = loader.load()

    logger.info(
        "PDF loading completed successfully. Loaded %s document pages",
        len(documents)
    )

    if documents:
        logger.info(
            "First document source: %s",
            documents[0].metadata.get("source")
        )

    return documents


if __name__ == "__main__":
    logger.info("Starting loader test")

    docs = load_document()

    if docs:
        print(docs[0])
    else:
        logger.warning("No documents found")

    logger.info("Loader test completed")