from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.loader import load_document
from src.logger import get_logger

logger = get_logger(__name__)


def pdy_splitter(document):
    """
    Split the documents into chunks
    """

    logger.info("Splitting %s documents", len(document))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(document)

    logger.info(
        "Splitting completed: %s chunks formed",
        len(chunks)
    )

    return chunks


if __name__ == "__main__":
    document = load_document()
    chunks = pdy_splitter(document)

    print(chunks)