from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader


def loade_document():
    """Load PDF documents"""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    loader = PyPDFDirectoryLoader(str(data_dir))
    document = loader.load()

    return document


if __name__ == "__main__":
    docs = loade_document()
    print(docs[0])