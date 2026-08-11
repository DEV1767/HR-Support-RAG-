from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.loader import loade_document


def pdy_splitter(document):
    """
    Split the documnets into chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

    chunks = splitter.split_documents(document)

    return chunks


if __name__ == "__main__":
    document = loade_document()
    chunks = pdy_splitter(document)
    
    print(chunks)
