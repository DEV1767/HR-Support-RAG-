from dotenv import load_dotenv
from src.loader import loade_document
from src.Splitter import pdy_splitter
from langchain_jina import LateChunkEmbeddings

load_dotenv()


def embedding():
    """
    Generate Embeddings
    """
    embedding_model = LateChunkEmbeddings(
        model="jina-embeddings-v5-text-small"
    )

    return embedding_model


if __name__ == "__main__":

   
    loader = loade_document()

    
    splitter = pdy_splitter(loader)

    
    first_chunk = splitter[0].page_content

   
    embedding_model = embedding()

    
    vector = embedding_model.embed_query(first_chunk)

    print("Chunk:")
    print(first_chunk)

    print("\nVector dimensions:")
    print(len(vector))

    print("\nFirst 10 values:")
    print(vector[:10])