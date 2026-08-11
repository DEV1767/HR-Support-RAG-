import os
import time
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from embeddings import embedding

load_dotenv()


def get_retriver():
    """
    Retrieve data from Vector store
    """

    embedding_model = embedding()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="Hr_support",
    )

    retriver = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

    return retriver


if __name__ == "__main__":

    retriver = get_retriver()

    query = "What is leave policy for the company?"
   
    start=time.perf_counter()
    docs = retriver.invoke(query)
    
    print(f"Retrieval time: {time.perf_counter() - start:.2f}s")
    print("=" * 60)
    print(f"Retrieved {len(docs)} Documents")
    print("=" * 60)

    for i, doc in enumerate(docs, start=1):

        print(f"\nDocument {i}")
        print("-" * 40)

        print(doc.page_content[:500])

        print("\nMetadata:")
        print(doc.metadata)