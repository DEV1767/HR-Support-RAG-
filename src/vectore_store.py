import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from loader import loade_document
from Splitter import pdy_splitter
from embeddings import embedding

load_dotenv()


def create_vector_store():

    document = loade_document()

    chunks = pdy_splitter(document)

    print(f"Total chunks: {len(chunks)}")

    embedding_model = embedding()

    print("Connecting to Qdrant Cloud.......")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=60
    )

    print(client.get_collections())
    print("Connected successfully!")

    texts = [chunk.page_content for chunk in chunks]

    print("Generating embeddings...")

    vectors = embedding_model.embed_documents(texts)

    print(f"Generated vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    points = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):

        point = PointStruct(
            id=i,
            vector=vector,
            payload={"page_content": chunk.page_content, "metadata": chunk.metadata},
        )

        points.append(point)

    print("Uploading vectors to Qdrant...")

    client.upsert(collection_name="Hr_support", points=points)

    print("Upload completed successfully!")

    collection_info = client.get_collection("Hr_support")

    print(f"Points currently in Hr_support: " f"{collection_info.points_count}")

    return client


if __name__ == "__main__":
    create_vector_store()

    print("\nVector database created successfully!")
