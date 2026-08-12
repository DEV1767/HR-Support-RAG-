import time

from dotenv import load_dotenv
from operator import itemgetter

from src.retriver import get_retriver
from src.logger import get_logger

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda


load_dotenv()

logger = get_logger(__name__)


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

retriver = get_retriver()


prompt = ChatPromptTemplate.from_template("""
You are a professional AI HR Support Assistant working instead of the Company HR.

Answer ONLY from the provided context.

If the answer is not available in the context, reply:

"I couldn't find the information in the company's knowledge base."

Context:
{context}

Question:
{question}

Answer:
""")


def retrieve_and_format(question):

    start = time.perf_counter()

    logger.info("Retrieving documents for question: %s", question)

    docs = retriver.invoke(question)

    retrieval_time = time.perf_counter() - start

    logger.info(
        "Retrieved %s documents in %.2f seconds",
        len(docs),
        retrieval_time
    )

    return "\n\n".join(
        doc.page_content for doc in docs
    )


parallel_chain = RunnableParallel(
    {
        "context": (
            itemgetter("question")
            | RunnableLambda(retrieve_and_format)
        ),
        "question": itemgetter("question"),
    }
)


rag_chain = (
    parallel_chain
    | prompt
    | model
    | StrOutputParser()
)


if __name__ == "__main__":

    while True:

        question = input("\nEmployee: ")

        if question.lower() == "exit":
            break

        start = time.perf_counter()

        logger.info("RAG request started")

        print("\nAI Support:\n")

        for chunk in rag_chain.stream(
            {
                "question": question
            }
        ):
            print(chunk, end="", flush=True)

        print()

        total_time = time.perf_counter() - start

        print(
            f"\nTotal time: "
            f"{total_time:.2f}s"
        )

        logger.info(
            "RAG response completed in %.2f seconds",
            total_time
        )