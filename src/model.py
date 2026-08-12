import time

from dotenv import load_dotenv
from operator import itemgetter

from src.retriver import get_retriver
from src.logger import get_logger

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

from src.guardial import check_input, check_output, Refusal_message

load_dotenv()

logger = get_logger(__name__)


model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


retriver = get_retriver()


prompt = ChatPromptTemplate.from_template("""
You are a professional and friendly AI HR Support Assistant
working instead of the Company HR.

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

    logger.info("Retrieved %s documents in %.2f seconds", len(docs), retrieval_time)

    return "\n\n".join(doc.page_content for doc in docs)


parallel_chain = RunnableParallel(
    {
        "context": (itemgetter("question") | RunnableLambda(retrieve_and_format)),
        "question": itemgetter("question"),
    }
)


rag_chain = parallel_chain | prompt | model | StrOutputParser()


if __name__ == "__main__":

    while True:

        question = input("\nEmployee: ")

        if question.lower() == "exit":
            break

        start = time.perf_counter()

        logger.info("User input received")

        logger.info("Checking input with guard model")

        input_result = check_input(question)

        logger.info("Input guard result: %s", input_result)

        is_input_safe, input_reason = input_result

        if not is_input_safe:

            logger.warning("Input rejected by guard | reason: %s", input_reason)

            print(f"\nAI Support:\n{Refusal_message}")

            continue

        logger.info("Input approved by guard")

        logger.info("RAG request started")

        try:

            answer_chunks = []

            for chunk in rag_chain.stream({"question": question}):

                answer_chunks.append(chunk)

            answer = "".join(answer_chunks)

            logger.info("RAG answer generated successfully")

        except Exception as error:

            logger.exception("RAG request failed: %s", error)

            print(
                "\nAI Support:\n"
                "Sorry, something went wrong while processing your request."
            )

            continue

        logger.info("Checking output with guard model")

        output_result = check_output(answer)

        logger.info("Output guard result: %s", output_result)

        is_output_safe, output_reason = output_result

        if not is_output_safe:

            logger.warning("Output rejected by guard | reason: %s", output_reason)

            print(f"\nAI Support:\n{Refusal_message}")

            continue

        logger.info("Output approved by guard")

        print(f"\nAI Support:\n{answer}")

        total_time = time.perf_counter() - start

        print(f"\nTotal time: " f"{total_time:.2f}s")

        logger.info("RAG response completed in %.2f seconds", total_time)
