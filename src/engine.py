from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings, OllamaLLM

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

llm = OllamaLLM(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_conversational_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, formulate a "
                "standalone question which can be understood without the chat "
                "history. Do NOT answer the question, just reformulate it.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Use the following retrieved "
                "context to answer the question. Keep your answer concise and "
                "practical.\n\nContext: {context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)
