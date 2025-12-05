import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #OpenAIEmbeddings() defaults to model text-embedding-3-large, which outputs 1536-dim vectors.

    llm = ChatOpenAI()

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
#    result = chain.invoke(input={})
#    print(result.content)


    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})
    # this is the retrivel chain, not llm  connected

    print(result)


    template= """ Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say you don't know.
    Don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say thanks for asking at the end of the answer
    
    {context}
    
    Question: {question}
    
    Helpful answer:"""

    custom_rag_prompt= PromptTemplate.from_template(template)

    rag_chain= (
        {"context": vectorstore.as_retriever() | format_docs, "question":RunnablePassthrough()}
        # context: is about how to get context after retrieve. Runnable is getting the last respond
        | custom_rag_prompt
        | llm
    )

    res= rag_chain.invoke(query)
    print(res)