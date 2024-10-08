from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class EmbeddingManager:
    def __init__(self, persist_directory='docs/chroma/'):
        self.persist_directory = persist_directory
        self.vectordb = None

    def create_vector_database(self, documents):
        if not documents:
            print("No documents to create vector database.")
            return

        embedding = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=self.persist_directory
        )
        #print('vectordb collection count', self.vectordb._collection.count())
        return self.vectordb