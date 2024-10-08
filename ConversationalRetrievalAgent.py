from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class ConversationalRetrievalAgent:
    def __init__(self, vectordb):
        self.qa_chain = None
        self.vectordb = vectordb
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        self.store = {}
    
    def contextualize_q_prompt(self):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

        contextualized_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}") # This adds the latest user question to the template, which is referenced as {input}. 
                #("human", "{context}") # This ensures the prompt accepts context as an input variable
            ]
        )
        return contextualized_q_prompt
    
    def create_qa_prompt(self):
        system_prompt = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer \
                         the question. If you don't know the answer, say that you don't know. Keep the answer concise."
                        "\n\n"
                        "{context}"
                        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return qa_prompt
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def setup_bot(self):
        if not self.vectordb:
            print("Vector database not created.")
            return
        
        # # Non-history aware retriever: this is enough, return qa_chain
        # self.qa_chain = RetrievalQA.from_chain_type(
        #     self.llm,
        #     retriever=self.vectordb.as_retriever()
        # )

        # 1. Contextualize prompt and Create a chain that takes conversation history and returns documents.
        contextualized_q_prompt = self.contextualize_q_prompt()

        history_aware_retriever = create_history_aware_retriever(
            llm = self.llm, 
            retriever = self.vectordb.as_retriever(), 
            prompt = contextualized_q_prompt
        )
        
        # 2. Create a chain for passing a list of Documents to a model.
        qa_prompt = self.create_qa_prompt()
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # 3.
        retrieval_chain = create_retrieval_chain(retriever = history_aware_retriever, 
                                               combine_docs_chain = question_answer_chain)
        
        # 4. 
        self.qa_chain = RunnableWithMessageHistory(
            retrieval_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return self.qa_chain
    

    def ask(self, query, sessionid):
        result = self.qa_chain.invoke({"input": query},
                                       config={"configurable": {"session_id": sessionid}})
        return result
    