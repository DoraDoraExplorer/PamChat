import os
import openai
from dotenv import load_dotenv, find_dotenv
import gradio as gr
datetime.today().strftime('%Y-%m-%d')
import schedule


# env vars for llm
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
USER_AGENT = os.environ['USER_AGENT']


from DocumentManager import DocumentManager
from EmbeddingManager import EmbeddingManager
from ConversationalRetrievalAgent import ConversationalRetrievalAgent

github_repos = [
    "https://github.com/pamgene/UKA_app",
    "https://github.com/pamgene/UKA_FC_app",
    "https://github.com/pamgene/MTvC_app",
    'https://github.com/pamgene/dascombat_shiny_fit_operator',
    'https://github.com/pamgene/log_cutoff_operator'
    ]




def setup_chatbot():
    rm -rf /docs/chroma  # remove old database files if any. this works in ubuntu
    
    # 1. Process repositories
    doc_manager = DocumentManager(github_repos)
    split_documents = doc_manager.process_repositories()
    
    # 2. Create vector database
    embedding_manager = EmbeddingManager()
    vectordb = embedding_manager.create_vector_database(split_documents)
    
    # Setup bot
    bot = ConversationalRetrievalAgent(vectordb)
    bot.setup_bot()
    return bot

def query_fn(input_name, input_q):
    session_id = input_name + datetime.today().strftime('%Y-%m-%d')
    response = bot.ask(query = input_q, 
                       sessionid = session_id)['answer']
    return response


if __name__ == "__main__":
    # setup chatbot every day once from github
    bot = setup_chatbot()
    # bot = schedule.every().day.at("00:00").do(setup_chatbot)

    # Build the UI
    iface = gr.Interface(
        fn=query_fn,
        inputs=[gr.components.Textbox(label="Enter your name"), gr.components.Textbox(label="Enter your question")],
        outputs=gr.components.Textbox(label="Answer"),
        title="PamGene Documentation Chat"
    )

    # Launch the UI but do not share it publicly
    iface.launch(share=False, server_port=8080)




