# Loads, splits, cleans documents

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from rank_bm25 import BM25Okapi
import subprocess
import uuid
import re
import nltk
import tempfile
import os

class DocumentManager:
    def __init__(self, github_repos):
        self.github_repos = github_repos
        self.all_split_documents = []

    # @staticmethod
    # def clean_and_tokenize(text):
    #     text = re.sub(r'\s+', ' ', text) # replace whitespace characters (spaces, tabs, newlines) with 1 space
    #     text = re.sub(r'<[^>]*>', '', text) # remove all HTML tags
    #     text = re.sub(r'\b(?:http|ftp)s?://\S+', '', text) # removes URLs starting with http or ftp.
    #     text = re.sub(r'\W', ' ', text) # replace non-word characters (anything other than letters, digits, and underscores) with a space.
    #     #text = re.sub(r'\d+', '', text) # removes digits
    #     #text = re.sub(r'\[.*?\]', '', text) # removes text within square brackets, including the brackets themselves.
    #     #text = re.sub(r'\(.*?\)', '', text) # removes text within parentheses, including the parentheses themselves.
    #     text = text.lower()
    #     return nltk.word_tokenize(text)

    @staticmethod
    def clone_github_repo(github_url, local_path):
        try:
            subprocess.run(['git', 'clone', github_url, local_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return False
    
    @staticmethod
    def split_documents_dict(documents_dict):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]) 
        split_documents = []
        for file_id, original_doc in documents_dict.items():
            split_docs = text_splitter.split_documents([original_doc])
            for split_doc in split_docs:
                split_doc.metadata['file_id'] = original_doc.metadata['file_id']
                split_doc.metadata['source'] = original_doc.metadata['source']
                split_doc.metadata['repo_name'] = original_doc.metadata['repo_name']

            split_documents.extend(split_docs)

        return split_documents

    def load_and_index_repo_readme(self, repo_path, repo_name):   
        glob_pattern = f'**/*.md'
        documents_dict = {}
        loader = DirectoryLoader(repo_path, glob=glob_pattern, loader_cls=UnstructuredMarkdownLoader)
        loaded_documents = loader.load() if callable(loader.load) else []
        for doc in loaded_documents:
            file_path = doc.metadata['source']
            relative_path = os.path.relpath(file_path, repo_path)
            file_id = str(uuid.uuid4())
            doc.metadata['source'] = relative_path
            doc.metadata['file_id'] = file_id
            doc.metadata['repo_name'] = repo_name
            documents_dict[file_id] = doc
        
        split_documents = self.split_documents_dict(documents_dict)
        return split_documents
    
        # index = None
        # if split_documents:
        #     tokenized_documents = [self.clean_and_tokenize(doc.page_content) for doc in split_documents]
        #     index = BM25Okapi(tokenized_documents)
        #     if index is None:
        #         print("No documents were found to index. Exiting.")
        #         exit()
        # return index, split_documents, [doc.metadata['repo_name'] for doc in split_documents]


    def process_repositories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo_url in self.github_repos:
                repo_name = repo_url.split('/')[-1]
                repo_local_path = os.path.join(temp_dir, repo_name)
                
                if self.clone_github_repo(repo_url, repo_local_path):
                    split_documents = self.load_and_index_repo_readme(repo_local_path, repo_name)
                    self.all_split_documents.extend(split_documents)
                else:
                    print(f"Skipping repository: {repo_url}")

        if not self.all_split_documents:
            print("No documents were found to index. Exiting.")
            return None

        return self.all_split_documents