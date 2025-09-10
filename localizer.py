from werkzeug.utils import secure_filename
import os
import zipfile
import time
from pathlib import Path
import networkx as nx
import pydot
import tempfile
import json
from openai import OpenAI
import pickle
import os
from tqdm import tqdm
import re
import networkx as nx
import subprocess
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bug_localization import buglocalization
import threading
import json
import shutil
import hashlib
import argparse

def run_doxygen(doxyfile_path=None, output_dir=None):

    """
    Runs Doxygen with an optional Doxyfile path.
    If doxyfile_path is None, it will look for 'Doxyfile' in the current directory.
    """
    try:
        if doxyfile_path:
            result = subprocess.run(
                ["doxygen", doxyfile_path],
                cwd=output_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                return
        else:
            result = subprocess.run(
                ["doxygen"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )


    except subprocess.CalledProcessError as e:
        pass
    except FileNotFoundError:
        pass


def create_unique_folder():
    """
    Creates a unique folder with a 10-character hash from sha256.
    Ensures no duplication by checking existing folders.
    Returns the folder name created.
    """
    while True:
        # Generate random bytes and hash them
        rand_bytes = os.urandom(16)
        hash_str = hashlib.sha256(rand_bytes).hexdigest()

        # Take first 10 characters
        folder_name = hash_str[:15]

        folder_path = folder_name

        # If folder does not exist, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path


def localize_bug(doxyfile, project_dir, report, key):
    print("Starting to generate callgraph.....")
    run_doxygen(doxyfile, project_dir)
    print("Callgraph generation done!")
    print("Bug localization starts ...")
    bug_finder = buglocalization(project_dir, key, data["joern_dir"], data["model_name"])
    information = bug_finder.localize_bug(report)
    full_path = os.path.join(project_dir, "buginfo.txt")
    #with open(f"{full_path}", "w", encoding="utf-8") as f:
    #    f.write(f"{information}")
    print(information)
    

if __name__ == "__main__":
    #localize_bug()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="path of the config file")
    args = parser.parse_args()
    config_file = args.config_file
    container_project_dir = create_unique_folder()
    os.makedirs(container_project_dir, exist_ok=True)
    with open("config.json", "r") as f:
        data = json.load(f)
    project_dir = data["project_dir"]
    project_dir = project_dir.split("/")[-1]
    shutil.move(project_dir, container_project_dir)
    #full_project_dir = os.path.join(container_project_dir, project_dir)
    report_file = data["report_file"]
    openai_key = data["openai_key"]
    doxyfile_dir = data["doxyfile_dir"]
    shutil.copy("./Doxyfile", container_project_dir)
    with open(openai_key) as f:
        key = f.read()
    key = key.strip()
    with open(report_file) as f:
        bug_report = f.read()
    localize_bug("./Doxyfile", container_project_dir, bug_report, key) 





