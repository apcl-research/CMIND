from flask import Flask, render_template, url_for, request, redirect, flash
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from bug_localization import buglocalization
import threading
import json
import shutil
import hashlib

with open("config.json", "r") as f:
    data = json.load(f)


openaikey = data["openai_key"]

with open(f'{openaikey}', 'r') as f:
    key = f.readline().strip()

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow only certain file extensions (optional)
ALLOWED_EXTENSIONS = {'zip'}



def extract_zip(zip_path, extract_to):
    # Check if the ZIP file exists
    if not os.path.isfile(zip_path):
        return
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

        
def send_email_notification(subject, body, to_email, file_path=None):
    sender_email = data["sender_email"]
    sender_password = data["sender_password"]  # Gmail App Password

    # Create multipart/mixed email
    msg = MIMEMultipart("mixed")
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Add text body
    msg.attach(MIMEText(body, "plain"))

    # Attach the file
    if file_path:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
            "Content-Disposition",
            f'attachment; filename="{os.path.basename(file_path)}"'
            )
            msg.attach(part)
        else:
            return

    try:
        # Connect to Gmail SMTP
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        pass



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








@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def build_file_tree(root_dir, base_url=""):
    file_tree = []
    for item in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, item)
        rel_path = os.path.join(base_url, item)
        if os.path.isdir(path):
            file_tree.append({
                'name': item,
                'type': 'folder',
                'children': build_file_tree(path, rel_path)
            })
        else:
            file_tree.append({
                'name': item,
                'type': 'file',
                'path': rel_path.replace("\\", "/")
            })
    return file_tree


def create_unique_folder(base_dir="."):
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
        folder_name = hash_str[:10]

        folder_path = base_dir + "_" + folder_name

        # If folder does not exist, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path

def localize_bug(doxyfile, project_dir, report, key, name, email):
    run_doxygen(doxyfile, project_dir)
    bug_finder = buglocalization(project_dir, key, data["joern_dir"], data["model_name"])
    information = bug_finder.localize_bug(report)
    full_path = os.path.join(project_dir, "buginfo.txt")
    with open(f"{full_path}", "w", encoding="utf-8") as f:
        f.write(f"{information}")
    send_email_notification(
            subject="Bug Localization Finished",
            body=f"Hi {name},\nwe have finished your bug localization. The attached is the result.",
            to_email=f"{email}",
            file_path=f"{full_path}"
    )
    if os.path.exists(project_dir) and os.path.isdir(project_dir):
        shutil.rmtree(project_dir)


@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':

        name = request.form.get('name')
        email = request.form.get('email')
        report = request.form.get('report')
        # Make sure the upload folder exists
        file = request.files.get('file')
        email = email.strip()
        project_dir = create_unique_folder(email.split("@")[0])
        os.makedirs(project_dir, exist_ok=True)
        shutil.copy(os.path.join(data["doxyfile_dir"], "Doxyfile"), os.path.join(project_dir, "Doxyfile"))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(project_dir, filename))
            extract_zip(os.path.join(project_dir,filename), project_dir)
            threading.Thread(
                target=localize_bug,
                args=("Doxyfile", project_dir, report, key, name, email),
                daemon=True
            ).start()
            
            send_email_notification(
                subject="Bug Localization Start",
                body=f"Hi {name},\nwe have started to localize the bug from the bug report and file you submitted. We will notify you when it's done.",
                to_email=f"{email}", 
                #file_path="/home/chiayi/bug_localization/test.txt"
            )
            flash('File successfully uploaded and message received!', 'success')
        else:
            flash('Invalid file or no file uploaded.', 'danger')
    return redirect(url_for('home'))

    #files = os.listdir(UPLOAD_FOLDER)
    #tree = build_file_tree(UPLOAD_FOLDER)
    #return render_template('results.html', tree=tree)


@app.route('/view')
def view_file():
    rel_path = request.args.get('file')
    abs_path = os.path.join(UPLOAD_FOLDER, rel_path)

    tree = build_file_tree(UPLOAD_FOLDER)

    if not os.path.isfile(abs_path):
        return "File not found", 404

    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    return render_template('results.html', tree=tree, selected_file=rel_path, file_content=content)







if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
