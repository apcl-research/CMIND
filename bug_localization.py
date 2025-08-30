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
import time
from pathlib import Path
import networkx as nx
import pydot
import tempfile
import json




def extract_file_hierarchy(root_dir):
    hierarchy = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Add directory itself
        hierarchy.append(os.path.normpath(dirpath))
        # Add all files with their full path
        for f in filenames:
            full_path = os.path.normpath(os.path.join(dirpath, f))
            hierarchy.append(full_path)
    return hierarchy


def extract_graph_name(dot_string):
    match = re.search(r'digraph\s+"([^"]+)"', dot_string)
    return match.group(1) if match else None

def generate_path(edges, method):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    all_paths = {}
    final_paths = []
    target_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    for edge in edges:
        if(edge[0] == edge[1]):
            target_nodes.append(edge[0])
    for target in target_nodes:
        try:                                                                                                                                                    # Get all paths from Node39 to the target node
            paths = list(nx.all_simple_paths(G, source=f"{method}", target=target))
            if(target in all_paths):
                all_paths[target].extend(paths)
            else:
                all_paths[target] = paths
        except nx.NetworkXNoPath:
            all_paths[target] = []
        except nx.NodeNotFound:
            all_paths[target] = []
    temp_paths = []
    for target, paths in all_paths.items():
        for path in paths:
            final_paths.append(path)
    return final_paths

def run_joern_command(command, joern_dir):
    process = subprocess.Popen(
        [joern_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Enable text mode for easier string handling
        )
    
    stdout, stderr = process.communicate(input=command + "\n")
    
    if process.returncode == 0:
        return stdout
    else:
        return stderr



def get_callgraph(callgraphfiles):
    allcallgraph = []
    for file_path in callgraphfiles:

        with open(file_path, 'r',  encoding='latin-1') as file:
            content = file.read()
            graph_name= extract_graph_name(content)
            graph_name = graph_name.split(".")

        try:
            graph = nx.drawing.nx_pydot.read_dot(file_path)
        except:
            continue
        node_2_label = {}
        start_node = '.'.join(graph_name[-2:])
        for node in graph.nodes(data=True):
            node_name = node[0]
            node_label = node[1].get('label', node_name)  # Use node name if no label is present
            if "\\l" in node_label:
                node_label = node_label.replace("\\l", "")

            node_label = node_label.split(".")[-2:]
            node_label = '.'.join(node_label)
            node_label = node_label.strip('\"')
            node_2_label[node_name] = node_label
        edges = []
        allcalls = []
        for edge in graph.edges():
            source, destination = edge
            allcalls.append(source.split(".")[-1])
            allcalls.append(destination.split(".")[-1])
            source = node_2_label[source]
            destination = node_2_label[destination]
            edges.append((source, destination))
        allcallpaths = generate_path(edges, start_node)
        allcalls = list(set(allcalls))
        if("icgraph" in file_path):
            for call in allcallpaths:
                allcallgraph.append('<-'.join(call))
        else:
            for call in allcallpaths:
                allcallgraph.append('->'.join(call))
    return allcallgraph

def get_methods(command, joern_dir):
    process = subprocess.Popen(
        [joern_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Enable text mode for easier string handling
        )
    stdout, stderr = process.communicate(input=command + "\n")
    start_number = stdout.split("lineNumber = ")[-1]
    start_number = start_number.split("Some\x1b[39m(value = \x1b[32m")[1]
    start_number = start_number.split("\x1b[39m)")[0]

    end_number = stdout.split("lineNumberEnd = ")[-1]

    end_number = end_number.split("\x1b[33mSome\x1b[39m(value = \x1b[32m")[1]
    end_number = end_number.split("\x1b[39m)")[0]
    method_line = (int(start_number), int(end_number))
    return method_line


def find_file(filename, search_dir):
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def get_callgraph_methods(files, joern_dir):
    allcallmethods = {}
    for callfile in tqdm(files):
        #if("_8h_" in callfile):
        #    continue
        with open(f'{callfile}', 'r') as file:
            callgraph = file.read()
        pydot_graph = pydot.graph_from_dot_data(callgraph)[0]
        graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)

        node_info = {}
        for node in pydot_graph.get_nodes():
            name = node.get_name()
            if not name.startswith("Node"):
                continue
            attrs = node.get_attributes()
            label = attrs.get("label", "").replace("\\l", "").strip('"')
            url = attrs.get("URL", "").lstrip("$")
            file_path = url.split(".html")[0].replace('_8', '.').replace('__', '/')
            node_info[name] = (label, file_path)

        results = []
        for src, dst in graph.edges():
            if dst in node_info:
                callee, file_path = node_info[dst]
                results.append((callee, file_path))
    #source_file = find_method_in_dot_files()
    #if source_file:
    #    source_code = find_method_in_source_file(source_file)

        for method, path in results:
            path = path.replace('$', '')
            path = path.replace('\"', '')
            path = path.split("_2")[-1]
            code_root_path = callfile.split("html/")[0]
            filename = find_file(path,code_root_path )
            if(filename == None):
                continue
            with open(f'{filename}', 'r') as file:
                source_code = file.read()
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as tmp:
                tmp.write(source_code.encode())
                tmp_path = tmp.name
            method_query = f"""
            importCode(\"{tmp_path}/\");
            val method = cpg.method.name("{method}").head

            """
            try:
                method_line = get_methods(method_query, joern_dir)
            except:
                continue
            source_code = source_code.split("\n")
            allcallmethods[method] = '\n'.join(source_code[method_line[0]-1:method_line[1]])
            #allcallmethods.append('\n'.join(source_code[method_line[0]-1:method_line[1]]))
            os.remove(tmp_path)
    return allcallmethods


def remove_includes_and_comments(code):
    # Remove #include lines
    code = re.sub(r'^\s*#include.*$', '', code, flags=re.MULTILINE)
    
    # Remove /* ... */ multiline comments (including multiline)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Remove // single line comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    
    # Optionally, remove extra blank lines resulted from removals
    code = re.sub(r'\n\s*\n', '\n', code)
    
    return code.strip()

def get_methods_from_files(methods, project_dir, joern_dir, files=[]):
    pattern = re.compile(r'\d+\.\s+([a-zA-Z_][a-zA-Z0-9_:]*)')
    methods = pattern.findall(methods)
    paths = []
    for method in methods[:]:
        result = subprocess.run(
            ["grep", "-Er", "--include=*.c","--include=*.cpp", f'{method}', f"{project_dir}"],
            capture_output=True,
            text=True
            )
        paths.extend(re.findall(r'^(.*?):', result.stdout, re.MULTILINE))
    paths = [path.strip() for path in paths]
    paths = [p.removeprefix('./') for p in paths]
    paths = list(set(paths))
    codeblocks = ""
    source_code = ""
    codeblocks_temp = ""
    for file in files:
        file = file.strip()
        if(file in paths or paths == [] or "NONE" in methods):
            try:
                with open(f'{file}', 'r') as f:
                    codeblocks_temp += remove_includes_and_comments(f.read())
            except:
                continue
    methods = [method.strip() for method in methods]
    if(len(paths) >= 20 or paths == []):
        return codeblocks_temp
    else:
        for path in tqdm(paths):
            for method in methods:
                method = method.split("::")[-1]
                method = method.strip()
                command = f"""
                    importCode(\"{path}\");
                    val method = cpg.method.name("{method}").head
                """
                process = subprocess.Popen(
                    [f"{joern_dir}"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True  # Enable text mode for easier string handling
                )


    
                stdout, stderr = process.communicate(input=command + "\n")
                try:
                    start_number = stdout.split("lineNumber = ")[-1]
                    start_number = start_number.split("Some\x1b[39m(value = \x1b[32m")[1]
                    start_number = start_number.split("\x1b[39m)")[0]
                    end_number = stdout.split("lineNumberEnd = ")[-1]
                    end_number = end_number.split("\x1b[33mSome\x1b[39m(value = \x1b[32m")[1]
                    end_number = end_number.split("\x1b[39m)")[0]
                    method_line = (int(start_number), int(end_number))
                    with open(f'{path}', 'r') as f:
                        source_code = f.read()
                    source_code_line = source_code.split("\n")
                    codeblocks += '\n'.join(source_code_line[method_line[0]-1:method_line[1]])
                    break
               
                except:
                    continue
        return codeblocks + codeblocks_temp 

def bug_reasoning(codeblocks, callmethods, callpaths, allfiles, path_to_explore, bug_report, llm_bug_reasoning, project_dir, joern_dir):
    global final_output
    count_reason = 0
    reason_steps = []
    while(True):
        prompt2 = f'''The methods to localize the bugs are forward reasoning, backward reasoning, and code comprehension. Give you the related methods {codeblocks} and methods in call graph {callmethods}, and the call chain {path_to_explore}, could you use one of the methods to reason the bugs and localize the bugs based on bug report {bug_report}? Please do not assume any information that is not provided, any information not in call chain and related methods, the code snippets not in the provided code, and the name of the method. You should only look at the provided method to reason the bugs. If the methods that you need are not in the call chain or related methods, you can request the methods. Please remember to choose either forward reasoning, backward reasoning or code comprehension to reason where the bugs localize and keep it consistence with the previous reasoning methods if you have and do not generate new code snippet or new call chain for the specific methods as your task is not to generate anything. Instead, your task is only to localize the bugs. Please use the template REASONING METHODS:\t\nREASONING STEPS:\t\nHypothesis:\t\nMETHOD MISSING:\t\nMETHOD MISSING:\t\n(if you have multiple methods and please only provide method name and no other information is needed). Please do not make up any method name. Please follow the call chain. Otherwise, use REASONING METHODS:\t\nREASONING STEPS:\t\nHypothesis:\t\n. Please do not make up anything outside the provided information, but you can request it if needed. The reasoning method should be the same as the previous one if you have already chose one and you should follow the call chai and please make your reasoning steps specific and do not make up any method name.'''
        results = llm_bug_reasoning({"question": prompt2})
        reason = results["answer"]
        reason_steps.append(reason)
        pattern = re.compile(
            r'METHOD MISSING:[:\s]*([a-zA-Z_][\w:]+)',
            #r"METHOD MISSING:\s*(?:\n\s*|\s+)(?P<method>\w+)",
            re.MULTILINE
        )
        matches = list(pattern.finditer(reason))
        if(matches == [] or count_reason >= 2):
            break
        count_reason += 1
        related_methods = "METHOD:\n"
        for index, m in enumerate(matches):
            related_methods += f"{index+1}.\t{m.group(1)}\n"
        codeblocks += get_methods_from_files(related_methods, project_dir, joern_dir)

    return reason_steps, codeblocks


def bug_reasoning_dataflow(codeblocks, dataflow, allfiles, bug_report, llm_bug_reasoning, project_dir, joern_dir):

    global final_output
    count_reason = 0
    reasoning_steps = []
    

    while(True):

        prompt2 = f'''The methods to localize the bugs are forward reasoning, backward reasoning, and code comprehension. Give you the related methods {codeblocks} and dataflow {dataflow} could you use one of the methods to reason the bugs based on bug report {bug_report} for me ? Please DO NOT assume any information that is not provided and do not assume any information not in dataflow, the code snippets not in the provided code, and the name of the method. You should only look at the provided method to reason the bugs. If the methods that you need are not in the call chain or related methods, you can request the methods. Please remember to choose either forward reasoning, backward reasoning or code comprehension to reason where the bugs localize and keep it consistence with the previous reasoning methods if you have and do not generate new code snippet or data flow information for the specific methods as your task is not to generate anything. Instead, your task is only to localize the bugs. Please use the template REASONING METHODS:\t\nREASONING STEPS:\t\nHypothesis:\t\nMETHOD MISSING:\t\nMETHOD MISSING:\t\n(Only method name is needed).Otherwise, use REASONING METHODS:\t\nREASONING STEPS:\t\nHypothesis:\t\n. Please do not make up anything outside the provided information, but you can request it if needed. The reasoning method should be the same as the previous one if you have already chose one and you should follow the data flow. If the method has already in the related method, please use the template REASONING METHODS:\t\nREASONING STEPS:\t\nHypothesis:\t\n. Remember the format and do not mess up and please make your reasoning steps and the location of the bugs should be very specific e.g. you should mention the name of the variables or under which method. You should not mention the line numbers'''

        results = llm_bug_reasoning({"question": prompt2})
        reason = results["answer"]
        reasoning_steps.append(reason)
        pattern = re.compile(
            r'METHOD MISSING:[:\s]*([a-zA-Z_][\w:]+)',
            #r"METHOD MISSING:\s*(?:\n\s*|\s+)(?P<method>\w+)",
            re.MULTILINE
        )

        matches = list(pattern.finditer(reason))
        additional_methods = []
        additional_files = []
        if(not matches or count_reason >= 2):
            break
        count_reason += 1
        related_methods = "METHOD:\n"
        for index, m in enumerate(matches):
            related_methods += f"{index+1}.\t{m}\n"
        codeblocks += get_methods_from_files(related_methods, project_dir, joern_dir)



    return reasoning_steps, codeblocks


no_rag_prompt = PromptTemplate.from_template("""
{chat_history}

User: {question}
AI:""")

no_memory_prompt = PromptTemplate.from_template("""
User: {question}
AI:""")

class buglocalization:
    def __init__(self, project_dir, openaikey, joern_dir, model_name):
        client = OpenAI(api_key=openaikey)
        self.joern_dir = joern_dir
        os.environ["OPENAI_API_KEY"] = openaikey
        memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
        allfiles = extract_file_hierarchy(project_dir)
        self.allfiles = [x for x in allfiles if x.endswith(".c") or x.endswith(".cpp")]
        llm = ChatOpenAI(model_name=model_name, temperature=1.0)
        self.llm_method_finding = LLMChain(
            llm=llm,
            prompt=no_rag_prompt,
            memory=memory,
            output_key="answer"
        )
        self.llm_reason_sa = LLMChain(
            llm=llm,
            prompt=no_memory_prompt,
            memory=None,
            output_key="answer"
        )
        self.llm_bug_reasoning = LLMChain(
            llm=llm,
            prompt=no_memory_prompt,
            memory=None,
            output_key="answer"
        )
        self.project_dir = project_dir
        
    def localize_bug(self, bug_report):
        method_finding_prompot = f'''Given the bug report {bug_report}, can you tell me which methods and files would you look at based on the bug report. Please use the format: METHOD:\n1.\nFILE:\n1.\n. Please do not make up anything and just provide me the name of the methods and the name of the files without any explanation. Please only find three most important methods/files and please do not include () or any other information in the method name. The file should be in {self.allfiles}. Please be sure to reason the method name or file path based on the bug report and do not make up any method name as your job is to look at the bug report and find the related methods only. Note that you do not need to provide the file if you do not see the file. In this case, you can just say METHOD:\n1.FILE:1.NONE If you provide the file, please follow the provided file path. If you do not know the name of the method, please just give me the file name. In this case, you just need to give me METHOD:\n1.NONE\nFILE:\n1.\n. If you have multiple files/methods, please use the format METHOD:\n1.\n2.\nFILE:\n1.\n instead of outputing multplie tiomes 
    '''
        results = self.llm_method_finding({"question": method_finding_prompot})
        related_methods = results["answer"]
        files = []
        methodblocks = related_methods.split("METHOD:")[-1]
        fileblocks = related_methods.split("FILE:")[-1]
        methodbloks = "METHOD:" + methodblocks
        pattern = r"\d+\.\s+(NONE|[^\n]+\.c(?:pp)?)"
        files = re.findall(r"\d+\.\s*(NONE|[^\n]+?\.c(?:pp)?)", fileblocks)
        files = [f for f in files if f.endswith(('.c', '.cpp'))]
        files = list(set(files))
        codeblocks = get_methods_from_files(methodblocks, self.project_dir, self.joern_dir, files)
        prompt2 = f'''Now you can use call graph analysis, a method to analyze how the methods call each other and data flow analysis, a method to analyze how data flow in the problem. Given the bug report {bug_report} and the related methods {codeblocks}, could you tell me which method would you use to analyze the bugs? Please just tell me what you would use and be sure to refer to the context and not make up anything yourself. please use the template: data flow analysis:\tsource:\tsink:\t if you think data flow analysis is an appropriate method. Please note that source should be the name of the method in {related_methods} and sink should be in the a function that is inside the same method (if this is not the case, please suggest call graph) and do not provide any other information and the example is data flow analysis:\tsource:\ta\nsink:\tb\n where a is the method name and b is the function name. otherwise: call graph analysis.
'''
        results = self.llm_reason_sa({"question": prompt2})
        analysis_methods = results["answer"]

        dataflow_analysis = re.search(
        r'data flow analysis:\s*source:\s*(\w+)\s*sink:\s*(\w+)',
        analysis_methods, re.IGNORECASE
        )
        if dataflow_analysis:
            source = dataflow_analysis.group(1)
            sink = dataflow_analysis.group(2)
            source = source.strip()
            sink = sink.strip()
            with open(os.path.join(self.project_dir, "temp.c"), "w") as f:
                f.write(codeblocks)
            query = f"""
            importCode(\"temp.c\");
            def source = cpg.method.name("{source}").parameter
            def sink = cpg.call.name("{sink}").argument
            sink.reachableByFlows(source).l
            """
            dataflow = run_joern_command(query, self.joern_dir)
            reasoning_steps, codeblocks = bug_reasoning_dataflow(codeblocks, dataflow, self.allfiles, bug_report, self.llm_bug_reasoning, self.project_dir, self.joern_dir)
        else:

            callgraph_file_path = []
            methods_section = re.search(r'METHOD:\s*(.*?)\n\s*FILE:', related_methods, re.S)
            methods = re.findall(r'\d+\.\s*([\w]+)', methods_section.group(1))
            #if(methods == []):
            #for file in files:
            #    command = f"""
            #            importCode(\"{file}\");
            #            cpg.method.internal.name.filterNot(_.startsWith("<")).l           
            #    """
            #    process = subprocess.Popen(
            #            [f"{self.joern_dir}"],
            #            stdin=subprocess.PIPE,
            #            stdout=subprocess.PIPE,
            #            stderr=subprocess.PIPE,
            #            text=True  # Enable text mode for easier string handling
            #        )
            #    stdout, stderr = process.communicate(input=command + "\n")
            #    print("-0-----", stdout)
            for method in methods:
                method = method.strip()
                callgraph_dir = os.path.join(self.project_dir, "html")
                result = subprocess.run(
                ["grep", "-rn", "--include=*.dot", f'digraph "{method}"', f"{callgraph_dir}"],
                capture_output=True,
                text=True
                )
                callgraph_file_path.extend([line.split(":", 1)[0] for line in result.stdout.strip().splitlines()])
            callmethod_dir = os.path.join(self.project_dir,"callgraph.pkl")
            if os.path.isfile(f"{callmethod_dir}"):
                callmethods = pickle.load(open(f"{callmethod_dir}", "rb"))
            else:
                callmethods = get_callgraph_methods(callgraph_file_path, self.joern_dir)
                pickle.dump(callmethods, open(f"{callmethod_dir}", "wb"))
            callpaths = get_callgraph(callgraph_file_path)
            prompt2 = f'''Now, can you look at the call path {callpaths} and tell me which one do you need to locate the bugs based on the bug report {bug_report} and related methods {codeblocks}. Please give me the complete paths as shown in call path with the template path:\n1.\t The between method sign is "<-" or "->". Please only tell me the paths that help to localize the bugs and the callchain should be in the provided paths and please only provide the call chain without any explanation. Please follow the call chain that I give you. Do not have () in the call chain.
    '''
            results = self.llm_reason_sa({"question": prompt2})
            path_to_explore = results["answer"]
            matches = re.findall(r'\d+\.\s+(.*)', path_to_explore)
            callchain = []
            callchain.extend([list(reversed([fn.strip() for fn in path.split('<-')])) for path in matches])
            callchain.extend([list(reversed([fn.strip() for fn in path.split('<-')])) for path in matches])
            callchain_methods = []
            for chain in callchain:
                for method_name in chain:
                    try:
                        callchain_methods.append(callmethods[method_name])
                    except:
                        continue
            callmethods = '\n'.join(callchain_methods)
            reasoning_steps, codeblocks = bug_reasoning(codeblocks, callmethods, callpaths, self.allfiles, path_to_explore, bug_report, self.llm_bug_reasoning, self.project_dir, self.joern_dir)

        prompt2 = f'''Based on the reasoning steps {reasoning_steps} and related code blocks {codeblocks} that you have, could you summarize the information for me? Do not assume anything outside of provided information and do not generate the code snippets that you do not have. Your task is only to summarize the bugs based on the information that you have.'''
        results = self.llm_bug_reasoning({"question": prompt2})
        information = results["answer"]
        return information







