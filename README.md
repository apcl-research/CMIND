# CMind: An AI Agent for Localizing C Memory Bugs

Proposed by 
- [Chia-Yi Su](https://chiayisu.github.io/)
- [Collin McMillan](https://sdf.org/~cmc/)

## Quick link
- [To-do list](#to-do-list)
- [How to run it](#how-to-run-it)
- [Citation](#citation)

## To-do list
  
- To set up your local environment, run the following command. We recommend the use of a virtual environment for running the experiments.
```
pip install -r requirements.txt
```

- Please also install [Docker](https://www.docker.com/get-started/) as this requires docker to install some dependencies.

- Set-up the path in ```config.json```. The parameters are as follows:
```
doxyfile_dir: Path of Doxyfile for generating callgraph
model_name: GPT models tag
joern_dir: Joern directory e.g. "/opt/joern/joern-cli/joern". Note that it needs to be the directory in the docker container not local directory
_joern_dir: Joern directory for tesing locally e.g. /home/chiayi/bin/joern/joern-cli/joern
openai_key: Path for your OpenAI key
project_dir: Dirctory of your source code
report_file: Directory for your bug report e.g. "./bug_report.txt"
container_name: The name of your docker container e.g. debugger_container
```

## How to run it 

```
python3 run.py --config-file=config.json
```
```
--config-file: Path of your config.json file
```


## Citation
If you use this work in an academic paper, please cite the following:
```
