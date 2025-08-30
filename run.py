import subprocess
import json
import argparse
import os

# Path to input file on host




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="path of the config file")
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as f:
        data = json.load(f)


    container_name = data["container_name"]
    container_path = "/app"
    project_dir = data["project_dir"]

    # Step 1: Copy the input file into the container

    try:
        subprocess.run(
            ["sudo", "docker", "build", "-t", "debugger", "."],
            check=True
        )
        print("✅ Image 'debugger' built successfully")
    except subprocess.CalledProcessError as e:
        print("❌ Error building image:", e)

    # Step 2: Run the container
    try:
        subprocess.run(
        ["sudo", "docker", "run", "-d", "--name", "debugger_container", "debugger", "tail", "-f", "/dev/null"],
        check=True
        )
        print("✅ Container 'debugger_container' started successfully")
    except subprocess.CalledProcessError as e:
        print("❌ Error starting container:", e)


    try:

        subprocess.run(
            ["docker", "cp", project_dir, f"{container_name}:{container_path}"],
            check=True
        )
        subprocess.run(
            ["docker", "cp", config_file, f"{container_name}:{container_path}"],
            check=True
        )

        print(f"Copied {project_dir} to {container_name}:{container_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")

    subprocess.run(["docker", "exec", container_name, "python3", "localizer.py",  "--config-file", f"{config_file}"], check=True)
    #subprocess.run(["docker", "stop", container_name], check=True)
    #print("docker container has been stoped")
    #subprocess.run(["docker", "rm", container_name], check=True)
    #print("docker container has been removed")
'''


# Step 1: Copy the input file into the container
try:
    subprocess.run(
        ["docker", "cp", host_file, f"{container_name}:{container_path}"],
        check=True
    )
    print(f"Copied {host_file} to {container_name}:{container_path}")
except subprocess.CalledProcessError as e:
    print(f"Error copying file: {e}")
# Step 2: Start the container interactively
try:
    
    subprocess.run(["docker", "start", container_name], check=True)
    print(f"Started container '{container_name}'")

    # Step 2: Execute the Python script inside the container
    subprocess.run(["docker", "exec", container_name, "python3", "run.py"], check=True)
    print("Script finished; exiting.")
    subprocess.run(["docker", "stop", container_name], check=True)
    #subprocess.run(["docker", "rm", container_name], check=True)


    #subprocess.run(["docker", "start", "-ai", container_name], check=True)
    
    #subprocess.run([ "python3", "/app/test.py"], check=True)
    #subprocess.run(["docker", "exec", container_name, "python3", "/app/test.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error starting container: {e}")
'''
