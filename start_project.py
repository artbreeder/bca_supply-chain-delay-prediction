import subprocess
import sys

def run(cmd):
    print(f"> {cmd}")
    subprocess.check_call(cmd, shell=True)

run(f"{sys.executable} -m pip install -r requirements.txt")
run(f"{sys.executable} main2.py")
run("streamlit run app.py")
