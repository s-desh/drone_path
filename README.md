## Install requirements and run

Create virtual environment (only once)

1. Make sure you have `venv` installed (usually available by default if you have python, else run `pip3 install venv`)
2. Create a virtual environment for your project where all dependecies are isolated
    ```
    python3 -m venv drone-env
    source drone-env/bin/activate
    pip3 install -r requirements.txt
    ```
3. Run `source drone-env/bin/activate` to activate the environment and `deactivate` to ..

## Run

`python3 gym_env/playground.py`