# HW4 README
## Installation
### Install [pyenv](https://github.com/pyenv/pyenv)
1. We use the automatic installer provided by pyenv here:
    ```bash
    curl https://pyenv.run | bash
    ```
2. Setup shell environment for pyenv:
    + For bash:
        ```bash
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        ```
    + For zsh:
        ```bash
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
        echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
        ```
3. Restart your shell:
    ```bash
    exec "$SHELL"
    ```
4. Install Python build dependencies:
    Follow the steps on https://github.com/pyenv/pyenv/wiki#suggested-build-environment.

### Install Python 3.10.15 using pyenv
```bash
pyenv install 3.10.15
```

### Install necessary packages
1. Change directory to the homework folder:
    ```bash
    cd /path/to/homework/submission
    ```
2. Specify the Python version to use:
    ```bash
    pyenv local 3.10
    ```
3. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Finally, install the packages in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Execute the code
### Problem 1
1. Execute
    ```bash
    python3 main.py 1
    ```
    to run the simulation program.
2. Resulting figures for all the subproblems are saved in `./1.png`.

