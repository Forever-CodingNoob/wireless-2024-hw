# HW3 README
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
2. Right after running the program, the initial 61-cell map is shown, on which the only mobile device at $(250, 0)$ is plotted. Note that the color of each cell corresponds to its cell ID. This map is saved as `q1_initial_map.png`.
3. Close the window showing the cell map to start the simulation.
4. When the simulation ends, the total number of handoff events that have occured in the simulation should be printed to the terminal, while the list of all handoff events is saved in as `q1_handoffs.csv`, each column of which follows the specified format: `<time>,<source_cell_ID>,<destination_cell_ID>`.
5. Meanwhile, the cell map at the end of the simulation is shown and saved as `q1_final_map.png`. This map displays the final geographic locations of all base stations, cells, and mobile devices. Note that there might be some cells in this map that are not plotted in the initial map. This is because the cell map will be extended (by plotting previously unplotted cells) during the simulation whenever a mobile device goes too far away from the central 19 cells and reaches the edge of the ploted cells. 



### Bonus
1. Execute
    ```bash
    python3 main.py 2
    ```
2. Right after running the program, the initial 61-cell map is shown, on which the 100 mobile devices uniformly distributed in the central 19 cells are plotted. Note that the color of each cell corresponds to its cell ID and that the ID assigned to each cell is labeled beside the base station (which is makred as `x`) in the cell. This map is saved as `bonus_initial_map.png`.
3. Close the window showing the cell map to start the simulation.
4. When the simulation ends, the total number of handoff events that have occured in the simulation should be printed to the terminal, while the list of all handoff events is saved in as `bonus_handoffs.csv`, each column of which follows the specified format: `<time>,<source_cell_ID>,<destination_cell_ID>`.
5. Meanwhile, the cell map at the end of the simulation is shown and saved as `bonus_final_map.png`. This map displays the final geographic locations of all base stations, cells, and mobile devices. Note that there might be some cells in this map that are not plotted in the initial map. This is because the cell map will be extended (by plotting previously unplotted cells) during the simulation whenever a mobile device goes too far away from the central 19 cells and reaches the edge of the ploted cells. 
