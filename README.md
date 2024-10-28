# Development Operations around a Machine Learning project
*(Heavly based on the [leggedrobotics/plr-exercise](https://github.com/leggedrobotics/plr-exercise) repository)*

> [!NOTE]
> During this laboratory we will do a refiament of your previous ML project. For instance, consider the [previous mini-projects](https://home.agh.edu.pl/~mdig/dokuwiki/doku.php?id=teaching:data_science:ml_en:topics:nn_intro).

---

## Prerequisites
- an [github.com](https://github.com/) account,
- a computer with GPU or a Google Account for the [Colaboratory](https://colab.research.google.com),
- an arbitrary machine learning project,

## 1. Dependency management
Before anything, the development environment (for the Python based project) must be defined. There are two main approaches:
1. **Contenerization** - create a Dockerfile to create an image with all dependencies.
2. **Virtual environment** - which allows to separate Python packages from the OS (and between the projects).

### Contenerization
If you have an access to the local machine with admin privileges, you should go with the containers (e.g. [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) or [Podman](https://podman.io/docs/installation)).

### Virtual environment
If you are just a local user, please continue with virtual environment.
```bash
# Create folder to store virtual environments
mkdir ~/venv
# Create the virtual environment
python3 -m venv ~/venv/mldevops
# Test the virtual environment
source ~/venv/mldevops/bin/activate
which python

# Create an alias for easier sourcing (edit the ~/.bashrc file)
nano ~/.bashrc
# Add the following line at the end of the file and save it
alias venv_plr="source ~/venv/plr/bin/activate"
```

### Remote development approach
Sometimes the local machine is not prepared for the project development and there is no option to install anything new. **Typically such machine can be used as a SSH terminal for accessing the remote development environment.**

It can also be used similarly to run the webbrowser to access a remote IDE, such as a *Jupyter Lab* instance or [Google Colaboratory](https://colab.research.google.com). It must be emphasised that it is not a good practice to work excusivly with the Jupyter Notebooks - a lot of things can go wrong.

The following laboratory exercies can be accomplished using just the Google Colaboratory, but refer to this more like a situation hack than an actual *good practice*.

Remember that you will need to interact with the `git` using such commands:
```bash
!git config --global user.email "student@student.agh.edu.pl"
!git config --global user.name "student"
!git clone https://github.com/vision-agh/mldevops_exercise.git
!cd mldevops_exercise && git status
!cd mldevops_exercise && git add filename.txt
!cd mldevops_exercise && git commit -m "message"
!cd mldevops_exercise && git push
```

## 2. Working with the git repository
For starters, let's start using the git for versioning our project.
1. Create a fork of the project on GitHub.
2. Clone the repository via SSH:
```bash
mkdir ~/ws
cd ~/ws
git clone git@github.com:vision-agh/mldevops_exercise.git
```
> (Replace the `vision-agh` with your `github_username`).

> [!IMPORTANT]
> Using `https` for cloning the repo will allow only to pull the changes. For both pulling and pushing it is necessary to use the `ssh` protocol. Information how to prepare keys and add them to the GitHub account can be found [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

3. Copy your previous ML project files (`.py` and `.ipynb`) the the root of the cloned repository.
```bash
cd ~/ws/mldevops_exercise
# Copy here your project
```

4. Commit and push your changes to the origin (i.e. GitHub) repository.
```bash
# We use the dot to add all files. Note, that it is not a typical practice.
git add .
git commit -m "initial project commit"
git push
```

5. Enable `Issues` feature in your GitHub repository:
  - Open `https://github.com/GITHUB_USERNAME/mldevops_exercise`,
  - Go to the `Settings` on the right,
  - Enable the `Issues` feature (the rest can be disabled),
6. Secure your default branch (`main`/`master`) from accendtial commits:
  - Open `https://github.com/GITHUB_USERNAME/mldevops_exercise`,
  - Go to the `Settings` on the right,
  - Select `Branches` on the left tree,
  - Click on the `Add branch ruleset` button,
  - Name the ruleset as "default", set the enforcement status to "enabled", setup the Targets with the `Add target` combo list by selecting the "Include default branch" option.
  - For the options, enable: `Restrict deletions`, `Require a pull request before merging`, `Block force pushes`
  - Finish by clicking the `Create` button.

After the setup, you are ready to proceed with the exercises.

## 3. The project workflow
1. For each task, create a branch called: 'feature/task_X`,
2. Commit all changes (**and only that changes**) for the particular task to its branch and push them to the GitHub.
3. To finish a task create a pull request (PR) from `feature/task_X` to `main`. The title of the PR should be set to the task description (see below).
4. **Do not delete the branches** after mergning the PR.

Tasks:
 * **Task 1**: Improve the formatting using `black`.
 * **Task 2**: Create a python package for your project.
 * **Task 3**: Add a online logging framework
 * **Task 4**: Use optuna to perform a hyperparameter search
 * **Task 5**: Add docstrings to every file.

## Task 1
Your first task is to install black and format the code. Take a look here: https://github.com/psf/black

```bash
pip3 install black
black --line-length 120 ~/ws/mldevops_exercise
```

Now everything looks pretty.

## Task 2
> [!CAUTION]
> The `poetry` should be a better learning example in A.D. 2024.

You have to correctly create a `setup.py` Then you can install the package as follows:
```bash
cd ~/ws/mldevops_exercise
pip3 install -e ./
```
We would like the repository structure to look as follows:
```
project_name:
├──results:
│    ├──YEAR_MONTH_DAY_TIME_experiment_name:
│        ├──results.yml
│        └──....
│
├──project_name:
│    ├──models:
│    |   ├──cnn.py
│    |   └──__init__.py
│    └──__init__.py
│
├──scripts:
│    ├──train.py
│    └──timing.py
│
├──setup.py
├──.gitignore
├──README.md
```

However, we do not want to commit the files within the results folder.

Modify the `.gitignore` file and add all the files within the results to be ignored.

## Task 3
Add the *Weights & Biases* (`wandb`) logger.
```bash
pip3 install wandb
```
1. Follow the [official wandb guide](https://docs.wandb.ai/quickstart/).
2. Log the `training_loss`, `validation_loss`, and *your code* as an artifact.
3. Create a screenshot of a run with the loss curve and the uploaded artifact.
4. Commit this screenshot to the repository.

## Task 4
Use the `optuna` to find the best hyperparamers (e.g. `learning rate` or `epoch`).

```bash
pip3 install optuna
```

Use the [official examples](https://optuna.org/#code_examples) for implement the search:
```python3
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}
```

## Task 5
Add docstrings to all classes and functions: https://peps.python.org/pep-0257/

## Things we did not cover
- Typing in Python
- Linting
- Automation (GitHub Actions)
- Pre-commit

---
## Sources:
- [github.com/leggedrobotics/plr-exercise](https://github.com/leggedrobotics/plr-exercise) by @JonasFrey96
