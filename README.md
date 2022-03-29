# IACV Project


## Install Conda on Windows

https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html

## Create the environment from yml file
    conda env create -f environment.yml

## Activate conda environment
    conda activate iacv

## Deactivate conda environment
    conda deactivate

## Install the libraries from requirements.txt
    pip install -r requirements.txt

## Install a new Library with pip
    pip install library_name

## Save new installed libraries in requirements.txt
    pip freeze > requirements.txt