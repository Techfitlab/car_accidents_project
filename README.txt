Project Repo:
artifacts: Models, Metrics and Experiments
bin: Run script for executing the python files in the bash operator in Airflow
data: All the data files used in training the model, usually these files are saved in the SQL database and imported in the python code rather than stored in the folder
log: log files are saved in the folder to track if the scripts executed successfully or not
src: scripts (project.ipynb - research script, train.py - model training and model.py - predict probability)
submission: Final output for results submission is saved on this location
www: All the contents (i.e. Model Images and artifacts) related to the model run are saved on this location