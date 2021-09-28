<h2>Project Details: </h2>
<ul>
<li> artifacts: Models and Experiments are saved in this location</li>
<li> bin: Run script for executing the python files in the bash operator in Airflow </li>
<li> data: All the data files used in training the model, usually these files are saved in the SQL database and imported in the python code rather than stored in the folder </li>
<li> log: log files are saved in the folder to track if the scripts executed successfully or not </li>
<li> src: scripts (project.ipynb - research script, train.py - model training and model.py - predict probability) </li>
<li> www: All the contents (i.e. Model Images and artifacts) related to the model run are saved on this location </li>
<li> mlflow.db: SQLite DB to track all the Metrics of different models and registered model's meta data is saved in this database </li>
</ul>
