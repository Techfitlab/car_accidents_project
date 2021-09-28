# Activate virtual environment
source "/home/airflow3/bin/activate"
DIR="/Users/greengodfitness/Desktop/Vodafone"
LOGFILE="$DIR/log"

# Log Details
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "$TIMESTAMP - Starting BASH script ->" >> $LOGFILE/bash.log
echo "$TIMESTAMP - Starting BASH script ->"

# Run the script
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "$TIMESTAMP - Running Python Service (ETL) ->" >> $LOGFILE/bash.log
echo "$TIMESTAMP - Running Python Service (ETL) ->"

echo " "
echo " "

cd "$DIR"
/home/airflow3/bin/python $DIR/src/train.py

# Log Details
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "$TIMESTAMP - Finishing BASH script..." >> $LOGFILE/bash.log
echo "$TIMESTAMP - Finishing BASH script..."