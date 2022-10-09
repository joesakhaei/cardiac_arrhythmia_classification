FROM ruslanmv/pyspark-jupyterlab:3.1.2

ADD train.py inference.py commands.sh ../Data_Analytics.ipynb source/arrhythmia.data /usr/local/

CMD sh /usr/local/commands.sh
