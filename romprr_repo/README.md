# Information

This folder contains the configurations, logs and notebook of the visualisations of lab 1 to 2 (up to the experiments by modifying parameters).  

Since i just drag and drop my folders from my local (other) repository there will only be one commit, as it is only the "clean" result of lab 1 and 2.

## Content

`/runs` contains the logs of the **CsvLogger** from fluke. The base runs from lab 1 and the baseline run from lab 2 are all in this folder.  

`/runs/<subfolder>` are subfolders specific to the parameter tuning experiments, for example for the clients parameter the path to the experiments will be `/runs/clients/X` where ``X`` is equal to the number of clients.

This architecture is als used for the config folder which contains the .yaml configuration files for the experiments.

The notebooks in `/notebooks` are made to visualise the results of the experiments metrics wise.

The `/models` and `/dataset` contains the python files used for custom models and datasets.