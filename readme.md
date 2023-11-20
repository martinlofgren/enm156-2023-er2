# ENM156 2023 group 12 - Gender classification of bees (ER2)

- Download data from https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip\
and unzip it to ./data/

- Add photos to ./data/Tests/ for testing

- run pip install -r requirements.txt

- py src/train.py to train model

- py src/eval.py to test it

- If you have a gpu that can run CUDA and you notice cpu being used instead: pip uninstall pytorch and install with:\
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121