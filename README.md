# clam
## Clustering with Associative Memories

**Introduction:**
This is mainly the Tensorflow implementation (amc_t.py) of the Clustering with Associative Memories.

**Requirements:**
* python >= 3.9.7 
* tensorflow >= 2.4.1
* numpy >= 1.21.4
* scikit-learn >= 1.0.2

Detailed requirements for both tensorflow and pytorch are listed in requiremnt.txt file.

**Dataset Used:**
1. Zoo (101x16, 7)
2. Yale (165x1024, 15)
3. GCM (191x16063, 15)
4. Ecoli (336x7, 8)
5. Movement_libras (360x90, 15)
6. Mice Protien Expression (1080x77, 8)
7. USPS (2007x256, 10)
8. CTG (2126x21, 10)
9. Segment (2310x19, 7)
10. Fashion MNIST (60000x784, 10)

All datasets except Fashion MNIST can be found in /data directory. Fashion MNIST can be dowloaded from [here](https://github.com/zalandoresearch/fashion-mnist).


**Main file to run:**
amc_t.py (Tensorflow),

**Config File:**
hyper-params.py

**Run Command:**
python3 ./amc_t.py

**Results:**
All results are saved as json files based on dataset and configurations and can be found in /results directory.