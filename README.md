## Overview
This code is for the paper _Revisiting Demonstration Selection Strategies in In-Context Learning_. Our code is based on the <a href="https://github.com/Shark-NLP/OpenICL/tree/main">OpenICL repository</a>.

## Installation
Note: OpenICL requires Python 3.8+


**Installation for local development:**
```
git clone https://github.com/Romainpkq/revisit_demon_selection_in_ICL.git

cd revisit_demon_selection_in_ICL
pip install -e .
```

## Examples
Following example shows you how to perform ICL on sentiment classification dataset.  More examples and tutorials can be found at [examples](https://github.com/Shark-NLP/OpenICL/tree/main/examples)
```python
# predict
accelerate launch --multi_gpu --num_processes {cuda_num} exp/run.py

# calculate the accuracy
python prediction.py
```
