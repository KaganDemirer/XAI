from nnsight import LanguageModel
from typing import List, Callable
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import clear_output

clear_output()
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
print(model)