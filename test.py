import numpy as np
from rank import RANKSVM
from DataModel import DataModel

dm = DataModel()
rsvm = RANKSVM(dm,"WAY-FR")
rsvm.train()
