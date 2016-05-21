import numpy as np
from LinearClassifier import LinearClassifier
from DataModel import DataModel

dm = DataModel()
lc = LinearClassifier(dm,"WAY-FR")
lc.train()
print lc.test()
