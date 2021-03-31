import numpy as np
import os, sys

glove_embeddings = dict()
with open(sys.argv[1], 'r', encoding='utf8') as glove_lines:
    for line in glove_lines:
        line = line.split()
        glove_embeddings[line[0]] = np.array(line[1:], dtype=np.float32)

with open('../global_data/glove/glove_data.pickle', 'wb') as handle:
    pickle.dump(glove_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
