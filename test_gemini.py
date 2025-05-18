import pickle
result = ['1','2']
with open("result.pkl", "wb") as f:
    pickle.dump(result, f)

import os
print(os.environ.get("GOOGLE_API_KEY"))