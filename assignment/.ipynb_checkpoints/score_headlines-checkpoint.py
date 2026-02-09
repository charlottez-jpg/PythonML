#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

MODEL_FILE = Path("svm.joblib")
LOCAL_EMBEDDER = Path("/opt/huggingface_models/all-MiniLM-L6-v2")
FALLBACK_EMBEDDER = "all-MiniLM-L6-v2"


# In[6]:


clf = joblib.load(MODEL_FILE)
clf


# In[7]:


embedder_id = str(LOCAL_EMBEDDER) if LOCAL_EMBEDDER.exists() else FALLBACK_EMBEDDER
embedder_id


# In[8]:


model = SentenceTransformer(embedder_id)


# In[9]:


headlines_path = Path("headlines_nyt_2024-12-02.txt")  # change if needed
headlines = [line.strip() for line in headlines_path.read_text(encoding="utf-8").splitlines() if line.strip()]
len(headlines), headlines[:5]


# In[10]:


embeddings = model.encode(headlines, convert_to_numpy=True, show_progress_bar=False)
labels = clf.predict(embeddings)

list(zip(labels[:10], headlines[:10]))


# In[11]:


import datetime as dt

source = "nyt"
today = dt.date.today()
out_path = Path(f"headline_scores_{source}_{today:%Y_%m_%d}.txt")

with out_path.open("w", encoding="utf-8") as f:
    for label, headline in zip(labels, headlines):
        f.write(f"{label},{headline}\n")

out_path


# In[ ]:




