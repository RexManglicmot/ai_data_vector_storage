## Status
This project is currently being developed and improved with additional features and testing.

## Inspiration for this project
In my path of exploring AI, majority of the content I read was based on prompt engineering, tuning, and the sort. Pure methods, to say the least. Those are cool and interesting and all, but I yearned for a different flavor. And as fate would have it, this project came to be because I watched an interesting video on YouTube on data storage by Pratik Mishra. 

Data storage was a topic that never really crossed my mind when thinking about AI/LLMs and the sort, nor while in school or self-study. Thus, I wanted to do a project that help me understand concepts of storing data and I thought this project would be a great start stepping stone into the field. 

An interesting component I learned about data storage was the financial standpoint. How much money does it really cost to store data and what are the business implications. Thus, in this project, I wanted to incorporate such. 

## Introduction

Text, images, and other data modes are often vectorized to represent their semantic meaning. These vectors are stored as floating point numbers such as float32 and float 16.  Both floats are two different foramts representing decimanl numbers in a binary matter and how many bits they use (32 and 16, respectively), which ultimately affects precision, range and storage size. For context, float32 is the default and is used for many ML suites while float16 is for GPU suties. This project focuses on benchmarking that measures how switching LLM embeddings from float32 → float16 affects many precision and efficiency in storing and financial metris. 

What makes this project so unique is that it does not invovle databases like ChromaDB and MondgoDB. This was intentional because these databasesstore the same arrays behind an index, with their uniqueness stemming from retrieval optimizations rather than raw storage format. By removing the database component, this project isolates and measures the direct impact of storage formats themselves.

## Dataset
This project focuses on synthethic data. The actual data itself doesnt matter in terms of storage because the storage results depend on the
**shape/precision** of the embedding matrix, not on document content

- **3 Corpus size tiers:** `500`, `1000`, `2000` documents  
  Embed up to the largest tier once, then slice for smaller tiers. This isolates the effect of index size on latency and cost.

- **Schema:** a single CSV with columns  
  - `id` — integer row identifier  
  - `text` — short “news-like” string  
  - `label` — one of `world`, `sports`, `business`, `sci_tech` (used only for Recall@10)

- **Queries (fixed set):**  
  We sample **100 queries** from the **smallest tier** (the first 500 documents). The same 100 queries are used for every tier so that comparisons across sizes are fair. Each query is embedded once, then searched with cosine similarity against each corpus tier.


Similarity: cocine via L2-normalized dot product...**NEED TO BRUSH UP and put materials in supplementary folder**

## Metrics

Storage: 

1) Disk Stoager (MB): on-disk size of the embedding file
2) Load time (ms): time to load/map embeddings

Latency: 

3) Query Latency (p50, p95 over 100 cosine searches): median & p95 per-query latency for 100 cosine 4 searches (L2-normalize then q @ M.T).

Quality:

4) Recall is a retreival for each query; it checks the top-10 retrieved items and mark a “hit” if **any** shares the query’s label (`world`, `sports`, `business`, `sci_tech`). Recall@10 is the fraction of queries with a hit. (0–1 proportion; e.g., 0.93 = 93%)

Scale and Cost:

5) Monthly cost USD: (size_mb / 1024) × COST_PER_GB_MONTH
6) Annual cost USD: monthly_cost_usd × 12
7) Savings vs fp32 used per month USD: (fp32_monthly − mode_monthly)

## Tech Stack
Python · Numpy · Pandas · Matplotlib · sentence-transformers (MiniLM)

## Project Architecture
```text

├─ 01_prep.py → generate synthetic corpus (id, text, label) → write:
|               • data/dataset.csv  (up to 2k docs)
|               • data/queries.csv  (100 fixed from smallest tier)
|
├─ 02a_embed_save.py → embed with MiniLM (Sentence-Transformers)
|    → save corpus/queries:
|       • data/emb_fp32.npy     (float32)
|       • data/emb_fp16.npy     (float16 cast of fp32)
|       • data/q_fp32.npy       (queries fp32)
|
├─ 02b_storage_metrics.py → measure per-mode file size + np.load time
|    → reports/storage_metrics.csv  (size_mb, load_ms, dim)
|
├─ 02cd_bench_latencies.py → for each tier n_docs ∈ {500,1000,2000}:
|    • L2-normalize matrices & queries (cosine via dot)
|    • time 100 searches → p50_ms / p95_ms
|    • compute Recall@10 (label hit in Top-10)
|    → reports/latency_both.csv
|
├─ 02e_merge_results.py → join storage + latency, add $:
|    • monthly_cost_usd, annual_cost_usd
|    • savings_vs_fp32_usd_month (per tier)
|    → reports/results.csv  (final table)
|
└─ 03_plot.py (optional) → figures:
     • size_mb vs n_docs
     • p95_ms vs n_docs
     → reports/figures/*.png
```


## Results



## Limitations


## Midnight Focus

## Next Steps
A potential next project would to focus on storage with regards to different databases such as ChromaDB and MongDB, and how they not only store but retrieve data and/or vectors. 