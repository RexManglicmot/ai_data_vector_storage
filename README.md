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
This project focuses on synthethic data. The actual data itself doesnt matter in terms of storage. 



## Metrics

Disk Stoager (MB): 
Load time (ms):
Query Latency (p50, p95 over 100 cosine searches):
Monthly cost USD:
Annual cost USD:
Savings vs fp32 used per month USD:



## Tech Stack


## Project Archetecture?



## Results



## Limitations


## Midnight Focus

## Night Steps
A potential next project would to focus on storage with regards to different databases such as ChromaDB and MongDB, and how they not only store but retrieve data and/or vectors. 