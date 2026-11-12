# My projects (long write-up)

## Project 1: QR Attendance System

The QR Attendance System is a web application that uses QR codes to record attendance. The idea is to make taking attendance faster and simpler than calling out names or passing around a paper sheet. I started it as a small web app and I am now expanding it into a multi-file architecture, which means I am splitting the code into separate files for different responsibilities so that the project is cleaner and easier to extend.

Working on the QR Attendance System taught me that a project that starts in one file becomes hard to manage as features are added. Separating the code into several files makes it easier to find bugs and add new parts.

## Project 2: My Digital Clone

My digital clone is a chatbot that answers questions as me. It is an AI clone, not the real me, and it says so clearly on the page. It is built with Streamlit and uses a Retrieval Augmented Generation (RAG) pipeline over documents I wrote about myself.

How it works, step by step:
1. My documents are loaded and split into overlapping chunks of about 250 words. Each chunk keeps its source file name and chunk ID.
2. The chunks are indexed in two ways: a BM25 keyword index, and sentence embeddings made with the all-MiniLM-L6-v2 model. The embeddings are computed once and saved in a .npy file.
3. When someone asks a question, the app can rank chunks with BM25 only, dense search only, or a hybrid of both.
4. The hybrid mode merges the two rankings with Reciprocal Rank Fusion (RRF), where each chunk gets a score of 1 divided by (60 plus its rank) from each list.
5. The best chunks and the question go to a language model through the Groq API, with a persona prompt that makes it answer in the first person, only from the context, and say "I don't have that in my notes" when the answer is missing.
6. The app shows the answer and also the retrieved chunks with their sources and scores.

### A problem I faced
At first the app called Hugging Face models for the answer, but I got a 402 Payment Required error because the free inference credits were used up, and some models were not available from the enabled providers. The retrieval part kept working and showed the most relevant note, but the answer was not generated. I fixed this by switching the language model part to the Groq API, which has a free tier, and by keeping the API key in secrets instead of in the code.

### Where my code lives
The code for my digital clone is on my GitHub in a repository called my-digital-clone, and the app is deployed so anyone with the link can chat with it.
