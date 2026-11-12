# My study notes: how RAG works

## What is RAG?
Retrieval Augmented Generation means that a language model does not answer from its own memory alone. First we retrieve passages from our own documents that are related to the question, then we give those passages to the model and ask it to answer using only that text. This reduces made up answers, because the model is told to stay inside the provided context and to admit when the answer is not there.

## The pipeline: chunk, index, retrieve, generate
- Chunk: long documents are cut into smaller pieces, usually a couple of hundred words, with a small overlap so that a sentence cut at the border still appears in both neighbouring pieces.
- Index: every chunk is stored in a form that can be searched quickly.
- Retrieve: for a question we find the chunks that are most related.
- Generate: the question and the best chunks are sent to the language model, which writes the final answer.

## BM25 keyword search
BM25 ranks chunks by the words they share with the question. Rare words count more than common words. It is very good when the question contains an exact special term, like a project name or a place. Its weakness is paraphrasing: if the question says "early riser" but the notes say "night owl", they share no words, so BM25 finds nothing.

## Dense embedding search
Dense search turns every chunk and the question into a list of numbers called an embedding, which captures the meaning of the text. Then it compares them with cosine similarity. Because it looks at meaning, it can match a paraphrased question to the right passage even when no words are shared. Its weakness is that it can miss an exact rare term, since a rare name has no strong meaning of its own.

## Hybrid search and fusion
The two methods fail in different ways, so combining them is useful. Reciprocal Rank Fusion merges two ranked lists using only the positions, not the raw scores. A chunk gets 1 divided by (60 plus its rank) from each list and the scores are added. Chunks that appear high in both lists rise to the top. The other common method is a weighted score: normalise each score with min-max scaling and mix them with a weight alpha.

## Keeping the model honest
The prompt must say: answer only from the context, speak in the first person, and say "I don't have that in my notes" when the context does not contain the answer. API keys must never be written in the code; they belong in secrets.
