import os, csv, argparse, re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from tqdm.autonotebook import tqdm

# To avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3
os.environ["OMP_NUM_THREADS"] = "3"

MODEL_NAME = "bert-base-cased" # Other options: "distilbert-base-uncased", "distilbert-base-cased", "bert-base-uncased", "roberta-base"
RESULTS_FILENAME = "results.csv"

MASK_MARKS = ["sensitive", "person", "dem", "loc",
                        "org", "datetime", "quantity", "misc",
                        "norp", "fac", "gpe", "product", "event",
                        "work_of_art", "law", "language", "date",
                        "time", "ordinal", "cardinal", "date_time", "datetime",
                        "nrp", "location", "organization", "\*\*\*"]


# Embedding/feature extraction
def get_all_embeddings(all_documents, remove_mask_marks=False)->list:
    # Create BERT-based model and tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True) # Whether the model returns all hidden-states.
    model.to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Collect embeddings
    all_embeddings = []
    for corpus in tqdm(all_documents):
        # Remove mask marks
        if remove_mask_marks:
            pattern = "|".join([m.upper() for m in MASK_MARKS])
            for i, text in enumerate(corpus):
                corpus[i] = re.sub(pattern, "", text)

        corpus_embeddings = np.empty((len(corpus), 768))  # 768 = BERT embedding size
        with tqdm(total=len(corpus)) as pbar:
            for i, text in enumerate(corpus):
                corpus_embeddings[i] = bert_embedding(text, model, tokenizer)
                pbar.update(1)

        all_embeddings.append(corpus_embeddings)

    return all_embeddings

def bert_embedding(texts, model, tokenizer, max_pooling=False):
  tokens = tokenizer.encode(texts, truncation=False, padding='max_length', add_special_tokens=True, return_tensors="pt")
  tokens = tokens.to(DEVICE)
  overlap_span = None

  # If longer than model max length, create multiple inputs
  len_multiplier = tokens.shape[1] / tokenizer.model_max_length
  if len_multiplier > 1:
    n_inputs = int(len_multiplier) + 1
    new_tokens = torch.empty((n_inputs, tokenizer.model_max_length), device=DEVICE, dtype=int)

    ini = 0
    for i in range(n_inputs):
      end = ini + tokenizer.model_max_length
      if end >= tokens.shape[1]:  # Last block
        overlap_span = (tokens.shape[1] - tokenizer.model_max_length, ini) # Span that will be processed twice
        end = tokens.shape[1]
        ini = end - tokenizer.model_max_length
      new_tokens[i, :] = tokens[0, ini:end]
    tokens = new_tokens

  # Predict
  with torch.no_grad():
      outputs = model(tokens)
      outputs = outputs[0].cpu()

  # Get embedding
  outputs = outputs.reshape((-1, outputs.shape[-1]))
  if overlap_span is not None:  # Remove overlap from last block
    idxs = list(range(len(outputs)))
    idxs = idxs[:overlap_span[0]] + idxs[overlap_span[1]:]
    outputs = outputs[idxs]

  # Apply max pooling (https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00564-9) or mean pooling
  if max_pooling:
    embeddings = outputs.max(axis=0)
  else:
    embeddings = outputs.mean(axis=0)

  return embeddings


# Clustering
def multi_clustering_eval(all_embeddings, k=None, n_clusterings=5, tries_per_clustering=50):
  results = np.empty((n_clusterings, len(all_embeddings)))
  for i in range(n_clusterings):
    true_labels, all_labels = get_all_clusterings(all_embeddings, k=k, tries=tries_per_clustering)
    results[i, :] = compare_clusterings(true_labels, all_labels, normalized_mutual_info_score)

  # Average per n_clusterings
  results = results.mean(axis=0)

  return results, all_labels

def get_all_clusterings(all_embeddings, k=None, tries=50):
    all_labels = []

    true_labels, inertia = clusterize(all_embeddings[0], k, tries=tries) # First used as groundtruth

    for embeddings in tqdm(all_embeddings):
        labels, inertia = clusterize(embeddings, k, tries=tries) # Repeating for the first allows to check the consistency of the groundtruth
        all_labels.append(labels)

    return true_labels, all_labels

def clusterize(embeddings, k, tries=50):
    inertia = 0
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=tries)
    labels = kmeans.fit_predict(embeddings)  # WikiActors and Wiki553
    inertia = kmeans.inertia_
    # Wiki553 bad manual | labels = DBSCAN(eps=1.1, min_samples=5, algorithm='kd_tree', metric='euclidean').fit(embeddings).labels_
    #labels = DBSCAN(eps=1.5, min_samples=5, algorithm='kd_tree', metric='euclidean').fit(embeddings).labels_ # Wiki533?
    #labels = OPTICS(min_samples=0.1, max_eps=1.25).fit(embeddings).labels_
    return labels, inertia

def compare_clusterings(true_labels, all_labels, eval_metric):
    metrics = []
    for labels in all_labels:
        metric = eval_metric(labels, true_labels)
        metrics.append(metric)
    return np.array(metrics)


# Result storage
def results_to_file(k, NMI_metrics, docs_columns, filename=RESULTS_FILENAME):
    base_title = f"{MODEL_NAME}|K={k}"

    with open(filename, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow((base_title, "NMI"))
        for i, column_name in enumerate(docs_columns):
            title = base_title + f"|{column_name}"
            writer.writerow((f"{title}", NMI_metrics[i]))


if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Computes evaluation metrics for text sanitization')
    parser.add_argument('anonymizations_df_filepath', type=str, 
                        help='path to the dataframe containing the anonymizations to evaluate')
    parser.add_argument('individual_column', type=str, nargs='?', default="doc_id",
                        help='name of the dataframe column containing the individual name')
    parser.add_argument('column_to_remove', type=str, nargs='?', default=None,
                        help='name of the dataframe column to neglect/remove from the process')
    args = parser.parse_args()

    # Read dataframe
    if args.anonymizations_df_filepath.endswith(".json"):
        data = pd.read_json(args.anonymizations_df_filepath)
    elif args.anonymizations_df_filepath.endswith(".csv"):
        data = pd.read_csv(args.anonymizations_df_filepath)
    else:
        data = pd.read_pickle(args.anonymizations_df_filepath)

    # Get data subsets
    if args.column_to_remove is not None:
        data = data.drop(columns=args.column_to_remove)
    names = data[args.individual_column]
    docs_columns = list(filter(lambda x: x!=args.individual_column, data.columns))
    print(f"Original documents column = {docs_columns[0]}") # First column must correspond to the original documents
    print(f"Anonymized documents columns = {docs_columns[1:]}") # Rest of columns correspond to the anonymized documents
    all_documents = [None]*len(docs_columns)
    for idx, column in enumerate(docs_columns):
        all_documents[idx] = list(data[column])

    # Get the embeddings
    all_embeddings = get_all_embeddings(all_documents)

    # Clustering
    k=4
    NMI_metrics, all_labels = multi_clustering_eval(all_embeddings, k=k)

    # Print results
    for elem in zip(docs_columns, NMI_metrics):
        print(elem)

    # Store results
    results_to_file(k, NMI_metrics, docs_columns)