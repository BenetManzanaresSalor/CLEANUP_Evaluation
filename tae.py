#region Imports


import json, re, abc, argparse, math, os, csv, gc
os.environ["OMP_NUM_THREADS"] = "1" # Done before loading MKL to avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from functools import partial
from collections import OrderedDict
from argparse import Namespace
from io import StringIO
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import spacy
import intervaltree

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, get_constant_schedule
from transformers import DataCollatorForLanguageModeling, pipeline, Pipeline
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg) #TODO: Change this from TRI to existing en_core_web_md?

#endregion


#region Constants


#region Input settings

# Configuration dictionary keys
CORPUS_CONFIG_KEY = "corpus_file_path"
ANONYMIZATIONS_CONFIG_KEY = "anonymizations"
RESULTS_CONFIG_KEY = "results_file_path"
METRICS_CONFIG_KEY = "metrics"
MANDATORY_CONFIG_KEYS = [CORPUS_CONFIG_KEY, ANONYMIZATIONS_CONFIG_KEY, RESULTS_CONFIG_KEY, METRICS_CONFIG_KEY]

# Corpus dictionary keys
DOC_ID_KEY = "doc_id"
TEXT_KEY = "text"
MANDATORY_CORPUS_KEYS = [DOC_ID_KEY, TEXT_KEY]
GOLD_ANNOTATIONS_KEY = "annotations"
ENTITY_MENTIONS_KEY = "entity_mentions"
ENTITY_ID_KEY = "entity_id"
START_OFFSET_KEY = "start_offset"
END_OFFSET_KEY = "end_offset"
ENTITY_TYPE_KEY = "entity_type"
IDENTIFIER_TYPE_KEY = "identifier_type"
INDENTIFIER_TYPE_DIRECT = "DIRECT"
INDENTIFIER_TYPE_QUASI = "QUASI"
INDENTIFIER_TYPE_NO_MASK = "NO_MASK"
IDENTIFIER_TYPES = [INDENTIFIER_TYPE_DIRECT, 
                          INDENTIFIER_TYPE_QUASI, 
                          INDENTIFIER_TYPE_NO_MASK]

# Metric names
PRECISION_METRIC_NAME = "Precision"
RECALL_METRIC_NAME = "Recall"
RECALL_PER_ENTITY_METRIC_NAME = "RecallPerEntity"
TPI_METRIC_NAME = "TPI"
TPS_METRIC_NAME = "TPS"
NMI_METRIC_NAME = "NMI"
TRIR_METRIC_NAME = "TRIR"
METRIC_NAMES = [PRECISION_METRIC_NAME, RECALL_METRIC_NAME, RECALL_PER_ENTITY_METRIC_NAME, TPI_METRIC_NAME, TPS_METRIC_NAME, NMI_METRIC_NAME, TRIR_METRIC_NAME]
METRICS_REQUIRING_GOLD_ANNOTATIONS = [PRECISION_METRIC_NAME, RECALL_METRIC_NAME, RECALL_PER_ENTITY_METRIC_NAME]

#endregion


#region General settings

SPACY_MODEL_NAME = "en_core_web_md"
IC_WEIGHTING_MODEL_NAME = "google-bert/bert-base-uncased"
TOKEN_WEIGHTING_KEY = "token_weighting"
IC_WEIGHTING_NAME = "IC"
IC_WEIGHTING_MAX_SEGMENT_LENGTH = 100
UNIFORM_WEIGHTING_NAME = "Uniform"
TOKEN_WEIGHTING_NAMES = [IC_WEIGHTING_NAME, UNIFORM_WEIGHTING_NAME]

# POS tags, tokens or characters that can be ignored from the recall scores 
# (because they do not carry much semantic content, and there are discrepancies
# on whether to include them in the annotated spans or not)
POS_TO_IGNORE = {"ADP", "PART", "CCONJ", "DET"} 
TOKENS_TO_IGNORE = {"mr", "mrs", "ms", "no", "nr", "about"}
CHARACTERS_TO_IGNORE = " ,.-;:/&()[]–'\" ’“”"

# Check for GPU with CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    DEVICE = torch.device("cpu")

#endregion


#region Metric-specific settings

# Precision default settings
PRECISION_TOKEN_LEVEL=True

# Recall default settings
RECALL_INCLUDE_DIRECT=True
RECALL_INCLUDE_QUASI=True
RECALL_TOKEN_LEVEL=True

# TPI default settings
TPI_TERM_ALTERNING = 6
TPI_USE_CHUNKING = True

# TPS default settings
TPS_TERM_ALTERNING = 6
TPS_USE_CHUNKING = True
TPS_SIMILARITY_MODEL_NAME = "paraphrase-albert-base-v2" # From the Sentence-Transformers library


# Default settings for NMI
NMI_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2" others from https://www.sbert.net/docs/sentence_transformer/pretrained_models.html or classic models such as "bert-base-cased" 
NMI_MIN_K = 2
NMI_MAX_K = 32
NMI_K_MULTIPLIER = 2
NMI_REMOVE_MASK_MARKS = False
NMI_MASKING_MARKS = ["SENSITIVE", "PERSON", "DEM", "LOC",
                 "ORG", "DATETIME", "QUANTITY", "MISC",
                 "NORP", "FAC", "GPE", "PRODUCT", "EVENT",
                 "WORK_OF_ART", "LAW", "LANGUAGE", "DATE",
                 "TIME", "ORDINAL", "CARDINAL", "DATE_TIME", "DATETIME",
                 "NRP", "LOCATION", "ORGANIZATION", "\*\*\*"]
NMI_N_CLUSTERINGS = 5
NMI_N_TRIES_PER_CLUSTERING = 50

#TODO: Default settings for TRIR

#endregion


#endregion


#region Classes and functions


#region Auxiliar

@dataclass
class MaskedDocument:
    """Represents a document in which some text spans are masked, each span
    being expressed by their (start, end) character boundaries"""

    doc_id: str
    masked_spans : List[Tuple[int, int]]
    replacements : List[str]

    def get_masked_offsets(self) -> set:
        """Returns the character offsets/indices that are masked"""
        if not hasattr(self, "masked_offsets"):
            self.masked_offsets = {i for start, end in self.masked_spans
                                   for i in range(start, end)}
        return self.masked_offsets
    
    def get_masked_text(self, original_text:str) -> str:
        masked_text = ""+original_text
        
        for (start_idx, end_idx), replacement in zip(reversed(self.masked_spans), reversed(self.replacements)):
            if replacement is None: # If there is no replacement, use empty string
                replacement = ""
            masked_text = masked_text[:start_idx] + replacement + masked_text[end_idx:]
        
        return masked_text

@dataclass
class MaskedCorpus(List[MaskedDocument]):
    def __init__(self, masked_docs_file_path:str):
        """Given a file path for a JSON file containing the spans to be masked for
        each document, builds a list of MaskedDocument objects"""

        masked_docs_list = []
        
        with open(masked_docs_file_path, "r", encoding="utf-8") as fd:
            masked_docs_dict = json.load(fd)
        
        if type(masked_docs_dict)!= dict:
            raise RuntimeError(f"List of MaskedDocuments in {masked_docs_file_path} must contain a dictionary mapping between document identifiers"
                                + " and lists of masked spans in this document")
        
        for doc_id, masked_spans in masked_docs_dict.items():
            doc = MaskedDocument(doc_id, [], [])
            if type(masked_spans)!=list:
                raise RuntimeError("Masked spans must be defined as [start, end, replacement] tuples (replacement is optional)")
            
            for elems in masked_spans:
                # Store span
                start = elems[0]
                end = elems[1]
                doc.masked_spans.append((start, end))

                # Store replacement (None if non-existent or it's an empty string)
                replacement = None if len(elems) < 3 or elems[2].strip() == "" else elems[2]
                doc.replacements.append(replacement)
                
            masked_docs_list.append(doc)

        # Create the class from the list
        super().__init__(masked_docs_list)

@dataclass
class AnnotatedEntity:
    """Represents an entity annotated in a document, with a unique identifier,
    a list of mentions (character-level spans in the document), whether it
    needs to be masked, and whether it corresponds to a direct identifier"""

    entity_id: str
    mentions: List[Tuple[int, int]]
    need_masking: bool
    is_direct: bool
    entity_type: str
    mention_level_masking: List[bool]

    def __post_init__(self):
        if self.is_direct and not self.need_masking:
            raise RuntimeError(f"Annotated entity {self.entity_id} is a direct identifier but it is not always masked")

    @property
    def mentions_to_mask(self) -> list:
        return [mention for i, mention in enumerate(self.mentions)
                if self.mention_level_masking[i]]

class Document:
    """Representation of a document, optionally including gold annotations"""

    doc_id:str
    text:str
    spacy_doc:spacy.tokens.Doc
    gold_annotated_entities:Dict[str, AnnotatedEntity]

    #region Initialization
    
    def __init__(self, doc_id:str, text:str, spacy_doc:Optional[spacy.tokens.Doc],
                 gold_annotations:Optional[Dict[str,List]]=None):
        """Creates a new annotated document with an identifier, a text content, and 
        (optionally) a set of gold annotations"""
        
        # The (unique) document identifier, its text and the spacy document
        self.doc_id = doc_id
        self.text = text
        self.spacy_doc = spacy_doc
        
        # Get gold annotated entities (indexed by id) if they exist
        self.gold_annotated_entities = {}
        for annotator, ann_by_person in gold_annotations.items():
            if ENTITY_MENTIONS_KEY in ann_by_person: # Optional key           
                for entity in self._get_entities_from_mentions(ann_by_person[ENTITY_MENTIONS_KEY]):                
                    if entity.entity_id in self.gold_annotated_entities: # Each entity_id is specific for each annotator
                        raise RuntimeError(f"Gold annotations of document {self.doc_id} have an entity ID repeated by multiple annotators: {entity.entity_id}")                        
                    entity.annotator = annotator
                    entity.doc_id = doc_id
                    self.gold_annotated_entities[entity.entity_id] = entity
    
    def _get_entities_from_mentions(self, entity_mentions:List[dict]) -> List[AnnotatedEntity]:
        """Returns a set of entities based on the annotated mentions"""
        entities = {}

        for mention in entity_mentions:                
            for key in [ENTITY_ID_KEY, IDENTIFIER_TYPE_KEY, START_OFFSET_KEY, END_OFFSET_KEY]:
                if key not in mention:
                    raise RuntimeError(f"Entity mention missing key {key}: {mention}")
            
            entity_id = mention[ENTITY_ID_KEY]
            start = mention[START_OFFSET_KEY]
            end = mention[END_OFFSET_KEY]
                
            if start < 0 or end > len(self.text) or start >= end:
                raise RuntimeError(f"Entity mention {entity_id} with invalid character offsets [{start}-{end}] for a text {len(self.text)} characters long")
            
            if mention[IDENTIFIER_TYPE_KEY] not in IDENTIFIER_TYPES:
                raise RuntimeError(f"Entity mention {entity_id} with unspecified or invalid identifier type: {mention['identifier_type']}")

            need_masking = mention[IDENTIFIER_TYPE_KEY] in [INDENTIFIER_TYPE_DIRECT, INDENTIFIER_TYPE_QUASI]
            is_direct = mention[IDENTIFIER_TYPE_KEY]==INDENTIFIER_TYPE_DIRECT
                
            # We check whether the entity is already defined
            if entity_id in entities:                    
                # If yes, we simply add a new mention
                current_entity = entities[entity_id]
                current_entity.mentions.append((start, end))
                current_entity.mention_level_masking.append(need_masking)
                    
            # Otherwise, we create a new entity with one single mention
            else:
                new_entity = AnnotatedEntity(entity_id, [(start, end)], need_masking, is_direct, 
                                             mention[ENTITY_TYPE_KEY], [need_masking])
                entities[entity_id] = new_entity
                
        for entity in entities.values():
            if set(entity.mention_level_masking) != {entity.need_masking}: # Solve inconsistent masking
                entity.need_masking = True
                #logging.warning(f"Entity {entity.entity_id} is inconsistently masked: {entity.mention_level_masking}")
                
        return list(entities.values())
    
    #endregion

    #region Functions

    def is_masked(self, masked_doc:MaskedDocument, entity:AnnotatedEntity) -> bool:
        """Given a document with a set of masked text spans, determines whether entity
        is fully masked (which means that all its mentions are masked)"""
        
        for incr, (mention_start, mention_end) in enumerate(entity.mentions):
            
            if self.is_mention_masked(masked_doc, mention_start, mention_end):
                continue
            
            # The masking is sometimes inconsistent for the same entity, 
            # so we verify that the mention does need masking
            elif entity.mention_level_masking[incr]:
                return False
        return True
    
    def is_mention_masked(self, masked_doc:MaskedDocument, mention_start:int, mention_end:int) -> bool:
        """Given a document with a set of masked text spans and a particular mention span,
        determine whether the mention is fully masked (taking into account special
        characters or PoS/tokens to ignore)"""
                
        # Computes the character offsets that must be masked
        offsets_to_mask = set(range(mention_start, mention_end))

        # We build the set of character offsets that are not covered
        non_covered_offsets = offsets_to_mask - masked_doc.get_masked_offsets()
            
        # If we have not covered everything, we also make sure punctuations
        # spaces, titles, etc. are ignored
        if len(non_covered_offsets) > 0:
            span = self.spacy_doc.char_span(mention_start, mention_end, alignment_mode="expand")
            for token in span:
                if token.pos_ in POS_TO_IGNORE or token.lower_ in TOKENS_TO_IGNORE:
                    non_covered_offsets -= set(range(token.idx, token.idx+len(token)))
        for i in list(non_covered_offsets):
            if self.text[i] in set(CHARACTERS_TO_IGNORE):
                non_covered_offsets.remove(i)

        # If that set is empty, we consider the mention as properly masked
        return len(non_covered_offsets) == 0

    def get_entities_to_mask(self, include_direct:bool=True, include_quasi:bool=True) -> list:
        """Return entities that should be masked, and satisfy the constraints 
        specified as arguments"""
        
        to_mask = []
        for entity in self.gold_annotated_entities.values():
            # We only consider entities that need masking and are the right type
            if not entity.need_masking:
                continue
            elif entity.is_direct and not include_direct:
                continue
            elif not entity.is_direct and not include_quasi:
                continue  
            to_mask.append(entity)
                
        return to_mask      
        
    def get_annotators_for_span(self, start_token:int, end_token:int) -> set:
        """Given a text span (typically for a token), determines which annotators 
        have also decided to mask it. Concretely, the method returns a (possibly
        empty) set of annotators names that have masked that span."""        
        
        # We compute an interval tree for fast retrieval
        if not hasattr(self, "masked_spans"):
            self.masked_spans = intervaltree.IntervalTree()
            for entity in self.gold_annotated_entities.values():
                if entity.need_masking:
                    for i, (start, end) in enumerate(entity.mentions):
                        if entity.mention_level_masking[i]:
                            self.masked_spans[start:end] = entity.annotator
        
        annotators = set()      
        for mention_start, mention_end, annotator in self.masked_spans[start_token:end_token]:
            
            # We require that the span is fully covered by the annotator
            if mention_start <=start_token and mention_end >= end_token:
                annotators.add(annotator)
                    
        return annotators

    def split_by_tokens(self, start:int, end:int):
        """Generates the (start, end) boundaries of each token included in this span"""
        
        for match in re.finditer(r"\w+", self.text[start:end]):
            start_token = start + match.start(0)
            end_token = start + match.end(0)
            yield start_token, end_token

    #endregion

class TokenWeighting:
    """Abstract class for token weighting schemes"""

    @abc.abstractmethod
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        """Given a text and a list of text spans, returns a NumPy array of numeric weights
        (of same length as the list of spans) representing the information content
        conveyed by each span.

        A weight close to 0 represents a span with low information content (i.e. which
        can be easily predicted from the remaining context), while a higher weight 
        represents a high information content (which is difficult to predict from the
        context)"""

        return
    
    def __del__(self):
        pass

class ICTokenWeighting(TokenWeighting):
    """Token weighting based on a BERT language model. The weighting mechanism
    runs the BERT model on a text in which the provided spans are masked. The
    weight of each token is then defined as its information content:
    -log(probability of the actual token value).
    
    In other words, a token that is difficult to predict will have a high
    information content, and therefore a high weight, whereas a token which can
    be predicted from its content will received a low weight."""

    max_segment_size:int
    model_name:str
    device:str

    model=None
    tokenizer=None
    
    def __init__(self, max_segment_size:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH, model_name:str=IC_WEIGHTING_MODEL_NAME, device:str=DEVICE):
        """Initialises the BERT tokenizers and masked language model"""
        self.max_segment_length = max_segment_size
        self.model_name = model_name
        self.device = device
    
    # TODO: Implement a batched version
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        """Returns a list of numeric information content weights, where each value
        corresponds to -log(probability of predicting the value of the text span
        according to the BERT model).
        
        If the span corresponds to several BERT tokens, the probability is the 
        mininum of the probabilities for each token."""

        # STEP 0: Create model if it is not already created
        if self.model is None:
            self.create_model()
        
        # STEP 1: we tokenise the text
        bert_tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = bert_tokens["input_ids"]
        input_ids_copy = np.array(input_ids)
        
        # STEP 2: we record the mapping between spans and BERT tokens
        bert_token_spans = bert_tokens["offset_mapping"]
        tokens_by_span = self._get_tokens_by_span(bert_token_spans, text_spans, text)

        # STEP 3: we mask the tokens that we wish to predict
        attention_mask = bert_tokens["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokenizer.mask_token_id
          
        # STEP 4: we run the masked language model     
        logits = self._get_model_predictions(input_ids, attention_mask)
        unnorm_probs = torch.exp(logits)
        probs = unnorm_probs / torch.sum(unnorm_probs, axis=1)[:,None]
        
        # We are only interested in the probs for the actual token values
        probs_actual = probs[torch.arange(len(input_ids)), input_ids_copy]
        probs_actual = probs_actual.detach().cpu().numpy()
              
        # STEP 5: we compute the weights from those predictions
        weights = []
        for (span_start, span_end) in text_spans:
            
            # If the span does not include any actual token, skip
            if not tokens_by_span[(span_start, span_end)]:
                weights.append(0)
                continue
            
            # if the span has several tokens, we take the minimum prob
            prob = np.min([probs_actual[token_idx] for token_idx in 
                           tokens_by_span[(span_start, span_end)]])
            
            # We finally define the weight as -log(p)
            weights.append(-np.log(prob))
        
        weights = np.array(weights) # Transform to NumPy array
        
        return weights

    def create_model(self):
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _get_tokens_by_span(self, bert_token_spans, text_spans, text:str):
        """Given two lists of spans (one with the spans of the BERT tokens, and one with
        the text spans to weight), returns a dictionary where each text span is associated
        with the indices of the BERT tokens it corresponds to."""            
        
        # We create an interval tree to facilitate the mapping
        text_spans_tree = intervaltree.IntervalTree()
        for start, end in text_spans:
            text_spans_tree[start:end] = True
        
        # We create the actual mapping between spans and tokens
        tokens_by_span = {span:[] for span in text_spans}
        for token_idx, (start, end) in enumerate(bert_token_spans):
            for span_start, span_end, _ in text_spans_tree[start:end]:
                tokens_by_span[(span_start, span_end)].append(token_idx) 
        
        # And control that everything is correct
        for span_start, span_end in text_spans:
            if len(tokens_by_span[(span_start, span_end)]) == 0:
                logging.warning(f"Span ({span_start},{span_end}) without any token [{repr(text[span_start:span_end])}]")
        
        return tokens_by_span
    
    def _get_model_predictions(self, input_ids, attention_mask):
        """Given tokenised input identifiers and an associated attention mask (where the
        tokens to predict have a mask value set to 0), runs the BERT language and returns
        the (unnormalised) prediction scores for each token.
        
        If the input length is longer than max_segment size, we split the document in
        small segments, and then concatenate the model predictions for each segment."""
        
        nb_tokens = len(input_ids)
        
        input_ids = torch.tensor(input_ids)[None,:].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None,:].to(self.device)
        
        # If the number of tokens is too large, we split in segments
        if nb_tokens > self.max_segment_length:
            nb_segments = math.ceil(nb_tokens/self.max_segment_length)
            
            # Split the input_ids (and add padding if necessary)
            split_pos = [self.max_segment_length * (i + 1) for i in range(nb_segments - 1)]
            input_ids_splits = torch.tensor_split(input_ids[0], split_pos)

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_splits, batch_first=True)
            
            # Split the attention masks
            attention_mask_splits = torch.tensor_split(attention_mask[0], split_pos)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_splits, batch_first=True)
                   
        # Run the model on the tokenised inputs + attention mask
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # And get the resulting prediction scores
        scores = outputs.logits
        
        # If the batch contains several segments, concatenate the result
        if len(scores) > 1:
            scores = torch.vstack([scores[i] for i in range(len(scores))])
            scores = scores[:nb_tokens]
        else:
            scores = scores[0]
        
        return scores     

    def __del__(self):
        # Dispose model and tokenizer (if defined)
        if not self.model is None:
            del self.model
        if not self.tokenizer is None:
            del self.tokenizer
        if not gc is None:
            gc.collect()
        if not torch is None and torch.cuda.is_available():
            torch.cuda.empty_cache()

class UniformTokenWeighting(TokenWeighting):
    """Uniform weighting (all tokens assigned to a weight of 1.0)"""
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        return np.ones(len(text_spans))

#endregion


#region TAE

class TAE:
    """Text Anonymization Evaluator (TAE), defined by the corpus for text anonymization.
    Optionally, the corpus can include gold annotations, used for precision and recall metrics."""

    documents:Dict[str, Document]
    spacy_nlp=None
    gold_annotations_ratio:int
    ic_weighting:ICTokenWeighting
    uniform_weighting:UniformTokenWeighting

    #region Initialization
    
    def __init__(self, corpus:List[Document], spacy_model_name:str=SPACY_MODEL_NAME):
        # Documents indexed by identifier
        self.documents = {}

        # Loading the spaCy model
        self.spacy_nlp = spacy.load(spacy_model_name, disable=["lemmatizer"])        
        
        # Load corpus
        n_docs_with_annotations = 0
        for doc in tqdm(corpus, desc=f"Loading corpus of {len(corpus)} documents"):
            for key in MANDATORY_CORPUS_KEYS:
                if key not in doc:
                    raise RuntimeError(f"Document {doc.doc_id} missing mandatory key: {key}")
            
            # Parsing the document with spaCy
            spacy_doc = self.spacy_nlp(doc[TEXT_KEY])

            # Get gold annotations (if present)
            gold_annotations = doc.get(GOLD_ANNOTATIONS_KEY, None)
            
            # Creating the actual document (identifier, text and gold annotations)
            new_doc = Document(doc[DOC_ID_KEY], doc[TEXT_KEY], spacy_doc, gold_annotations)
            self.documents[doc[DOC_ID_KEY]] = new_doc
            if len(new_doc.gold_annotated_entities) > 0:
                n_docs_with_annotations += 1
        
        # Notify the number and percentage of annotated documents
        self.gold_annotations_ratio = n_docs_with_annotations / len(self.documents)
        logging.info(f"Number of gold annotated documents: {n_docs_with_annotations} ({self.gold_annotations_ratio:.2%})")

        # Create both token weighting types
        self.ic_weighting = ICTokenWeighting()
        self.uniform_weighting = UniformTokenWeighting()

    @classmethod
    def from_file_path(cls, corpus_file_path:str):
        with open(corpus_file_path, encoding="utf-8") as f:
            corpus = json.load(f)
        if type(corpus)!=list:
            raise RuntimeError("Corpus JSON file must be a list of documents")
        return TAE(corpus)

    #endregion


    #region Evaluation

    def evaluate(self, anonymizations:Dict[str, List[MaskedDocument]], metrics:dict, results_file_path:Optional[str]=None) -> dict:
        results = {}

        # Initial checks
        self._eval_checks(anonymizations, metrics)

        # Write results file header
        if results_file_path:
            self._write_into_results(results_file_path, ["Metric/Anonymization"]+list(anonymizations.keys()))

        # For each metric
        for metric_name, metric_parameters in metrics.items():
            logging.info(f"############# Computing {metric_name} metric #############")
            metric_key = metric_name.split("_")[0] # Text before first underscore is name of the metric, the rest is freely used
            partial_eval_func = self._get_partial_eval_func(metric_key, metric_parameters)

            # If metric is invalid, results are None
            if partial_eval_func is None:      
                metric_results = {anon_name:None for anon_name in anonymizations.keys()}

            # For NMI and TRIR, evaluate all anonymizations at once
            elif partial_eval_func.func==self.get_NMI or partial_eval_func.func==self.get_TRIR:
                output = partial_eval_func(anonymizations)
                metric_results = output[0] if isinstance(output, tuple) else output # If tuple, the first is metric_results
            
            # Otherwise, compute metric for each anonymization
            else:
                metric_results = {}
                ICs_dict = None # ICs cache for TPI and TPS
                with tqdm(anonymizations.items(), desc="Processing each anonymization") as pbar:
                    for anon_name, masked_docs in pbar:
                        pbar.set_description(f"Processing {anon_name} anonymization")

                        # For TPI and TPS, cache ICs
                        if partial_eval_func.func==self.get_TPI or partial_eval_func.func==self.get_TPS:
                            output = partial_eval_func(masked_docs, ICs_dict=ICs_dict)
                            ICs_dict = output[2]
                        # Otherwise, normal computation
                        else:
                            output = partial_eval_func(masked_docs)
                        
                        metric_results[anon_name] = output[0] if isinstance(output, tuple) else output  # If tuple, the first is metric's value
                        logging.info(f"{metric_name} for {anon_name}: {metric_results[anon_name]}")
            
            # Save results
            results[metric_name] = metric_results
            if not results_file_path is None: #TODO: CSV is maybe a bad format for complex results such as those from recall_per_entity_type. Is JSON better instead (worse for Excel)
                self._write_into_results(results_file_path, [metric_name]+list(metric_results.values()))
            
            # Show results all together for easy comparison
            msg = f"Results for {metric_name}:"
            for name, value in results[metric_name].items():
                msg += f"\n\t{name}: {value}"
            logging.info(msg)
        
        return results

    def _eval_checks(self, anonymizations:Dict[str, List[MaskedDocument]], metrics:dict):
        # Check each anonymization has a masked version of all the documents in the corpus
        for anon_name, masked_docs in anonymizations.items():
            corpus_doc_ids = set(self.documents.keys())
            for masked_doc in masked_docs:
                if masked_doc.doc_id in corpus_doc_ids:
                    corpus_doc_ids.remove(masked_doc.doc_id)
                else:
                    logging.warning(f"Anonymization {anon_name} includes a masked document (ID={masked_doc.doc_id}) not present in the corpus")
            if len(corpus_doc_ids) > 0:
                raise RuntimeError(f"Anonymization {anon_name} misses masked documents for the following {len(corpus_doc_ids)} ID/s: {corpus_doc_ids}")
        
        # Check all metrics are valid and can be computed
        for name, parameters in metrics.items():
            metric_key = name.split("_")[0]
            if not metric_key in METRIC_NAMES:
                logging.warning(f"Metric {metric_key} (from {name}) is unknown, therefore its results will be None | Options: {METRIC_NAMES}") #TODO: Maybe this can become an logging.error after testing
            elif name in METRICS_REQUIRING_GOLD_ANNOTATIONS and self.gold_annotations_ratio < 1:
                raise RuntimeError(f"Metric {name} depends on gold annotations, but these are not present for all documents (only for a {self.gold_annotations_ratio:.2%})")
            elif TOKEN_WEIGHTING_KEY in parameters and not parameters[TOKEN_WEIGHTING_KEY] in TOKEN_WEIGHTING_NAMES:
                raise RuntimeError(f"Metric {name} has an invalid {TOKEN_WEIGHTING_KEY} ({parameters[TOKEN_WEIGHTING_KEY]}) | Options: {TOKEN_WEIGHTING_NAMES}")

    def _get_partial_eval_func(self, name:str, parameters:dict) -> Optional[partial]:
        partial_func = None # Result would be None if name is invalid

        # Replace token weighting string sentence by the proper instance of TokenWeighting
        if TOKEN_WEIGHTING_KEY in parameters:
            if parameters[TOKEN_WEIGHTING_KEY]==IC_WEIGHTING_NAME:
                parameters[TOKEN_WEIGHTING_KEY] = self.ic_weighting
            elif parameters[TOKEN_WEIGHTING_KEY]==UNIFORM_WEIGHTING_NAME:
                parameters[TOKEN_WEIGHTING_KEY] = self.uniform_weighting
        
        # Get the partial function
        if name == PRECISION_METRIC_NAME:
            partial_func = partial(self.get_precision, **parameters)
        elif name == RECALL_METRIC_NAME:
            partial_func = partial(self.get_recall, **parameters)
        elif name == RECALL_PER_ENTITY_METRIC_NAME:
            partial_func = partial(self.get_recall_per_entity_type, **parameters)
        elif name == TPI_METRIC_NAME:
            partial_func = partial(self.get_TPI, **parameters)
        elif name == TPS_METRIC_NAME:
            partial_func = partial(self.get_TPS, **parameters)
        elif name == NMI_METRIC_NAME:
            partial_func = partial(self.get_NMI, **parameters)
        elif name == TRIR_METRIC_NAME:
            partial_func = partial(self.get_TRIR, **parameters)

        return partial_func

    def _write_into_results(self, results_file_path:str, values:list):
        # Create containing directory if it does not exist
        directory = os.path.dirname(results_file_path)
        if directory and not os.path.exists(directory): # If it does not exist
            os.makedirs(directory, exist_ok=True) # Create directory (including intermediate ones)

        # Store the row of results
        with open(results_file_path, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([datetime_str]+values)
    
    #endregion


    #region Metrics


    #region Precision
            
    def get_precision(self, masked_docs:List[MaskedDocument], token_weighting:Optional[TokenWeighting]=None, 
                      token_level:bool=PRECISION_TOKEN_LEVEL) -> float:
        """Returns the weighted, token-level precision of the masked spans when compared 
        to the gold annotations. Arguments:
        - masked_docs: documents together with spans masked by the system
        - token_weighting: mechanism for weighting the information content of each token
        
        If token_level is set to true, the precision is computed at the level of tokens, 
        otherwise the precision is at the mention-level. The masked spans/tokens are weighted 
        by their information content, given the provided weighting scheme. If annotations from 
        several annotators are available for a given document, the precision corresponds to a 
        micro-average over the annotators."""
        
        weighted_true_positives = 0.0
        weighted_system_masks = 0.0

        # Default token_weighting is Uniform
        if token_weighting is None:
            token_weighting = self.uniform_weighting
                
        for doc in masked_docs:
            gold_doc = self.documents[doc.doc_id]
            
            # We extract the list of spans (token- or mention-level)
            system_masks = []
            for start, end in doc.masked_spans:
                if token_level:
                    system_masks += list(gold_doc.split_by_tokens(start, end))
                else:
                    system_masks += [(start,end)]
            
            # We compute the weights (information content) of each mask
            weights = token_weighting.get_weights(gold_doc.text, system_masks)
            
            # We store the number of annotators in the gold standard document
            nb_annotators = len(set(entity.annotator for entity in gold_doc.gold_annotated_entities.values()))
            
            for (start, end), weight in zip(system_masks, weights):
                
                # We extract the annotators that have also masked this token/span
                annotators = gold_doc.get_annotators_for_span(start, end)
                
                # And update the (weighted) counts
                weighted_true_positives += (len(annotators) * weight)
                weighted_system_masks += (nb_annotators * weight)
        try:
            return weighted_true_positives / weighted_system_masks
        except ZeroDivisionError:
            return 0

    #endregion
    

    #region Recall

    def get_recall(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                    include_quasi:bool=RECALL_INCLUDE_QUASI, token_level:bool=RECALL_TOKEN_LEVEL) -> float:
        """Returns the mention or token-level recall of the masked spans when compared 
        to the gold annotations. 
        
        Arguments:
        - masked_docs: documents together with spans masked by the system
        - include_direct: whether to include direct identifiers in the metric
        - include_quasi: whether to include quasi identifiers in the metric
        - token_level: whether to compute the recall at the level of tokens or mentions
                
        If annotations from several annotators are available for a given document, the recall 
        corresponds to a micro-average over the annotators."""

        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        nb_masked_elements = sum(nb_masked_by_type.values())
        nb_elements = sum(nb_by_type.values())
                
        try:
            return nb_masked_elements / nb_elements
        except ZeroDivisionError:
            return 0

    #TODO: There is no precision per entity type?
    def get_recall_per_entity_type(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                                   include_quasi:bool=RECALL_INCLUDE_QUASI, token_level:bool=RECALL_TOKEN_LEVEL) -> dict:
        """Returns the mention or token-level recall of the masked spans when compared 
        to the gold annotations, and factored by entity type. 
        
        Arguments:
        - masked_docs: documents together with spans masked by the system
        - include_direct: whether to include direct identifiers in the metric
        - include_quasi: whether to include quasi identifiers in the metric
        - token_level: whether to compute the recall at the level of tokens or mentions
                
        If annotations from several annotators are available for a given document, the recall 
        corresponds to a micro-average over the annotators."""
        
        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        return {ent_type:nb_masked_by_type[ent_type]/nb_by_type[ent_type]
                for ent_type in nb_by_type}
                
    def _get_mask_counts(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                                   include_quasi:bool=RECALL_INCLUDE_QUASI, token_level:bool=RECALL_TOKEN_LEVEL) -> Tuple[dict, dict]:
        
        nb_masked_elements_by_type = {}
        nb_elements_by_type = {}
        
        for doc in masked_docs:            
            gold_doc = self.documents[doc.doc_id]           
            for entity in gold_doc.get_entities_to_mask(include_direct, include_quasi):
                
                if entity.entity_type not in nb_elements_by_type:
                    nb_elements_by_type[entity.entity_type] = 0
                    nb_masked_elements_by_type[entity.entity_type] = 0         
                
                spans = list(entity.mentions)
                if token_level:
                    spans = [(start, end) for mention_start, mention_end in spans
                             for start, end in gold_doc.split_by_tokens(mention_start, mention_end)]
                
                for start, end in spans:
                    if gold_doc.is_mention_masked(doc, start, end):
                        nb_masked_elements_by_type[entity.entity_type] += 1
                    nb_elements_by_type[entity.entity_type] += 1
        
        return nb_masked_elements_by_type, nb_elements_by_type

    #endregion
    

    #region TPS and TPI
    
    def get_TPI(self, masked_docs:List[MaskedDocument], term_alterning=TPI_TERM_ALTERNING,
            use_chunking:bool=TPI_USE_CHUNKING, token_weighting:Optional[TokenWeighting]=None, 
            ICs_dict:Optional[Dict[str,np.ndarray]]=None) -> Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
        tpi_array = np.empty(len(masked_docs))
        if ICs_dict is None:
            ICs_dict = {}
        IC_multiplier_array = np.empty(len(masked_docs))

        # Default token_weighting is IC-based
        if token_weighting is None:
            token_weighting = self.ic_weighting

        for i, masked_doc in enumerate(masked_docs):
            doc = self.documents[masked_doc.doc_id]

            # Get terms spans and mask
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)
            masked_spans = self._filter_masked_spans(doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, masked_spans) # Non-masked=True(1), Masked=False(0)

            # Get IC for all spans
            if masked_doc.doc_id in ICs_dict:
                spans_IC = ICs_dict[masked_doc.doc_id]
            else:
                spans_IC = self._get_ICs(spans, doc, term_alterning, token_weighting)
                ICs_dict[masked_doc.doc_id] = spans_IC # Store ICs (useful as cache)
            
            # Get TIC of the original and masked documents
            original_TIC = spans_IC.sum()
            masked_TIC = spans_IC[spans_mask].sum()

            # Compute document TPI
            tpi_array[i] = masked_TIC / original_TIC if original_TIC != 0 else 0

            # Compute document IC multiplier
            n_terms = len(spans)
            n_masked_terms = np.count_nonzero(spans_mask==0)
            info_loss = original_TIC - masked_TIC
            masked_term_IC = info_loss / n_masked_terms if n_masked_terms != 0 else 0
            n_nonmasked_terms = n_terms - n_masked_terms
            nonmasked_term_IC = masked_TIC / n_nonmasked_terms if n_nonmasked_terms != 0 else 0
            IC_multiplier_array[i] = masked_term_IC / nonmasked_term_IC if nonmasked_term_IC != 0 else 0

        # Get corpus TPI as the mean
        tpi = tpi_array.mean()

        return tpi, tpi_array, ICs_dict, IC_multiplier_array

    def get_TPS(self, masked_docs:List[MaskedDocument], term_alterning=TPS_TERM_ALTERNING,
                similarity_model_name:str=TPS_SIMILARITY_MODEL_NAME,
                use_chunking:bool=TPS_USE_CHUNKING, token_weighting:Optional[TokenWeighting]=None,
                ICs_dict:Optional[Dict[str,np.ndarray]]=None) -> Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
        tps_array = np.empty(len(masked_docs))
        if ICs_dict is None:
            ICs_dict = {}
        similarity_array = []
        
        # Load embedding model and function for similarity
        embedding_func, embedding_model = self._get_embedding_func(similarity_model_name)

        # Default token_weighting is IC-based
        if token_weighting is None:
            token_weighting = self.ic_weighting
        
        # Process each masked document
        for idx, masked_doc in enumerate(masked_docs):
            doc = self.documents[masked_doc.doc_id]

            # Get text spans
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)

            # Get IC for all spans
            if masked_doc.doc_id in ICs_dict:
                spans_IC = ICs_dict[masked_doc.doc_id]
            else:
                spans_IC = self._get_ICs(spans, doc, term_alterning, token_weighting)
                ICs_dict[masked_doc.doc_id] = spans_IC # Store ICs (useful as cache)

            # Get replacements, corresponding masked texts and corresponding spans indexes
            repl_out = self._get_replacements_info(masked_doc, doc, spans)
            (replacements, masked_texts, spans_idxs_per_replacement) = repl_out

            # Measure similarities of replacements
            masked_spans = self._filter_masked_spans(doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, masked_spans) # Non-masked=True(1), Masked=False(0)
            spans_sims = np.array(spans_mask, dtype=float) # Similarities for terms: Non-masked=1, Supressed=0, Replaced=[0,1]
            if len(replacements) > 0:
                texts_to_embed = masked_texts + replacements
                embeddings = embedding_func(texts_to_embed)
      
                masked_embedds = embeddings[:len(masked_texts)]
                repl_embedds = embeddings[len(masked_texts):]
                for masked_embed, repl_embed, spans_idxs in zip(masked_embedds, repl_embedds, spans_idxs_per_replacement):
                    similarity = self._cos_sim(masked_embed, repl_embed)
                    spans_sims[spans_idxs] = similarity
                    similarity_array.append(similarity)
                
                # Limit similarities to range [0,1]
                spans_sims[spans_sims < 0] = 0
                spans_sims[spans_sims > 1] = 1

            # Get TPS
            masked_TIC_sim = (spans_IC * spans_sims).sum()
            original_TIC = spans_IC.sum()
            tps_array[idx] = masked_TIC_sim / original_TIC
        
        # Dispose embedding model
        if not embedding_model is None:
            del embedding_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()                

        # Get mean TPS
        tps = tps_array.mean()

        # All similarities to NumPy array
        similarity_array = np.array(similarity_array)

        return tps, tps_array, ICs_dict, similarity_array

    def _get_terms_spans(self, spacy_doc:spacy.tokens.Doc, use_chunking:bool=True) -> List[Tuple[int, int]]:
        text_spans = []
        added_tokens = np.zeros(len(spacy_doc), dtype=bool)

        if use_chunking:
            for chunk in spacy_doc.ents:
                start = spacy_doc[chunk.start].idx
                last_token = spacy_doc[chunk.end - 1]
                end = last_token.idx + len(last_token)
                text_spans.append((start, end))
                added_tokens[chunk.start:chunk.end] = True

            for chunk in spacy_doc.noun_chunks:
                # If is it not already added
                if not added_tokens[chunk.start:chunk.end].any():
                    start = spacy_doc[chunk.start].idx
                    last_token = spacy_doc[chunk.end - 1]
                    end = last_token.idx + len(last_token)
                    text_spans.append((start, end))
                    added_tokens[chunk.start:chunk.end] = True
                

        # Add text spans after last chunk (or all spans, if chunks are ignored)
        for token_idx in range(len(spacy_doc)):
            if not added_tokens[token_idx]:
                token = spacy_doc[token_idx]            
                if token.text.strip() not in ["", "\n"]:  # Avoiding empty spans
                    start = token.idx
                    end = start + len(token)
                    text_spans.append((start, end))

        # Sort text spans by starting position
        text_spans = sorted(text_spans, key=lambda span: span[0], reverse=False)

        return text_spans

    def _filter_masked_spans(self, doc:Document, masked_doc:MaskedDocument) -> List[Tuple[int, int]]:
        filtered_masked_spans = []

        masking_array = np.zeros(len(doc.spacy_doc.text), dtype=bool)
        for (s, e) in masked_doc.masked_spans:
            masking_array[s:e] = True
        
        ini_current_mask = -1
        for idx, elem in enumerate(masking_array):
            # Start of mask
            if ini_current_mask == -1 and elem:
                ini_current_mask = idx
            # End of mask
            elif ini_current_mask >= 0 and not elem:
                filtered_masked_spans.append((ini_current_mask, idx))
                ini_current_mask = -1
        
        return filtered_masked_spans

    def _get_spans_mask(self, spans:List[Tuple[int, int]], masked_spans:List[Tuple[int, int]]) -> np.ndarray:
        spans_mask = np.empty(len(spans), dtype=bool)
        sorted_masked_spans = sorted(masked_spans, key=lambda span: span[0], reverse=False)

        for i, (span_start, span_end) in enumerate(spans):
            # True(1)=Non-masked, False(0)=Masked
            spans_mask[i] = True
            for (masked_span_start, masked_span_end) in sorted_masked_spans:
                if span_start <= masked_span_start < span_end or span_start < masked_span_end <= span_end:
                    spans_mask[i] = False
                elif masked_span_start > span_end: # Break if masked span starts too late
                    break

        return spans_mask

    def _get_ICs(self, spans:List[Tuple[int, int]], doc:Document, term_alterning:int, token_weighting:TokenWeighting) -> np.ndarray:
        spans_IC = np.empty(len(spans))
        if isinstance(term_alterning, int) and term_alterning > 1: # N-Term Alterning
            # Get ICs by masking each N term at a time, with all the document as context
            for i in range(term_alterning):
                spans_for_IC = spans[i::term_alterning]
                spans_IC[i::term_alterning] = self._get_spans_ICs(spans_for_IC, doc, token_weighting)
        
        elif isinstance(term_alterning, str) and term_alterning.lower() == "sentence": # Sentence-Term Alterning
            # Get ICs by masking 1 term of each sentence at a time, with the sentence as context
            # Get sentences spans
            sentences_spans = [[sent.start_char, sent.end_char] for sent in doc.spacy_doc.sents]
            # Iterate sentences
            ini_span_idx = 0
            for sentence_span in sentences_spans:
                sentence_start, sentence_end = sentence_span
                # Get spans in the sentence
                span_idx = ini_span_idx
                first_sentence_span_idx = -1
                is_sentence_complete = False
                while span_idx < len(spans) and not is_sentence_complete:
                    # If span belongs to sentence (first spans may not belong to any sentence)
                    if spans[span_idx][0] >= sentence_start and spans[span_idx][1] < sentence_end:
                        if first_sentence_span_idx == -1:  # If first sentence span
                            first_sentence_span_idx = span_idx  # Store first index
                        span_idx += 1  # Go to next span
                    # If not belongs and sentence is started, sentence completed
                    elif first_sentence_span_idx != -1:
                        is_sentence_complete = True
                    # Otherwise, go to next span
                    else:
                        span_idx += 1
                spans_for_IC = spans[first_sentence_span_idx:span_idx]
                # Update initial span index for sentece's spans searching
                ini_span_idx = span_idx
                # Get IC for each span of the sentence
                for span in spans_for_IC:
                    original_info, masked_info, n_masked_terms = self._get_spans_ICs(doc, [span], doc,
                                                                                     token_weighting, context_span=sentence_span)
                    original_doc_info += original_info
                    masked_doc_info += masked_info
                    total_n_masked_terms += n_masked_terms
        else:
            raise RuntimeError(f"Term alterning {term_alterning} is invalid. It must be an integer greater than 1 or \"sentence\".")

        return spans_IC
    
    def _get_spans_ICs(self, spans: List[Tuple[int,int]], doc:Document, token_weighting: TokenWeighting, context_span:Optional[Tuple[int,int]]=None) -> np.ndarray:
        # By default, context span is all the document
        if context_span is None:
            context_span = (0, len(doc.text))

        # Get context
        context_start, context_end = context_span
        context = doc.text[context_start:context_end]

        # Adjust spans to the context
        in_context_spans = []
        for (start, end) in spans:
            in_context_spans.append((start - context_start, end - context_start))

        # Obtain the weights (Information Content) of each word
        ICs = token_weighting.get_weights(context, in_context_spans)

        return ICs
    
    def _get_embedding_func(self, sim_model_name:str):
        embedding_model = None

        if sim_model_name is None: # Default spaCy model
            embedding_func = lambda x: np.array([self.spacy_nlp(text).vector for text in x])
        else:   # Sentence Transformer
            embedding_model = SentenceTransformer(sim_model_name, trust_remote_code=True)
            embedding_func = lambda x: embedding_model.encode(x, show_progress_bar=False)
        
        return embedding_func, embedding_model
    
    def _get_replacements_info(self, masked_doc:MaskedDocument, doc:Document, spans:List[Tuple[int, int]]) -> Tuple[list, list, list]:
        replacements = []
        masked_texts = []
        spans_idxs_per_replacement = []
        
        for replacement, (masked_span_start, masked_span_end) in zip(masked_doc.replacements, masked_doc.masked_spans):
            if not replacement is None: # If there is a replacement
                replacements.append(replacement)
                masked_texts.append(doc.text[masked_span_start:masked_span_end])
                replacement_spans_idxs = []
                for span_idx, (span_start, span_end) in enumerate(spans):
                    if span_start <= masked_span_start < span_end or span_start < masked_span_end <= span_end:
                        replacement_spans_idxs.append(span_idx)
                    elif span_start > masked_span_end:  # Break if candidate span starts too late
                        break
                spans_idxs_per_replacement.append(replacement_spans_idxs)
        
        return replacements, masked_texts, spans_idxs_per_replacement
    
    def _cos_sim(self, a:np.ndarray, b:np.ndarray) -> float:
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        sim = dot_product / (magnitude_a * magnitude_b)
        if np.isnan(sim):
            sim = 0
        return sim

    #endregion


    #region NMI

    def get_NMI(self, anonymizations:Dict[str, List[MaskedDocument]], min_k:int=NMI_MIN_K, max_k:int=NMI_MAX_K,
                k_multiplier:int=NMI_K_MULTIPLIER, clustering_embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=NMI_MASKING_MARKS,
                n_clusterings:int=NMI_N_CLUSTERINGS, n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> np.ndarray:
        
        # Create the corpora
        corpora = self._get_corpora_for_NMI(anonymizations)

        # Get the embeddings
        corpora_embeddings = self._get_corpora_embeddings(corpora, clustering_embedding_model_name,
                                                   remove_mask_marks=remove_mask_marks, mask_marks=mask_marks)
        
        # Clustering results based on the maximum silhouette
        values, all_labels, true_silhouettes = self._silhouette_based_NMI(corpora_embeddings, min_k=min_k, max_k=max_k, k_multiplier=k_multiplier,
                                                                      n_clusterings=n_clusterings, n_tries_per_clustering=n_tries_per_clustering)
        
        # Prepare results        
        values = values[1:] # Remove result for the first corpus (ground truth defined by the original texts)
        results = {anon_name:value for anon_name, value in zip(anonymizations.keys(), values)}
        
        return results, all_labels, true_silhouettes

    def _get_corpora_for_NMI(self, anonymizations:Dict[str, List[MaskedDocument]]) -> List[List[str]]:
        # The first corpus contains the original documents, sorted by doc_id
        original_corpus_with_ids = sorted(
            [(doc.doc_id, doc.text) for doc in self.documents.values()],
            key=lambda x: x[0]
        )
        corpora_with_ids = [original_corpus_with_ids]

        # Iterate through each anonymization method to create a masked corpora
        for masked_docs_list in anonymizations.values():
            # For each anonymization, generate the masked texts and pair with doc_ids
            masked_corpus_with_ids = sorted(
                [
                    (masked_doc.doc_id, masked_doc.get_masked_text(self.documents[masked_doc.doc_id].text))
                    for masked_doc in masked_docs_list
                ],
                key=lambda x: x[0]
            )
            corpora_with_ids.append(masked_corpus_with_ids)

        # Extract only the text from each corpus, maintaining the sorted order by doc_id
        corpora = [[text for _, text in corpus_with_ids] for corpus_with_ids in corpora_with_ids]

        return corpora

    #region Embedding/feature extraction

    def _get_corpora_embeddings(self, corpora:List[List[str]], clustering_embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                                 remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=NMI_MASKING_MARKS,
                                 device:str=DEVICE) -> List[np.ndarray]:
        corpora_embeddings = []

        # Load model
        model = SentenceTransformer(clustering_embedding_model_name, device=device)
        model.eval()
        
        # Collect embeddings
        mask_marks_re_pattern = "|".join([m.upper() for m in mask_marks])
        for corpus in tqdm(corpora, desc="Extracting embeddings"):
            # Remove mask marks if required
            if remove_mask_marks:
                corpus = [re.sub(mask_marks_re_pattern, "", text).strip() for text in corpus]            
            corpus_embeddings = model.encode(corpus,
                                             convert_to_numpy=True,
                                             show_progress_bar=False)
            corpora_embeddings.append(corpus_embeddings)
        
        # Remove model and tokenizer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return corpora_embeddings

    #endregion

    #region Clustering

    def _silhouette_based_NMI(self, corpora_embeddings:List[np.ndarray], min_k:int=NMI_MIN_K, max_k:int=NMI_MAX_K,
                k_multiplier:int=NMI_K_MULTIPLIER, n_clusterings:int=NMI_N_CLUSTERINGS, 
                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING):
        # For multiple ks, use results with maximum silhouette        
        outputs_by_k = {}
        max_silhouette = float("-inf")
        best_k = None
        k = min_k
        while k <= max_k:
            # Clustering for this k
            outputs_by_k[k] = self._get_corpora_multiclustering(corpora_embeddings, k, n_clusterings=n_clusterings,
                                                              n_tries_per_clustering=n_tries_per_clustering)            
            avg_silhouettee = outputs_by_k[k][2].mean() # Average of true_silhouettes
            if avg_silhouettee > max_silhouette:
                max_silhouette, best_k = avg_silhouettee, k
            k *= k_multiplier # By default, duplicate k

        logging.info(f"Clustering results for k={best_k} were selected because they correspond to the maximum silhouette ({max_silhouette:.2f})")
        values, all_labels, true_silhouettes = outputs_by_k[best_k]

        return values, all_labels, true_silhouettes

    def _get_corpora_multiclustering(self, corpora_embeddings:List[np.ndarray], k:int, n_clusterings:int=NMI_N_CLUSTERINGS,
                                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, np.ndarray]:
        results = np.empty((n_clusterings, len(corpora_embeddings)))
        true_silhouettes = np.empty(n_clusterings)
        for clustering_idx in tqdm(range(n_clusterings), desc=f"Clustering k={k}"):
            true_labels, corpora_labels, true_silhouettes[clustering_idx] = self._get_corpora_clustering(corpora_embeddings, k,
                                                                                                        tries_per_clustering=n_tries_per_clustering)
            results[clustering_idx, :] = self._compare_clusterings(true_labels, corpora_labels)

        # Average for the n_clusterings
        results = results.mean(axis=0)

        return results, corpora_labels, true_silhouettes

    def _get_corpora_clustering(self, corpora_embeddings:List[np.ndarray], k:int, tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, List[np.ndarray]]:
        corpora_labels = []

        # First corpus corresponds to the ground truth
        true_labels = self._get_corpus_clustering(corpora_embeddings[0], k, tries=tries_per_clustering)
        true_silhouette = silhouette_score(corpora_embeddings[0], true_labels, metric="cosine")

        # Clusterize for each corpus
        for corpus_embeddings in corpora_embeddings: # Repeating for the first one (ground truth) allows to check consistency
            labels = self._get_corpus_clustering(corpus_embeddings, k, tries=tries_per_clustering)            
            corpora_labels.append(labels)

        return true_labels, corpora_labels, true_silhouette

    def _get_corpus_clustering(self, corpus_embeddings, k:int, tries:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, float]:
        kmeanspp = KMeans(n_clusters=k, init='k-means++', n_init=tries)
        labels = kmeanspp.fit_predict(corpus_embeddings)
        return labels

    def _compare_clusterings(self, true_labels:np.ndarray, corpora_labels:List[np.ndarray],
                             eval_metric=normalized_mutual_info_score) -> np.ndarray:
        metrics = np.empty(len(corpora_labels))
        
        for idx, corpus_labels in enumerate(corpora_labels):
            metric = eval_metric(corpus_labels, true_labels)
            metrics[idx] = metric
        
        return metrics

    #endregion

    #endregion


    #region TRIR

    def get_TRIR(self, anonymizations:Dict[str, List[MaskedDocument]], 
                 output_folder_path:str, background_knowledge_file_path:str):
        #TODO: Modify and use _get_corpora_for_NMI
        # Transform corpora into dataframe
        results = {None for name, masked_docs in anonymizations.items()}

        tri = TRI(output_folder_path="To Be Defined",
            data_file_path="To Be Defined",
            individual_name_column=DOC_ID_KEY,
            background_knowledge_column="public_knowledge",
            anonymize_background_knowledge=False,
            use_document_curation=False,
            pretraining_epochs=1,
            finetuning_epochs=1)
        
        # TODO        

        return results

    #endregion


    #endregion

#endregion


#region TRIR core

#region ###################################### TRI class ######################################

class TRI():
    #region ################### Properties ###################

    #region ########## Mandatory configs ##########

    mandatory_configs_names = ["output_folder_path", "data_file_path",
        "individual_name_column", "background_knowledge_column"]
    output_folder_path = None
    data_file_path = None
    individual_name_column = None
    background_knowledge_column = None

    #endregion

    #region ########## Optional configs with default values ##########

    optional_configs_names = ["load_saved_pretreatment", "add_non_saved_anonymizations",
        "anonymize_background_knowledge", "only_use_anonymized_background_knowledge", 
        "use_document_curation", "save_pretreatment", "load_saved_finetuning", "base_model_name", 
        "tokenization_block_size", "use_additional_pretraining", "save_additional_pretraining",
        "load_saved_pretraining", "pretraining_epochs", "pretraining_batch_size",
        "pretraining_learning_rate", "pretraining_mlm_probability", "pretraining_sliding_window",
        "save_finetuning", "load_saved_finetuning", "finetuning_epochs", "finetuning_batch_size",
        "finetuning_learning_rate", "finetuning_sliding_window", "dev_set_column_name"]
    load_saved_pretreatment = True
    add_non_saved_anonymizations = True
    anonymize_background_knowledge = True
    only_use_anonymized_background_knowledge = True
    use_document_curation = True
    save_pretreatment = True
    base_model_name = "distilbert-base-uncased"
    tokenization_block_size = 250
    use_additional_pretraining = True
    save_additional_pretraining = True
    load_saved_pretraining = True
    pretraining_epochs = 3
    pretraining_batch_size = 8
    pretraining_learning_rate = 5e-05
    pretraining_mlm_probability = 0.15
    pretraining_sliding_window = "512-128"
    save_finetuning = True
    load_saved_finetuning = True
    finetuning_epochs = 15
    finetuning_batch_size = 16
    finetuning_learning_rate = 5e-05
    finetuning_sliding_window = "100-25"
    dev_set_column_name = False

    #endregion

    #region ########## Derived configs ##########

    pretreated_data_path:str = None
    pretrained_model_path:str = None
    results_path:str = None
    tri_pipe_path:str = None
    pretraining_config = Namespace()
    finetuning_config = Namespace()

    #endregion

    #region ########## Functional properties ##########

    # Data
    data_df:pd.DataFrame = None
    train_df:pd.DataFrame = None
    eval_dfs:dict = None
    train_individuals:set = None
    eval_individuals:set = None
    all_individuals:set = None
    no_train_individuals:set = None
    no_eval_individuals:set = None
    label_to_name:dict = None
    name_to_label:dict = None
    spacy_nlp = None
    pretreated_data_loaded:bool = None

    # Build classifier
    device = None
    tri_model = None
    tokenizer = None
    pretraining_dataset:Dataset = None # Removed after usage
    finetuning_dataset:Dataset = None      
    finetuning_trainer:Trainer = None
    eval_datasets_dict:dict = None
    pipe:Pipeline = None

    # Predict
    trir_results:dict = None

    #endregion

    #endregion

    #region ################### Constructor and configurations ###################

    def __init__(self, **kwargs):
        self.set_configs(**kwargs, are_mandatory_configs_required=True)

    def set_configs(self, are_mandatory_configs_required=False, **kwargs):
        arguments = kwargs.copy()

        # Mandatory configs
        for setting_name in self.mandatory_configs_names:
            value = arguments.get(setting_name, None)
            if isinstance(value, str):
                self.__dict__[setting_name] = arguments[setting_name]
                del arguments[setting_name]
            elif are_mandatory_configs_required:
                raise AttributeError(f"Mandatory argument {setting_name} is not defined or it is not a string")
        
        # Store remaining optional configs
        for (opt_setting_name, opt_setting_value) in arguments.items():
            if opt_setting_name in self.optional_configs_names:                
                if isinstance(opt_setting_value, str) or isinstance(opt_setting_value, int) or \
                isinstance(opt_setting_value, float) or isinstance(opt_setting_value, bool):
                    self.__dict__[opt_setting_name] = opt_setting_value
                else:
                    raise AttributeError(f"Optional argument {opt_setting_name} is not a string, integer, float or boolean.")
            else:
                logging.warning(f"Unrecognized setting name {opt_setting_name}")

        # Generate derived configs
        self.pretreated_data_path = os.path.join(self.output_folder_path, "Pretreated_Data.json")
        self.pretrained_model_path = os.path.join(self.output_folder_path, "Pretrained_Model.pt")
        self.results_file_path = os.path.join(self.output_folder_path, "Results.csv")
        self.tri_pipe_path = os.path.join(self.output_folder_path, "TRI_Pipeline")

        self.pretraining_config.is_for_mlm = True
        self.pretraining_config.uses_labels = False
        self.pretraining_config.epochs = self.pretraining_epochs
        self.pretraining_config.batch_size = self.pretraining_batch_size
        self.pretraining_config.learning_rate = self.pretraining_learning_rate
        self.pretraining_config.sliding_window = self.pretraining_sliding_window
        self.pretraining_config.trainer_folder_path = os.path.join(self.output_folder_path, f"Pretraining")

        self.finetuning_config.is_for_mlm = False
        self.finetuning_config.uses_labels = True
        self.finetuning_config.epochs = self.finetuning_epochs
        self.finetuning_config.batch_size = self.finetuning_batch_size
        self.finetuning_config.learning_rate = self.finetuning_learning_rate
        self.finetuning_config.sliding_window = self.finetuning_sliding_window
        self.finetuning_config.trainer_folder_path = os.path.join(self.output_folder_path, f"Finetuning")

        # Check for GPU with CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            self.device = torch.device("cpu")

    #endregion

    #region ################### Run all blocks ###################
    
    def run(self, verbose=True):
        self.run_data(verbose=verbose)
        self.run_build_classifier(verbose=verbose)
        results = self.run_predict_trir(verbose=verbose)
        return results

    #endregion

    #region ################### Data ###################

    def run_data(self, verbose=True):
        if verbose: logging.info("######### START: DATA #########")
        self.pretreated_data_loaded = False
        self.pretreatment_done = False

        # Create output directory if it does not exist
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path, exist_ok=True)

        # Read pretreated data if it exists        
        if self.load_saved_pretreatment and os.path.isfile(self.pretreated_data_path):
            if verbose: logging.info("######### START: LOADING SAVED PRETREATED DATA #########")
            self.train_df, self.eval_dfs = self.load_pretreatment()            
            self.pretreated_data_loaded = True

            # If curate non-saved anonymizations and document curation are required
            if self.add_non_saved_anonymizations:
                # Pretreated saved anonymizations
                self.saved_anons = set(self.eval_dfs.keys())

                # Non-pretreated anonymizations from raw file
                new_data_df = self.read_data()
                _, new_eval_dfs = self.split_data(new_data_df)
                self.non_pretreated_anons = set(new_eval_dfs.keys())

                # Find non-pretreated anonymizations not present in pretreated saved anonymizations
                self.non_saved_anons = []
                for anon_name in self.non_pretreated_anons:
                    if not anon_name in self.saved_anons:
                        self.non_saved_anons.append(anon_name)

                # If there are non-pretreated anonymizations not present in saved anonymizations, add them
                if len(self.non_saved_anons) > 0:
                    if verbose: logging.info("######### START: ADDING NON-SAVED ANONYMIZATIONS #########")
                    if verbose: logging.info(f"The following non-saved anonymizations will be added: {self.non_saved_anons}")
                    for anon_name in self.non_saved_anons:
                        # Curate anonymizations if needed
                        if self.use_document_curation:
                            self.curate_df(new_eval_dfs[anon_name], self.load_spacy_nlp())
                        # Add to eval_dfs
                        self.eval_dfs[anon_name] = new_eval_dfs[anon_name]
                    self.pretreatment_done = True
                    if verbose: logging.info("Non-saved anonymizations added")
                    if verbose: logging.info("######### END: ADDING NON-SAVED ANONYMIZATIONS #########")
                else:
                    if verbose: logging.info("There are not non-saved anonymizations to add")
                    if verbose: logging.info("######### SKIPPING: ADDING NON-SAVED ANONYMIZATIONS #########")
            else:
                if verbose: logging.info("######### SKIPPING: ADDING NON-SAVED ANONYMIZATIONS #########")

            if verbose: logging.info("######### END: LOADING SAVED PRETREATED DATA #########")

        # Otherwise, read raw data
        else:
            if self.load_saved_pretreatment:
                if verbose: logging.info(f"Impossible to load saved pretreated data, file {self.pretreated_data_path} not found.")

            if verbose: logging.info("######### START: READ RAW DATA FROM FILE #########")

            if verbose: logging.info("Reading data")
            self.data_df = self.read_data()
            if verbose: logging.info("Data reading complete")

            # Split into train and evaluation (dropping rows where no documents are available)
            if verbose: logging.info("Splitting into train (background knowledge) and evaluation (anonymized) sets")
            self.train_df, self.eval_dfs = self.split_data(self.data_df)
            del self.data_df # Remove general dataframe for saving memory
            if verbose: logging.info("Train and evaluation splitting complete")
            
            if verbose: logging.info("######### END: READ RAW DATA FROM FILE #########")

        if verbose: logging.info("######### START: DATA STATISTICS #########")

        # Get individuals found in each set
        res = self.get_individuals(self.train_df, self.eval_dfs)
        self.train_individuals, self.eval_individuals, self.all_individuals, self.no_train_individuals, self.no_eval_individuals = res

        # Generat Label->Name and Name->Label dictionaries
        self.label_to_name, self.name_to_label, self.num_labels = self.get_individuals_labels(self.all_individuals)

        # Show relevant information
        if verbose:
            self.show_data_stats(self.train_df, self.eval_dfs, self.no_eval_individuals, self.no_train_individuals, self.eval_individuals)        

        if verbose: logging.info("######### END: DATA STATISTICS #########")

        # Pretreat data if required and not pretreatment loaded
        if (self.anonymize_background_knowledge or self.use_document_curation) and not self.pretreated_data_loaded:
            if verbose: logging.info("######### START: DATA PRETREATMENT #########")
            
            if self.anonymize_background_knowledge:
                if verbose: logging.info("######### START: BACKGROUND KNOWLEDGE ANONYMIZATION #########")        
                self.train_df = self.anonymize_bk(self.train_df)
                if verbose: logging.info("######### END: BACKGROUND KNOWLEDGE ANONYMIZATION #########")
            else:
                if verbose: logging.info("######### SKIPPING: BACKGROUND KNOWLEDGE ANONYMIZATION #########")

            if self.use_document_curation:
                if verbose: logging.info("######### START: DOCUMENT CURATION #########")
                self.document_curation(self.train_df, self.eval_dfs)
                if verbose: logging.info("######### END: DOCUMENT CURATION #########")
            else:
                if verbose: logging.info("######### SKIPPING: DOCUMENT CURATION #########")            

            self.pretreatment_done = True

            if verbose: logging.info("######### END: DATA PRETREATMENT #########")
        else:
            if verbose: logging.info("######### SKIPPING: DATA PRETREATMENT #########")

        # If save pretreatment is required and there is any pretreatment modification to save
        if self.save_pretreatment and self.pretreatment_done:
            if verbose: logging.info("######### START: SAVE PRETREATMENT #########")
            self.save_pretreatment_dfs(self.train_df, self.eval_dfs)
            if verbose: logging.info("######### END: SAVE PRETREATMENT #########")
        else:
            if verbose: logging.info("######### SKIPPING: SAVE PRETREATMENT #########")
        
        if verbose: logging.info("######### END: DATA #########")

    #region ########## Load pretreatment ##########

    def load_pretreatment(self):
        with open(self.pretreated_data_path, "r") as f:
            (train_df_json_str, eval_dfs_jsons) = json.load(f)        
        
        train_df = pd.read_json(StringIO(train_df_json_str))
        eval_dfs = OrderedDict([(name, pd.read_json(StringIO(df_json))) for name, df_json in eval_dfs_jsons.items()])

        return train_df, eval_dfs
    
    #endregion

    #region ########## Data reading ##########

    def read_data(self) -> pd.DataFrame:
        if self.data_file_path.endswith(".json"):
            data_df = pd.read_json(self.data_file_path)
        elif self.data_file_path.endswith(".csv"):
            data_df = pd.read_csv(self.data_file_path)
        else:
            raise Exception(f"Unrecognized file extension for data file [{self.data_file_path}]. Compatible formats are JSON and CSV.")
        
        # Check required columns exist
        if not self.individual_name_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the individual name column {self.individual_name_column}")
        if not self.background_knowledge_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the background knowledge column {self.background_knowledge_column}")
        if self.dev_set_column_name is not False and not self.dev_set_column_name in data_df.columns:
            raise Exception(f"Dataframe does not contain the dev set column {self.dev_set_column_name}")
        
        # Check there are additional columns providing texts to re-identify
        anon_cols = [col_name for col_name in data_df.columns if not col_name in [self.individual_name_column, self.background_knowledge_column]]        
        if len(anon_cols) == 0:
            raise Exception(f"Dataframe does not contain columns with texts to re-identify, only individual name and background knowledge columns")
        
        # Sort by individual name
        data_df.sort_values(self.individual_name_column).reset_index(drop=True, inplace=True)

        return data_df

    def split_data(self, data_df:pd.DataFrame):
        data_df.replace('', np.nan, inplace=True)   # Replace empty texts by NaN

        # Training data formed by labeled background knowledge
        train_cols = [self.individual_name_column, self.background_knowledge_column]
        train_df = data_df[train_cols].dropna().reset_index(drop=True)

        # Evaluation data formed by texts to re-identify
        eval_columns = [col for col in data_df.columns if col not in train_cols]
        eval_dfs = {col:data_df[[self.individual_name_column, col]].dropna().reset_index(drop=True) for col in eval_columns}

        return train_df, eval_dfs

    #endregion

    #region ########## Data statistics ##########

    def get_individuals(self, train_df:pd.DataFrame, eval_dfs:dict):
        train_individuals = set(train_df[self.individual_name_column])
        eval_individuals = set()
        for name, eval_df in eval_dfs.items():
            if name != self.dev_set_column_name: # Exclude dev_set from these statistics
                eval_individuals.update(set(eval_df[self.individual_name_column]))
        all_individuals = train_individuals.union(eval_individuals)
        no_train_individuals = eval_individuals - train_individuals
        no_eval_individuals = train_individuals - eval_individuals

        return train_individuals, eval_individuals, all_individuals, no_train_individuals, no_eval_individuals

    def get_individuals_labels(self, all_individuals:set):
        sorted_indvidiuals = sorted(list(all_individuals)) # Sort individuals for ensuring same order every time (required for automatic loading)
        label_to_name = {idx:name for idx, name in enumerate(sorted_indvidiuals)}
        name_to_label = {name:idx for idx, name in label_to_name.items()}
        num_labels = len(name_to_label)

        return label_to_name, name_to_label, num_labels

    def show_data_stats(self, train_df:pd.DataFrame, eval_dfs:dict, no_eval_individuals:set, no_train_individuals:set, eval_individuals:set):
        logging.info(f"Number of background knowledge documents for training: {len(train_df)}")

        eval_n_dict = {name:len(df) for name, df in eval_dfs.items()}
        logging.info(f"Number of protected documents for evaluation: {eval_n_dict}")

        if len(no_eval_individuals) > 0:
            logging.info(f"No protected documents found for {len(no_eval_individuals)} individuals.")
        
        if len(no_train_individuals) > 0:
            max_risk = (1 - len(no_train_individuals) / len(eval_individuals)) * 100
            logging.info(f"No background knowledge documents found for {len(no_train_individuals)} individuals. Re-identification risk limited to {max_risk:.3f}% (excluding dev set).")

    #endregion

    #region ########## Data pretreatment ##########

    def load_spacy_nlp(self):
        # Load if it is not already loaded
        if self.spacy_nlp is None:
            self.spacy_nlp = en_core_web_lg.load()
        return self.spacy_nlp

    #region ##### Anonymize background knowledge #####
    
    def anonymize_bk(self, train_df:pd.DataFrame) -> pd.DataFrame:
        # Perform anonymization
        spacy_nlp = self.load_spacy_nlp()        
        train_anon_df = self.anonymize_df(train_df, spacy_nlp)

        if self.only_use_anonymized_background_knowledge:
            train_df = train_anon_df # Overwrite train dataframe with the anonymized version
        else:
            train_df = pd.concat([train_df, train_anon_df], ignore_index=True, copy=False) # Concatenate to train dataframe

        return train_df

    def anonymize_df(self, df, spacy_nlp, gc_freq=5) -> pd.DataFrame:
        assert len(df.columns) == 2 # Columns expected: name and text

        # Copy
        anonymized_df = df.copy(deep=True)

        # Process the text column
        column_name = anonymized_df.columns[1]
        texts = anonymized_df[column_name]
        for i, text in enumerate(tqdm(texts, desc=f"Anonymizing {column_name} documents")):
            new_text = text

            # Anonymize by NER
            doc = spacy_nlp(text) # Usage of spaCy NER (https://spacy.io/api/entityrecognizer)
            for e in reversed(doc.ents): # Reversed to not modify the offsets of other entities when substituting
                start = e.start_char
                end = start + len(e.text)
                new_text = new_text[:start] + e.label_ + new_text[end:]

            # Remove doc and (periodically) use GarbageCollector to reduce memory consumption
            del doc
            if i % gc_freq == 0:
                gc.collect()

            # Assign new text
            texts[i] = new_text

        return anonymized_df

    #endregion

    #region ##### Document curation #####

    def document_curation(self, train_df:pd.DataFrame, eval_dfs:dict):
        spacy_nlp = self.load_spacy_nlp()

        # Perform preprocessing for both training and evaluation
        self.curate_df(train_df, spacy_nlp)
        for eval_df in eval_dfs.values():
            self.curate_df(eval_df, spacy_nlp)

    def curate_df(self, df, spacy_nlp, gc_freq=5):
        assert len(df.columns) == 2 # Columns expected: name and text

        # Predefined patterns
        special_characters_pattern = re.compile(r"[^ \nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ./]+")
        stopwords = spacy_nlp.Defaults.stop_words

        # Process the text column (discarting the first one, that is the name column)
        column_name = df.columns[1]
        texts = df[column_name]
        for i, text in enumerate(tqdm(texts, desc=f"Preprocessing {column_name} documents")):
            doc = spacy_nlp(text) # Usage of spaCy (https://spacy.io/)
            new_text = ""   # Start text string
            for token in doc:
                if token.text not in stopwords:
                    # Lemmatize
                    token_text = token.lemma_ if token.lemma_ != "" else token.text
                    # Remove special characters
                    token_text = re.sub(special_characters_pattern, '', token_text)
                    # Add to new text (without space if dot)
                    new_text += ("" if token_text == "." else " ") + token_text

            # Remove doc and (periodically) use force GarbageCollector to reduce memory consumption
            del doc
            if i % gc_freq == 0:
                gc.collect()

            # Store result
            texts[i] = new_text

    #endregion

    #region ##### Save pretreatment #####

    def save_pretreatment_dfs(self, train_df:pd.DataFrame, eval_dfs:dict):
        with open(self.pretreated_data_path, "w") as f:
            f.write(json.dumps((train_df.to_json(orient="records"),
                                {name:df.to_json(orient="records") for name, df in eval_dfs.items()})))        

    #endregion

    #endregion

    #endregion

    #region ################### Build classifier ###################
    # Implementation grounded on HuggingFace's Transformers (https://huggingface.co/docs/transformers/index)

    def run_build_classifier(self, verbose=True):
        if verbose: logging.info("######### START: BUILD CLASSIFIER #########")

        if self.load_saved_finetuning and os.path.exists(self.tri_pipe_path):
            if verbose: logging.info("######### START: LOAD ALREADY TRAINED TRI MODEL #########")

            # Get TRI classifier and tokenizer
            self.tri_model, self.tokenizer = self.load_trained_TRI_model()

            # Datasets for TRI
            res = self.create_datasets(self.train_df, self.eval_dfs, self.tokenizer, self.name_to_label, self.finetuning_config)
            self.finetuning_dataset, self.eval_datasets_dict = res

            # Create trainer for TRI
            self.finetuning_trainer = self.get_trainer(self.tri_model, self.finetuning_config,
                                                        self.finetuning_dataset, eval_datasets_dict=self.eval_datasets_dict)
            
            if verbose: logging.info("######### END: LOAD ALREADY TRAINED TRI MODEL #########")

        # Otherwise, pretrain (if required) and finetune a TRI model
        else:
            if self.load_saved_finetuning:
                if verbose: logging.info(f"Fail loading saved TRI pipeline: Folder {self.tri_pipe_path} not found.")

            if verbose: logging.info("######### START: CREATE BASE LANGUAGE MODEL #########")
            self.base_model, self.tokenizer = self.create_base_model(verbose=verbose)
            if verbose: logging.info("######### END: CREATE BASE LANGUAGE MODEL #########")

            if self.use_additional_pretraining:
                if verbose: logging.info("######### START: ADDITIONAL PRETRAINING #########")

                if self.load_saved_pretraining and os.path.exists(self.pretrained_model_path):
                    if verbose: logging.info("Loading additionally pretrained base model")
                    self.load_pretrained_base_model(self.base_model)
                    if verbose: logging.info("Additionally pretrained base model loaded")
                else:
                    if self.load_saved_pretraining:
                        if verbose: logging.info(f"Fail loading saved pretrained base model: File {self.pretrained_model_path} not found.")

                    # Datasets for additional pretraining
                    self.pretraining_dataset, _ = self.create_datasets(self.train_df, self.eval_dfs, self.tokenizer, 
                                                                       self.name_to_label, self.pretraining_config)

                    # Additionally pretrain the base language model
                    self.base_model = self.additional_pretraining(self.base_model, self.tokenizer, self.pretraining_config, 
                                                                  self.pretraining_dataset, verbose=verbose)
                    
                    if self.save_additional_pretraining:
                        if verbose: logging.info("Saving additionally pretrained base model")
                        self.save_pretrained_base_model(self.base_model)
                        if verbose: logging.info("Additionally pretrained base model saved")
                
                if verbose: logging.info("######### END: ADDITIONAL PRETRAINING #########")
            else:
                if verbose: logging.info("######### SKIPPING: ADDITIONAL PRETRAINING #########")            

            if verbose: logging.info("######### START: FINETUNING #########")

            # Datasets for finetuning
            self.finetuning_dataset, self.eval_datasets_dict = self.create_datasets(self.train_df, self.eval_dfs, 
                                                                                    self.tokenizer, self.name_to_label,
                                                                                      self.finetuning_config)

            # Finetuning for text re-identification
            self.tri_model, self.finetuning_trainer, _ = self.finetuning(self.base_model, self.num_labels,
                                                                     self.finetuning_config, self.finetuning_dataset,
                                                                     self.eval_datasets_dict, verbose=verbose)

            if self.save_finetuning:
                if verbose: logging.info("Saving finetuned TRI pipeline")
                self.pipe, self.tri_model = self.save_finetuned_tri_pipeline(self.tri_model, self.tokenizer)
                if verbose: logging.info("Finetuned TRI model pipeline saved")

            if verbose: logging.info("######### END: FINETUNING #########")

        if verbose: logging.info("######### END: BUILD CLASSIFIER #########")
    
    #region ########## Load already trained TRI model ##########

    def load_trained_TRI_model(self):
        tri_model = AutoModelForSequenceClassification.from_pretrained(self.tri_pipe_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tri_pipe_path)
        return tri_model, tokenizer

    #endregion

    #region ########## Create base language model ##########

    def create_base_model(self, verbose=True):
        base_model = AutoModel.from_pretrained(self.base_model_name)
        if verbose: logging.info(f"Model size = {sum([np.prod(p.size()) for p in base_model.parameters()])}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        return base_model, tokenizer

    #endregion
    
    #region ########## Load additional pretraining ##########

    def load_pretrained_base_model(self, base_model):
        base_model.load_state_dict(torch.load(self.pretrained_model_path))

    #endregion  

    #region ########## Additional pretraining ##########

    def additional_pretraining(self, base_model, tokenizer, pretraining_config:Namespace, pretraining_dataset:Dataset, verbose=True):
        # Create MLM model
        pretraining_model = AutoModelForMaskedLM.from_pretrained(self.base_model_name)
        pretraining_model = self.ini_extended_model(base_model, pretraining_model, link_instead_of_copy_base_model=True, verbose=verbose)

        # Create data collator for training
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=self.pretraining_mlm_probability)
        
        # Perform further pretraining
        pretraining_trainer = self.get_trainer(pretraining_model, pretraining_config, pretraining_dataset, data_collator=data_collator)
        pretraining_trainer.train()

        # Move base_model to CPU to free GPU memory
        base_model = base_model.cpu()
        
        # Clean memory
        del pretraining_model # Remove header from MaskedLM
        del pretraining_dataset
        del pretraining_trainer
        gc.collect()
        torch.cuda.empty_cache()

        return base_model

    #endregion

    #region ########## Save additional pretraining ##########

    def save_pretrained_base_model(self, base_model):
        torch.save(base_model.state_dict(), self.pretrained_model_path)

    #endregion

    #region ########## Finetuning ##########

    def finetuning(self, base_model, num_labels, finetuning_config, finetuning_dataset, eval_datasets_dict, verbose=True):
        # Create classifier
        tri_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, num_labels=num_labels)

        # Initialize model
        tri_model = self.ini_extended_model(base_model, tri_model, link_instead_of_copy_base_model=False, verbose=verbose)

        # Create trainer and train
        finetuning_trainer = self.get_trainer(tri_model, finetuning_config, finetuning_dataset,
                                                    eval_datasets_dict=eval_datasets_dict)
        training_results = finetuning_trainer.train()

        return tri_model, finetuning_trainer, training_results

    #endregion

    #region ########## Save finetuned TRI model ##########

    def save_finetuned_tri_pipeline(self, tri_model, tokenizer):
        pipe = pipeline("text-classification", model=tri_model, tokenizer=tokenizer)
        pipe.save_pretrained(self.tri_pipe_path)
        tri_model = tri_model.to(self.device) # Saving moves model to CPU, return it to defined DEVICE
        return pipe, tri_model

    #endregion

    #region ########## Common ##########

    def create_datasets(self, train_df, eval_dfs, tokenizer, name_to_label, task_config):
        train_dataset = TRIDataset(train_df, tokenizer, name_to_label, task_config.uses_labels, task_config.sliding_window, self.tokenization_block_size)
        eval_datasets_dict = OrderedDict([(name, TRIDataset(eval_df, tokenizer, name_to_label, task_config.uses_labels, task_config.sliding_window, self.tokenization_block_size)) for name, eval_df in eval_dfs.items()])
        return train_dataset, eval_datasets_dict

    def ini_extended_model(self, base_model, extended_model, link_instead_of_copy_base_model, verbose=True):
        # Link: Use base_model in extended model
        if link_instead_of_copy_base_model:
            if "distilbert" in self.base_model_name:
                old_base_model = extended_model.distilbert
                extended_model.distilbert = base_model
            elif "roberta" in self.base_model_name:
                old_base_model = extended_model.roberta
                extended_model.roberta = base_model
            elif "bert" in self.base_model_name:
                old_base_model = extended_model.bert
                extended_model.bert = base_model
            else:
                raise Exception(f"Not code available for base model [{self.base_model_name}]")
            
            # Remove old base model for memory saving
            del old_base_model
            gc.collect()

        # Copy: Clone the weights of base_model into extended model
        else:
            if "distilbert" in self.base_model_name:
                extended_model.distilbert.load_state_dict(base_model.state_dict())
            elif "roberta" in self.base_model_name:
                base_model_dict = base_model.state_dict()
                base_model_dict = dict(base_model_dict) # Copy
                base_model_dict.pop("pooler.dense.weight")  # Specific for transformers version 4.20.1
                base_model_dict.pop("pooler.dense.bias")
                extended_model.roberta.load_state_dict(base_model_dict)
            elif "bert" in self.base_model_name:
                extended_model.bert.load_state_dict(base_model.state_dict())
            else:
                raise Exception(f"No code available for base model [{self.base_model_name}]")

        # Model to device, and show size
        extended_model.to(self.device)
        if verbose: 
            logging.info(f"Extended model size = {sum([np.prod(p.size()) for p in extended_model.parameters()])}")

        return extended_model

    def get_trainer(self, model, task_config, train_dataset, eval_datasets_dict=None, data_collator=None):
        is_for_mlm = task_config.is_for_mlm

        # Settings for additional pretraining (Masked Language Modeling)
        if is_for_mlm:
            eval_strategy = "no"
            save_strategy = "no"
            load_best_model_at_end = False
            metric_for_best_model = None
            eval_datasets_dict = None
            results_file_path = None
        # Settings for finetuning
        else:
            eval_strategy = "epoch"
            save_strategy = "epoch"
            load_best_model_at_end = True
            if self.dev_set_column_name:
                metric_for_best_model = self.dev_set_column_name+"_eval_Accuracy" # Prefix (e.g., "eval_") will be added by the Trainer
            else:
                metric_for_best_model = "avg_Accuracy" # Prefix (e.g., "eval_") will be added later will be added by the Trainer
            eval_datasets_dict = self.eval_datasets_dict
            results_file_path = self.results_file_path

        # Define TrainingArguments
        args = TrainingArguments(
            output_dir=task_config.trainer_folder_path,
            overwrite_output_dir=True,
            load_best_model_at_end=load_best_model_at_end,
            save_strategy=save_strategy,
            save_total_limit=1,
            num_train_epochs=task_config.epochs,
            per_device_train_batch_size=task_config.batch_size,
            per_device_eval_batch_size=task_config.batch_size,
            logging_strategy="epoch",
            logging_steps=500,
            eval_strategy=eval_strategy,
            disable_tqdm=False,
            eval_accumulation_steps=5,  # Number of eval steps before move preds are moved from GPU to RAM        
            dataloader_num_workers=0,
            metric_for_best_model=metric_for_best_model,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
        )

        # Define optimizer
        optimizer = AdamW(model.parameters(), lr=task_config.learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0)
        scheduler = get_constant_schedule(optimizer)

        # Use Accelerate
        accelerator = Accelerator()
        (model, optimizer, scheduler, train_dataset) = accelerator.prepare(model, optimizer, scheduler, train_dataset)

        # Define trainer    
        trainer = TRITrainer(results_file_path,
                            model=model,
                            args=args,
                            train_dataset=train_dataset,
                            eval_dataset=eval_datasets_dict,
                            optimizers=[optimizer, scheduler],
                            compute_metrics=get_TRI_accuracy,
                            data_collator=data_collator
                        )
        
        return trainer

    #endregion

    #endregion

    #region ################### Predict TRIR ###################
    
    def run_predict_trir(self, verbose=True):
        if verbose: logging.info("######### START: PREDICT TRIR #########")

        # Predict
        self.finetuning_trainer.evaluate()
        
        # Show results from the last epoch (i.e., just already done evaluation)
        self.trir_results = self.finetuning_trainer.all_results[-1]
        
        # Show results
        if verbose:
            for dataset_name, res in self.trir_results.items():
                #res_key = list(filter(lambda x:x.endswith("_Accuracy"), res.keys()))[0]
                logging.info(f"TRIR {dataset_name} = {res['eval_Accuracy']}%")
        
        if verbose: logging.info("######### END: PREDICT TRIR #########")
        
        return self.trir_results

    #endregion

#endregion

#region ###################################### TRI dataset ######################################

class TRIDataset(Dataset):
    def __init__(self, df, tokenizer, name_to_label, return_labels, sliding_window_config, tokenization_block_size):
        # Dataframe must have two columns: name and text
        assert len(df.columns) == 2
        self.df = df

        # Set general attributes
        self.tokenizer = tokenizer
        self.name_to_label = name_to_label
        self.return_labels = return_labels

        # Set sliding window
        self.sliding_window_config = sliding_window_config
        try:
            sw_elems = [int(x) for x in sliding_window_config.split("-")]
            self.sliding_window_length = sw_elems[0]
            self.sliding_window_overlap = sw_elems[1]
            self.use_sliding_window = True
        except:
            self.use_sliding_window = False # If no sliding window (e.g., "No"), use sentence splitting

        if self.use_sliding_window and self.sliding_window_length > self.tokenizer.model_max_length:
            logging.exception(f"Sliding window length ({self.sliding_window_length}) must be lower than the maximum sequence length ({self.tokenizer.model_max_length})")     

        self.tokenization_block_size = tokenization_block_size

        # Compute inputs and labels
        self.generate()

    def generate(self, gc_freq=5):
        texts_column = list(self.df[self.df.columns[1]])
        names_column = list(self.df[self.df.columns[0]])
        labels_idxs = list(map(lambda x: self.name_to_label[x], names_column))   # Compute labels, translated to the identity index
        
        # Sliding window
        if self.use_sliding_window:
            texts = texts_column            
            labels = labels_idxs
        # Sentence splitting
        else:
            texts = []
            labels = []

            # Load spacy model for sentence splitting
            # Create spaCy model. Compontents = tok2vec, tagger, parser, senter, attribute_ruler, lemmatizer, ner
            # disable = ["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "ner"]) # Required components: "senter" and "parser"
            spacy_nlp = en_core_web_lg.load(disable = ["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"])
            spacy_nlp.add_pipe('sentencizer')

            # Get texts and labels per sentence
            for idx, (text, label) in tqdm(enumerate(zip(texts_column, labels_idxs)), total=len(texts_column),
                                                    desc="Processing sentence splitting"):
                for paragraph in text.split("\n"):
                    if len(paragraph.strip()) > 0:
                        doc = spacy_nlp(paragraph)
                        for sentence in doc.sents:
                            # Parse sentence to text
                            sentence_txt = ""
                            for token in sentence:
                                sentence_txt += " " + token.text
                            sentence_txt = sentence_txt[1:] # Remove initial space
                            # Ensure length is less than the maximum
                            sent_token_count = len(self.tokenizer.encode(sentence_txt, add_special_tokens=True))
                            if sent_token_count > self.tokenizer.model_max_length:
                                logging.exception(f"ERROR: Sentence with length {sent_token_count} > {self.tokenizer.model_max_length} at index {idx} with label {label} not included because is too long | {sentence_txt}")
                            else:
                                # Store sample
                                texts.append(sentence_txt)
                                labels.append(label)
                    
                        # Delete document for reducing memory consumption
                        del doc
                    
                # Periodically force GarbageCollector for reducing memory consumption
                if idx % gc_freq == 0:
                    gc.collect()
                
        # Tokenize texts
        self.inputs, self.labels = self.tokenize_data(texts, labels)        

    def tokenize_data(self, texts, labels):
        # Sliding window
        if self.use_sliding_window:
            input_length = self.sliding_window_length
            padding_strategy = "longest"
        # Sentence splitting
        else:
            input_length = self.tokenizer.model_max_length            
            padding_strategy = "max_length"

        all_input_ids = torch.zeros((0, input_length), dtype=torch.int)
        all_attention_masks = torch.zeros((0, input_length), dtype=torch.int)
        all_labels = []

        # For each block of data
        with tqdm(total=len(texts)) as pbar:
            for ini in range(0, len(texts), self.tokenization_block_size):
                end = min(ini+self.tokenization_block_size, len(texts))
                pbar.set_description("Tokenizing (progress bar frozen)")
                block_inputs = self.tokenizer(texts[ini:end],
                                            add_special_tokens=not self.use_sliding_window,
                                            padding=padding_strategy,  # Warning: If an text is longer than tokenizer.model_max_length, an error will raise on prediction
                                            truncation=False,
                                            max_length=self.tokenizer.model_max_length,
                                            return_tensors="pt")
                
                # Force GarbageCollector after tokenization
                gc.collect()

                # Sliding window
                if self.use_sliding_window:                    
                    all_input_ids, all_attention_masks, all_labels = self.do_sliding_window(labels[ini:end], input_length, all_input_ids, all_attention_masks, all_labels, pbar, block_inputs)
                # Sentence splitting
                else:
                    # Concatenate to all data            
                    all_input_ids = torch.cat((all_input_ids, block_inputs["input_ids"]))
                    all_attention_masks = torch.cat((all_attention_masks, block_inputs["attention_mask"]))
                    all_labels = labels
                    pbar.update(len(block_inputs))

        # Get inputs
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_masks}

        # Transform labels to tensor
        labels = torch.tensor(all_labels)

        return inputs, labels

    def do_sliding_window(self, block_labels, input_length, all_input_ids, all_attention_masks, all_labels, pbar, block_inputs):
        # Predict number of windows
        n_windows = 0
        old_seq_length = block_inputs["input_ids"].size()[1]
        window_increment = self.sliding_window_length - self.sliding_window_overlap - 2 # Minus 2 because of the CLS and SEP tokens
        for old_attention_mask in block_inputs["attention_mask"]:
            is_sequence_finished = False
            is_padding_required = False
            ini = 0
            end = ini + self.sliding_window_length - 2
            while not is_sequence_finished:
                # Get the corresponding window's ids and mask
                if end > old_seq_length:
                    end = old_seq_length
                    is_padding_required = True
                window_mask = old_attention_mask[ini:end]
                            
                # Check end of sequence
                is_sequence_finished = end == old_seq_length or is_padding_required or window_mask[-1] == 0

                # Increment indexes
                ini += window_increment
                end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens

                n_windows += 1
                    
        # Allocate memory for ids and masks
        all_sequences_windows_ids = torch.empty((n_windows, input_length), dtype=torch.int)
        all_sequences_windows_masks = torch.empty((n_windows, input_length), dtype=torch.int)                                   

        # Sliding window for block sequences' splitting
        window_idx = 0
        old_seq_length = block_inputs["input_ids"].size()[1]
        pbar.set_description("Processing sliding window")
        for block_seq_idx, (old_input_ids, old_attention_mask) in enumerate(zip(block_inputs["input_ids"], block_inputs["attention_mask"])):
            ini = 0
            end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens
            is_sequence_finished = False
            is_padding_required = False
            n_windows_in_seq = 0
            while not is_sequence_finished:
                # Get the corresponding window's ids and mask
                if end > old_seq_length:
                    end = old_seq_length
                    is_padding_required = True
                window_ids = old_input_ids[ini:end]
                window_mask = old_attention_mask[ini:end]

                # Check end of sequence
                is_sequence_finished = end == old_seq_length or is_padding_required or window_mask[-1] == 0

                # Add CLS and SEP tokens
                num_attention_tokens = torch.count_nonzero(window_mask)
                if num_attention_tokens == window_mask.size()[0]:  # If window is full
                    window_ids = torch.cat(( torch.tensor([self.tokenizer.cls_token_id]), window_ids, torch.tensor([self.tokenizer.sep_token_id]) ))
                    window_mask = torch.cat(( torch.tensor([1]), window_mask, torch.tensor([1]) )) # Attention to CLS and SEP
                else: # If window has empty space (to be padded later)
                    window_ids[num_attention_tokens] = torch.tensor(self.tokenizer.sep_token_id) # SEP at last position
                    window_mask[num_attention_tokens] = 1 # Attention to SEP
                    window_ids = torch.cat(( torch.tensor([self.tokenizer.cls_token_id]), window_ids, torch.tensor([self.tokenizer.pad_token_id]) )) # PAD at the end of sentence
                    window_mask = torch.cat(( torch.tensor([1]), window_mask, torch.tensor([0]) )) # No attention to PAD

                # Padding if it is required
                if is_padding_required:
                    padding_length = self.sliding_window_length - window_ids.size()[0]
                    padding = torch.zeros((padding_length), dtype=window_ids.dtype)
                    window_ids = torch.cat((window_ids, padding))
                    window_mask = torch.cat((window_mask, padding))

                # Store ids and mask
                all_sequences_windows_ids[window_idx] = window_ids
                all_sequences_windows_masks[window_idx] = window_mask

                # Increment indexes
                ini += self.sliding_window_length - self.sliding_window_overlap - 2 # Minus 2 because of the CLS and SEP tokens
                end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens
                n_windows_in_seq += 1
                window_idx += 1
                        
            # Stack lists and concatenate with new data
            all_labels += [block_labels[block_seq_idx]] * n_windows_in_seq
            pbar.update(1)
                    
        # Concat the block data        
        all_input_ids = torch.cat((all_input_ids, all_sequences_windows_ids))
        all_attention_masks = torch.cat((all_attention_masks, all_sequences_windows_masks))

        # Force GarbageCollector after sliding window
        gc.collect()

        return all_input_ids, all_attention_masks, all_labels
    
    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, index):
        # Get each value (tokens, attention...) of the item
        input = {key: value[index] for key, value in self.inputs.items()}

        # Get label if is required
        if self.return_labels:
            label = self.labels[index]
            input["labels"] = label
        
        return input

#endregion

#region ###################################### TRI trainer ######################################

class TRITrainer(Trainer):
    def __init__(self, results_file_path:str = None, **kwargs):
        Trainer.__init__(self, **kwargs)
        self.results_file_path = results_file_path
        
        if self.results_file_path is not None and "eval_dataset" in self.__dict__ and isinstance(self.eval_dataset, dict):
            self.do_custom_eval = True
            self.eval_dataset_dict = self.eval_dataset
        else:
            self.do_custom_eval = False
        
        if self.do_custom_eval:
            self.eval_datasets_dict = self.eval_dataset
            self.all_results = []
            self.evaluation_epoch = 1   # Start epoch counter
            self.initialize_results_file()
    
    def current_time_str(self):
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
    def initialize_results_file(self):
        text = f"{self.current_time_str()}\n"
        text += "Time,Epoch"
        for dataset_name in self.eval_datasets_dict.keys():
            text+=f",{dataset_name}"
        text+=",Average"
        text += "\n"
        self.write_results(text)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.do_custom_eval:
            metrics = OrderedDict()
            structured_results = OrderedDict()
            avg_loss = 0
            loss_key = f"{metric_key_prefix}_loss"
            avg_acc = 0
            acc_key = f"{metric_key_prefix}_Accuracy"

            # Get results
            for dataset_name, dataset in self.eval_datasets_dict.items():
                dataset_metrics = Trainer.evaluate(self, eval_dataset=dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)                
                avg_loss += dataset_metrics[loss_key] / len(self.eval_datasets_dict)
                avg_acc += dataset_metrics[acc_key] / len(self.eval_datasets_dict)
                structured_results[dataset_name] = dataset_metrics
                dataset_metrics = {f"{metric_key_prefix}_{dataset_name}_{key}":val for key, val in dataset_metrics.items()} # Add dataset name to results keys
                metrics.update(dataset_metrics)
                
            
            # Add average metrics to results
            metrics.update({f"{metric_key_prefix}_avg_loss": avg_loss, f"{metric_key_prefix}_avg_Accuracy": avg_acc})
            
            # Save results into file and list
            self.store_results(metrics)
            self.all_results.append(structured_results)

            # Increment evaluation epoch
            self.evaluation_epoch += 1

            # Add average metrics with the prefix, for compatibility with super class
            metrics.update({loss_key: avg_loss, acc_key: avg_acc})

            return metrics
        # Otherwise, standard evaluation with eval_dataset
        else:
            return Trainer.evaluate(self, eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)        
    
    def store_results(self, eval_results:dict):
        current_time = self.current_time_str()
        try:
            results_text = f"{current_time},{self.evaluation_epoch}"
            for key, value in eval_results.items():
                if key.endswith("_Accuracy"):
                    results_text += f",{value:.3f}"
            results_text += "\n"
            self.write_results(results_text)
        except Exception as e:
            self.write_results(f"{current_time}, Error writing the results of epoch {self.evaluation_epoch} ({e})")
            logging.info(f"ERROR writing the results: {e}")

    def write_results(self, text:str):
        with open(self.results_file_path, "a+") as f:
            f.write(text)

def get_TRI_accuracy(results):
    logits, labels = results

    # Get predictions sum
    logits = torch.from_numpy(logits)
    logits_dict = {}
    for logit, label in zip(logits, labels):
        current_logits = logits_dict.get(label, torch.zeros_like(logit))
        logits_dict[label] = current_logits.add_(logit)
    
    # Cumpute final predictions
    num_preds = len(logits_dict)
    all_preds = torch.zeros(num_preds, device="cpu")
    all_labels = torch.zeros(num_preds, device="cpu")
    for idx, item in enumerate(logits_dict.items()):
        label, logits = item
        all_labels[idx] = label
        probs = F.softmax(logits, dim=-1)
        all_preds[idx] = torch.argmax(probs)
    
    correct_preds = torch.sum(all_preds == all_labels)
    accuracy = (float(correct_preds)/num_preds)*100
    return {"Accuracy": accuracy}

#endregion

#endregion


#endregion


#region Main

if __name__ == "__main__":

    #region Arguments parsing

    parser = argparse.ArgumentParser(description='Computes evaluation metrics for text anonymization')
    parser.add_argument('config_file_path', type=str,
                        help='the path to the JSON file containing the configuration for the evaluation')
    args = parser.parse_args()

    # Load configuration dictionary
    with open(args.config_file_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    for key in MANDATORY_CONFIG_KEYS:
        if not key in config.keys():
            raise RuntimeError(f"Configuration JSON file misses mandatory key {key}")
    
    #endregion


    #region Initialization

    logging.info(f"Device for models: {DEVICE}")

    # Create TAE from corpus
    corpus_file_path = config[CORPUS_CONFIG_KEY]
    tae = TAE.from_file_path(corpus_file_path)

    # Create anonymizations
    anonymizations = {}
    anonymizations_config = config[ANONYMIZATIONS_CONFIG_KEY]
    for anon_name, anon_file_path in anonymizations_config.items():
        masked_docs = MaskedCorpus(anon_file_path)
        anonymizations[anon_name] = masked_docs
    
    # Get metrics
    metrics = config[METRICS_CONFIG_KEY]

    # Get file_path for results CSV file
    results_file_path = config[RESULTS_CONFIG_KEY]

    #endregion


    #region Evaluate

    tae.evaluate(anonymizations, metrics, results_file_path)

    #endregion


#endregion