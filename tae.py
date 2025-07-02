#region Imports


import json, re, abc, argparse, math, os, csv, gc
os.environ["OMP_NUM_THREADS"] = "1" # Done before loading MKL to avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
import spacy
import intervaltree

import torch
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

#endregion


#region Constants


#region Input settings

# Configuration dictionary keys
CORPUS_CONFIG_KEY = "corpus_filepath"
ANONYMIZATIONS_CONFIG_KEY = "anonymizations"
RESULTS_CONFIG_KEY = "results_filepath"
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
NMI_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Other options: "bert-base-cased", "distilbert-base-uncased", "distilbert-base-cased", "bert-base-uncased", "roberta-base" or models from https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
NMI_EMBEDDING_SIZE = 768 # BERT embedding size #TODO: This might not be necessary anymore
NMI_K = 4 #TODO: This should not be defaulted
NMI_REMOVE_MASK_MARKS = False
NMI_MASKING_MARKS = ["SENSITIVE", "PERSON", "DEM", "LOC",
                 "ORG", "DATETIME", "QUANTITY", "MISC",
                 "NORP", "FAC", "GPE", "PRODUCT", "EVENT",
                 "WORK_OF_ART", "LAW", "LANGUAGE", "DATE",
                 "TIME", "ORDINAL", "CARDINAL", "DATE_TIME", "DATETIME",
                 "NRP", "LOCATION", "ORGANIZATION", "\*\*\*"]
NMI_MAX_POOLING = False
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
    def __init__(self, masked_docs_filepath:str):
        """Given a file path for a JSON file containing the spans to be masked for
        each document, builds a list of MaskedDocument objects"""

        masked_docs_list = []
        
        with open(masked_docs_filepath, "r", encoding="utf-8") as fd:
            masked_docs_dict = json.load(fd)
        
        if type(masked_docs_dict)!= dict:
            raise RuntimeError(f"List of MaskedDocuments in {masked_docs_filepath} must contain a dictionary mapping between document identifiers"
                                + " and lists of masked spans in this document")
        
        for doc_id, masked_spans in masked_docs_dict.items():
            doc = MaskedDocument(doc_id, [], [])
            if type(masked_spans)!=list:
                raise RuntimeError("A masked span is defined as [start, end, replacement] tuple (replacement is optional)")
            
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
                    raise RuntimeError(f"Entity mention missing key: {key}")
            
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
            if set(entity.mention_level_masking) != {entity.need_masking}:
                entity.need_masking = True
                #logging.info(f"Warning: inconsistent masking of entity {entity.entity_id}: {entity.mention_level_masking}") # TODO: Check this
                
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
        
        mention_to_mask = self.text[mention_start:mention_end].lower() #TODO: Check this unused var
                
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
        #transformers.logging.set_verbosity_error() # Suppress warnings #TODO: Check this
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
        #transformers.logging.set_verbosity_warning() # Restore warnings #TODO: Check this
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
    def from_filepath(cls, corpus_filepath:str):
        with open(corpus_filepath, encoding="utf-8") as f:
            corpus = json.load(f)
        if type(corpus)!=list:
            raise RuntimeError("Corpus JSON file must be a list of documents")
        return TAE(corpus)

    #endregion


    #region Evaluation

    def evaluate(self, anonymizations:Dict[str, List[MaskedDocument]], metrics:dict, results_filepath:Optional[str]=None, verbose:bool=True) -> dict:
        results = {}

        # Initial checks
        self._eval_checks(anonymizations, metrics)

        # Write results file header
        if results_filepath:
            self._write_into_results(results_filepath, ["Metric/Anonymization"]+list(anonymizations.keys()))

        # For each metric
        for metric_name, metric_parameters in metrics.items():
            if verbose:
                logging.info(f"Computing {metric_name} metric")
            metric_key = metric_name.split("_")[0] # Text before first underscore is name of the metric, the rest is freely used
            partial_eval_func = self._get_partial_eval_func(metric_key, metric_parameters)

            # If metric is invalid, results are None
            if partial_eval_func is None:      
                metric_results = {anon_name:None for anon_name in anonymizations.keys()}

            # For NMI, evaluate all anonymizations at once
            elif partial_eval_func.func==self.get_NMI:
                values, all_labels, true_intertias = partial_eval_func(anonymizations)
                metric_results = {anon_name:value for anon_name, value in zip(anonymizations.keys(), values)}                
                metric_name += f" (Inertia={true_intertias.mean():.2f})" # Append intertia info to metric name #TODO: Check if intertia is the proper metric
            
            # Otherwise, compute metric for each anonymization
            else:                
                metric_results = {}
                with tqdm(anonymizations.items(), desc="Processing each anonymization") as pbar:
                    for anon_name, masked_docs in pbar:
                        pbar.set_description(f"Processing {anon_name} anonymization")
                        output = partial_eval_func(masked_docs)
                        value = output[0] if isinstance(output, tuple) else output  # If tuple, the first is the actual value of the metric
                        metric_results[anon_name] = value

            # Save and show results
            results[metric_name] = metric_results
            if verbose:
                logging.info(f"Results: {results[metric_name]}") #TODO: Print results in a fancy way
            if results_filepath: #TODO: CSV is bad format for complex results such as those from recall_per_entity_type. Is JSON better instead (worse for Excel)
                self._write_into_results(results_filepath, [metric_name]+list(metric_results.values()))
        
        return results

    def _eval_checks(self, anonymizations:Dict[str, List[MaskedDocument]], metrics:dict):
        # Check each anonymization has a masked version of all the documents in the corpus
        for anon_name, masked_docs in anonymizations.items():
            corpus_doc_ids = set(self.documents.keys())
            for masked_doc in masked_docs:
                if masked_doc.doc_id in corpus_doc_ids:
                    corpus_doc_ids.remove(masked_doc.doc_id)
                else:
                    logging.warning(f"Anonymization {anon_name} includes a masked document (ID={masked_doc.doc_id}) not present in the corpus") #TODO: Standarize error an logging texts
            if len(corpus_doc_ids) > 0:
                raise RuntimeError(f"Anonymization {anon_name} misses masked documents for the following {len(corpus_doc_ids)} ID/s: {corpus_doc_ids}")
        
        # Check all metrics are valid and can be computed
        for name, parameters in metrics.items():
            metric_key = name.split("_")[0]
            if not metric_key in METRIC_NAMES:
                logging.warning(f"Metric {name} has an unknown key ({metric_key}), so its results will be None | Options: {METRIC_NAMES}") #TODO: Maybe this could be an error after testing
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

    def _write_into_results(self, results_filepath:str, row:list):
        # Create containing directory if it does not exist
        directory = os.path.dirname(results_filepath)
        if directory and not os.path.exists(directory): # If it does not exist
            os.makedirs(directory, exist_ok=True) # Create directory (including intermediate ones)

        # Store the row of results
        with open(results_filepath, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")        
            writer.writerow([datetime_str]+row)
    
    #endregion


    #region Metrics


    #region Precision
            
    def get_precision(self, masked_docs:List[MaskedDocument], token_weighting:TokenWeighting, 
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

    def get_TPS(self, masked_docs:List[MaskedDocument], term_alterning=TPS_TERM_ALTERNING,
                similarity_model_name:str=TPS_SIMILARITY_MODEL_NAME,
                use_chunking:bool=TPS_USE_CHUNKING, token_weighting:Optional[TokenWeighting]=None) -> Tuple[float, np.ndarray, np.ndarray]:
        tps_array = np.empty(len(masked_docs))
        similarity_array = []
        
        # Load embedding model and function for similarity
        embedding_func, embedding_model = self._get_embedding_func(similarity_model_name)

        # Default token_weighting is IC-based
        if token_weighting is None:
            token_weighting = self.ic_weighting
        
        # Process each masked document
        for i, masked_doc in enumerate(masked_docs):
            doc = self.documents[masked_doc.doc_id]

            # Get text spans
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)

            # Get IC for all spans
            spans_IC = self._get_ICs(spans, doc, token_weighting, term_alterning)

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
            tps_array[i] = masked_TIC_sim / original_TIC
        
        # Dispose embedding model and token_weighting
        if not embedding_model is None:
            del embedding_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()                

        # Get mean TPS
        tps = tps_array.mean()

        # All similarities to NumPy array
        similarity_array = np.array(similarity_array)

        return tps, tps_array, similarity_array 

    def get_TPI(self, masked_docs:List[MaskedDocument], term_alterning=TPI_TERM_ALTERNING,
            use_chunking:bool=TPI_USE_CHUNKING, token_weighting:Optional[TokenWeighting]=None) -> Tuple[float, np.ndarray, np.ndarray]:
        tpi_array = np.empty(len(masked_docs))
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
            spans_IC = self._get_ICs(spans, doc, token_weighting, term_alterning)
            
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

        return tpi, tpi_array, IC_multiplier_array

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

    def _get_ICs(self, spans:List[Tuple[int, int]], doc:Document, token_weighting:TokenWeighting, term_alterning:int) -> np.ndarray:
        #TODO: Reuse those computated before?
        spans_IC = np.empty(len(spans))
        if isinstance(term_alterning, int) and term_alterning > 1: # N Word Alterning
            # Get ICs by masking each N words, with all the document as context
            for i in range(term_alterning):
                spans_for_IC = spans[i::term_alterning]
                spans_IC[i::term_alterning] = self._get_spans_ICs(spans_for_IC, doc, token_weighting)
        
        elif isinstance(term_alterning, str) and term_alterning.lower() == "sentence": # Sentence Word Alterning
            # Get masks by masking 1 word of each sentence, with the sentence as context
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
    
    def _get_spans_ICs(self, spans: List[Tuple[int, int]], doc:Document, token_weighting: TokenWeighting, context_span=None) -> np.ndarray:
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

    def get_NMI(self, anonymizations:Dict[str, List[MaskedDocument]], k:int=NMI_K, 
                clustering_embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                clustering_embedding_size:int=NMI_EMBEDDING_SIZE,
                remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=NMI_MASKING_MARKS,
                max_pooling:bool=NMI_MAX_POOLING,
                n_clusterings:int=NMI_N_CLUSTERINGS, n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> np.ndarray:
        
        # Create the corpora
        corpora = self._get_corpora_for_NMI(anonymizations)

        # Get the embeddings
        corpora_embeddings = self._get_corpora_embeddings(corpora, clustering_embedding_model_name, clustering_embedding_size,
                                                   remove_mask_marks=remove_mask_marks, mask_marks=mask_marks,
                                                   max_pooling=max_pooling)

        # Clustering
        results, all_labels, true_intertias = self._multi_clustering_eval(corpora_embeddings, k, n_clusterings=n_clusterings,
                                                              n_tries_per_clustering=n_tries_per_clustering)
        
        # Remove result for the first corpus (ground truth defined by the original texts)
        results = results[1:]
        
        return results, all_labels, true_intertias

    def _get_corpora_for_NMI(self, anonymizations:Dict[str, List[MaskedDocument]]) -> List[List[str]]:
        # The first corpus contains the original documents, sorted by doc_id
        original_corpus_with_ids = sorted(
            [(doc.doc_id, doc.text) for doc in self.documents.values()],
            key=lambda x: x[0]
        )
        corpora_with_ids = [original_corpus_with_ids]

        # Iterate through each anonymization method to create masked corpora
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
                                 clustering_embedding_size:int=NMI_EMBEDDING_SIZE, #TODO: embedding size not required for this method if using SentenceTransformers
                                 remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=NMI_MASKING_MARKS,
                                 max_pooling:bool=NMI_MAX_POOLING, device:str=DEVICE) -> List[np.ndarray]:
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
    
    def _get_corpora_embeddings_original(self, corpora:List[List[str]], clustering_embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                                 clustering_embedding_size:int=NMI_EMBEDDING_SIZE,
                                 remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=NMI_MASKING_MARKS,
                                 max_pooling:bool=NMI_MAX_POOLING, device:str=DEVICE) -> List[np.ndarray]:
        corpora_embeddings = []

        # Create BERT-based model and tokenizer
        model = AutoModel.from_pretrained(clustering_embedding_model_name, output_hidden_states=True) # The model returns all hidden-states.
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(clustering_embedding_model_name)
        
        # Collect embeddings
        mask_marks_re_pattern = "|".join([m.upper() for m in mask_marks])
        for corpus in tqdm(corpora, desc="Extracting embeddings"):
            # Remove mask marks if required
            if remove_mask_marks:
                corpus = [re.sub(mask_marks_re_pattern, "", text).strip() for text in corpus]
            
            corpus_embeddings = model.encode(corpus,
                                             convert_to_numpy=True,
                                             show_progress_bar=False)

            corpus_embeddings = np.empty((len(corpus), clustering_embedding_size))
            for idx, text in enumerate(corpus):
                corpus_embeddings[idx] = self._get_text_embedding(text, model, tokenizer, max_pooling=max_pooling)

            corpora_embeddings.append(corpus_embeddings)
        
        # Dipose model and tokenizer
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return corpora_embeddings
    
    def _get_text_embedding_original(self, text:str, model, tokenizer, max_pooling:bool=NMI_MAX_POOLING, device:str=DEVICE): #TODO: Check if use this
        tokens = tokenizer.encode(text, truncation=False, padding='max_length', add_special_tokens=True, return_tensors="pt")
        tokens = tokens.to(DEVICE)
        overlap_span = None

        # If longer than model max length, create multiple inputs
        len_multiplier = tokens.shape[1] / tokenizer.model_max_length
        if len_multiplier > 1:
            n_inputs = int(len_multiplier) + 1
            new_tokens = torch.empty((n_inputs, tokenizer.model_max_length), device=device, dtype=int)

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
        if not overlap_span is None:  # Remove overlap from last block
            idxs = list(range(len(outputs)))
            idxs = idxs[:overlap_span[0]] + idxs[overlap_span[1]:]
            outputs = outputs[idxs]

        # Apply max pooling (https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00564-9) or mean pooling
        if max_pooling:
            embeddings = outputs.max(axis=0)
        else:
            embeddings = outputs.mean(axis=0)

        return embeddings

    def _get_text_embedding_attention_mask(self, text:str, model, tokenizer, max_pooling:bool=NMI_MAX_POOLING, device:str=DEVICE): #TODO: Check if use this
        encoded_input = tokenizer.encode_plus(
            text,
            truncation=False, # Truncation will be handled manually if len_multiplier > 1
            padding='longest', # 'longest' or 'max_length' are both fine, 'max_length' is explicit
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True # Explicitly request attention mask
        )

        tokens = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        overlap_span = None
        # If longer than model max length, create multiple inputs
        len_multiplier = tokens.shape[1] / tokenizer.model_max_length
        if len_multiplier > 1:
            n_inputs = int(len_multiplier) + 1
            new_tokens = torch.empty((n_inputs, tokenizer.model_max_length), device=device, dtype=int)
            new_attention_mask = torch.empty((n_inputs, tokenizer.model_max_length), device=device, dtype=int)


            ini = 0
            for i in range(n_inputs):
                end = ini + tokenizer.model_max_length
                if end >= tokens.shape[1]:  # Last block
                    overlap_span = (tokens.shape[1] - tokenizer.model_max_length, ini) # Span that will be processed twice
                    end = tokens.shape[1]
                    ini = end - tokenizer.model_max_length
                new_tokens[i, :] = tokens[0, ini:end]
                new_attention_mask[i, :] = attention_mask[0, ini:end] # Also slice attention mask
            tokens = new_tokens
            attention_mask = new_attention_mask # Update attention_mask for multi-input case


        # Predict
        with torch.no_grad():
            # Pass attention_mask to the model
            outputs = model(tokens, attention_mask=attention_mask)
            outputs = outputs[0].cpu()

        # Get embedding
        outputs = outputs.reshape((-1, outputs.shape[-1]))
        if not overlap_span is None:  # Remove overlap from last block
            idxs = list(range(len(outputs)))
            idxs = idxs[:overlap_span[0]] + idxs[overlap_span[1]:]
            outputs = outputs[idxs]

        # Apply max pooling (https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00564-9) or mean pooling
        if max_pooling:
            embeddings = outputs.max(axis=0).values # .values to get the tensor, not a named tuple
        else:
            embeddings = outputs.mean(axis=0)

        return embeddings

    #endregion

    #region Clustering

    def _multi_clustering_eval(self, corpora_embeddings:List[np.ndarray], k:int, n_clusterings:int=NMI_N_CLUSTERINGS,
                                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, np.ndarray]:
        results = np.empty((n_clusterings, len(corpora_embeddings)))
        true_intertias = np.empty(n_clusterings)
        for clustering_idx in tqdm(range(n_clusterings), desc="Clustering"):
            true_labels, corpora_labels, true_intertias[clustering_idx] = self._get_all_clusterings(corpora_embeddings, k,
                                                                                                    tries_per_clustering=n_tries_per_clustering)
            results[clustering_idx, :] = self._compare_clusterings(true_labels, corpora_labels, normalized_mutual_info_score)

        # Average for the n_clusterings
        results = results.mean(axis=0)

        return results, corpora_labels, true_intertias

    def _get_all_clusterings(self, corpora_embeddings:List[np.ndarray], k:int, tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, List[np.ndarray]]:
        corpora_labels = []

        # First corpus corresponds to the ground truth
        true_labels, true_inertia = self._clusterize(corpora_embeddings[0], k, tries=tries_per_clustering)

        # Clusterize for each corpus
        for corpus_embeddings in corpora_embeddings: # Repeating for the first one (ground truth) allows to check consistency
            labels, inertia = self._clusterize(corpus_embeddings, k, tries=tries_per_clustering)
            corpora_labels.append(labels)

        return true_labels, corpora_labels, true_inertia

    def _clusterize(self, corpus_embeddings, k:int, tries:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, float]:
        inertia = 0
        kmeanspp = KMeans(n_clusters=k, init='k-means++', n_init=tries)
        labels = kmeanspp.fit_predict(corpus_embeddings)
        inertia = kmeanspp.inertia_        
        return labels, inertia

    def _compare_clusterings(self, true_labels:np.ndarray, corpora_labels:List[np.ndarray], eval_metric) -> np.ndarray:
        metrics = np.empty(len(corpora_labels))
        for idx, corpus_labels in enumerate(corpora_labels):
            metric = eval_metric(corpus_labels, true_labels)
            metrics[idx] = metric
        return metrics

    #endregion

    #endregion


    #region TRIR

    def get_TRIR(self, masked_docs:List[MaskedDocument]):
        return 0 #TODO

    #endregion


    #endregion

#endregion


#endregion


#region Main

if __name__ == "__main__":

    #region Arguments parsing

    parser = argparse.ArgumentParser(description='Computes evaluation metrics for text anonymization')
    parser.add_argument('config_filepath', type=str,
                        help='the path to the JSON file containing the configuration for the evaluation')
    args = parser.parse_args()

    # Load configuration dictionary
    with open(args.config_filepath, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    for key in MANDATORY_CONFIG_KEYS:
        if not key in config.keys():
            raise RuntimeError(f"Configuration JSON file is not well formed: Missing mandatory key {key}")
    
    #endregion


    #region Initialization

    # Create TAE from corpus
    corpus_filepath = config[CORPUS_CONFIG_KEY]
    tae = TAE.from_filepath(corpus_filepath)

    # Create anonymizations
    anonymizations = {}
    anonymizations_config = config[ANONYMIZATIONS_CONFIG_KEY]
    for anon_name, anon_filepath in anonymizations_config.items():
        masked_docs = MaskedCorpus(anon_filepath)
        anonymizations[anon_name] = masked_docs
    
    # Get metrics
    metrics = config[METRICS_CONFIG_KEY]

    # Get filepath for results CSV file
    results_filepath = config[RESULTS_CONFIG_KEY]

    #endregion


    #region Evaluate

    tae.evaluate(anonymizations, metrics, results_filepath)

    #endregion


#endregion