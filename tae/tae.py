#region Imports

import json, re, abc, logging, math, os, csv, gc
os.environ["OMP_NUM_THREADS"] = "1" # Done before loading MKL to avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1
from typing import Dict, List, Tuple, Optional, Set, Iterator, Union
from datetime import datetime
from dataclasses import dataclass
from functools import partial

from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import spacy
import intervaltree

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from .tri import TRI

#endregion


#region Constants/Settings #TODO: Move (some) of these constants to the TAE class?


#region Input

# Configuration dictionary keys
CORPUS_CONFIG_KEY = "corpus_file_path" #TODO: Perhaps also accept a HuggingFace's dataset
ANONYMIZATIONS_CONFIG_KEY = "anonymizations"
RESULTS_CONFIG_KEY = "results_file_path"
METRICS_CONFIG_KEY = "metrics"
MANDATORY_CONFIG_KEYS = [CORPUS_CONFIG_KEY, ANONYMIZATIONS_CONFIG_KEY, RESULTS_CONFIG_KEY, METRICS_CONFIG_KEY]

# Corpus dictionary keys
DOC_ID_KEY = "doc_id"
ORIGINAL_TEXT_KEY = "text"
MANDATORY_CORPUS_KEYS = [DOC_ID_KEY, ORIGINAL_TEXT_KEY]
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
WEIGHTED_PRECISION_METRIC_NAME = "WeightedPrecision"
RECALL_METRIC_NAME = "Recall"
RECALL_PER_ENTITY_METRIC_NAME = "RecallPerEntity"
TPI_METRIC_NAME = "TPI"
TPS_METRIC_NAME = "TPS"
NMI_METRIC_NAME = "NMI"
TRIR_METRIC_NAME = "TRIR"
METRIC_NAMES = [PRECISION_METRIC_NAME, WEIGHTED_PRECISION_METRIC_NAME, RECALL_METRIC_NAME, RECALL_PER_ENTITY_METRIC_NAME, TPI_METRIC_NAME, TPS_METRIC_NAME, NMI_METRIC_NAME, TRIR_METRIC_NAME]
METRICS_REQUIRING_GOLD_ANNOTATIONS = [PRECISION_METRIC_NAME, WEIGHTED_PRECISION_METRIC_NAME, RECALL_METRIC_NAME, RECALL_PER_ENTITY_METRIC_NAME]

#endregion


#region General

SPACY_MODEL_NAME = "en_core_web_md"
IC_WEIGHTING_MODEL_NAME = "google-bert/bert-base-uncased" #TODO: Update this and all other models
IC_WEIGHTING_MAX_SEGMENT_LENGTH = 100
BACKGROUND_KNOWLEDGE_KEY = "background_knowledge" # For TRIR background knowledge file

# POS tags, tokens or characters that can be ignored scores 
# (because they do not carry much semantic content, and there are discrepancies
# on whether to include them in the annotated spans or not)
POS_TO_IGNORE = {"ADP", "PART", "CCONJ", "DET"} 
TOKENS_TO_IGNORE = {"mr", "mrs", "ms", "no", "nr", "about"}
CHARACTERS_TO_IGNORE = " ,.-;:/&()[]–'\" ’“”"

MASKING_MARKS = ["SENSITIVE", "PERSON", "DEM", "LOC",
                 "ORG", "DATETIME", "QUANTITY", "MISC",
                 "NORP", "FAC", "GPE", "PRODUCT", "EVENT",
                 "WORK_OF_ART", "LAW", "LANGUAGE", "DATE",
                 "TIME", "ORDINAL", "CARDINAL", "DATE_TIME", "DATETIME",
                 "NRP", "LOCATION", "ORGANIZATION", "\*\*\*"]

# Check for GPU with CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    DEVICE = torch.device("cpu")

#endregion


#region Metric-specific

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

# NMI default settings
NMI_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" #TODO: Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2" others from https://www.sbert.net/docs/sentence_transformer/pretrained_models.html or classic models such as "bert-base-cased"
NMI_MIN_K = 2
NMI_MAX_K = 32
NMI_K_MULTIPLIER = 2
NMI_REMOVE_MASK_MARKS = True
NMI_N_CLUSTERINGS = 5
NMI_N_TRIES_PER_CLUSTERING = 50

# TRIR default settings are defined in the TRI class (in tri.py file)

#endregion


#endregion


#region Utils

@dataclass
class MaskedDocument:
    """Represents a document in which some text spans are masked, each span
    being expressed by their (start, end) character boundaries.
    Optionally, spans can also have replacement strings.
    """

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
        """Applies masking to the original text based on masked spans and replacements.

        Args:
            original_text (str): The text to be masked.
        
        Returns:
            str: The masked text.
        """

        masked_text = ""+original_text
        
        for (start_idx, end_idx), replacement in zip(reversed(self.masked_spans), reversed(self.replacements)):
            if replacement is None: # If there is no replacement, use first masking mark
                replacement = MASKING_MARKS[0]
            masked_text = masked_text[:start_idx] + replacement + masked_text[end_idx:]
        
        return masked_text

@dataclass
class MaskedCorpus(List[MaskedDocument]):
    """Auxiliar class that inherits from List[MaskedDocument] for the loading of a MaskedDocument list from file"""

    def __init__(self, masked_docs_file_path:str):
        """
        Initializes the `MaskedCorpus` object from a JSON file.

        Args:
            masked_docs_file_path (str): Path to a JSON file with masked spans (and replacements, optionally)
        """

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
        """Checks that direct identifiers are masked"""

        if self.is_direct and not self.need_masking:
            raise RuntimeError(f"Annotated entity {self.entity_id} is a direct identifier but it is not always masked")

    @property
    def mentions_to_mask(self) -> list:
        """List of mentions to mask based on the mention level masking"""

        return [mention for i, mention in enumerate(self.mentions)
                if self.mention_level_masking[i]]

class Document:
    """Representation of a document, with an identifier and textual content. 
    Ooptionally, it can include its spaCy document object and/or gold annotations"""

    doc_id:str
    text:str
    spacy_doc:spacy.tokens.Doc
    gold_annotated_entities:Dict[str, AnnotatedEntity]

    #region Initialization
    
    def __init__(self, doc_id:str, text:str, spacy_doc:Optional[spacy.tokens.Doc]=None,
                 gold_annotations:Optional[Dict[str,List]]=None):
        """
        Initializes a new `Document`, optionally including gold annotations.

        Args:
            doc_id (str): The unique document identifier.
            text (str): The text content of the document.
            spacy_doc (Optional[spacy.tokens.Doc]): The spaCy document object.
            gold_annotations (Optional[Dict[str, List]]): Gold annotations, if available.
                Check the `README.md` for more information.
        """

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
        """
        Processes a list of entity mentions and returns a list of unique AnnotatedEntity objects.

        Args:
            entity_mentions (List[dict]): A list of dictionaries, where each dictionary represents an entity mention.
                Each mention dictionary must contain `entity_id`, `identifier_type`, `start_offset`, and `end_offset` keys.

        Returns:
            List[AnnotatedEntity]: A list of AnnotatedEntity objects, where each object represents a unique entity
            found in the input mentions, consolidating all its mentions.
        """

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
        """
        Given a document with a set of masked text spans, determines whether entity
        is fully masked (which means that all its mentions are masked).

        Args:
            masked_doc (MaskedDocument): The document with masked text spans.
            entity (AnnotatedEntity): The entity to check for masking.

        Returns:
            bool: True if the entity is fully masked, False otherwise.
        """

        for incr, (mention_start, mention_end) in enumerate(entity.mentions):
            
            if self.is_mention_masked(masked_doc, mention_start, mention_end):
                continue
            
            # The masking is sometimes inconsistent for the same entity, 
            # so we verify that the mention does need masking
            elif entity.mention_level_masking[incr]:
                return False
        
        return True
    
    def is_mention_masked(self, masked_doc:MaskedDocument, mention_start:int, mention_end:int) -> bool:
        """
        Given a document with a set of masked text spans and a particular mention span,
        determine whether the mention is fully masked (taking into account special
        characters or PoS/tokens to ignore).

        Args:
            masked_doc (MaskedDocument): The document with masked text spans.
            mention_start (int): The starting character offset of the mention.
            mention_end (int): The ending character offset of the mention.

        Returns:
            bool: True if the mention is fully masked, False otherwise.
        """

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

    def get_entities_to_mask(self, include_direct:bool=True, include_quasi:bool=True) -> List[AnnotatedEntity]:
        """Return entities that should be masked and satisfy the constraints specified as arguments.

        Args:
            include_direct (bool): Whether to include direct entities. Defaults to True.
            include_quasi (bool): Whether to include quasi entities. Defaults to True.

        Returns:
            List[AnnotatedEntity]: A list of entities that should be masked.
        """
        
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
        
    def get_annotators_for_span(self, start_token:int, end_token:int) -> Set[str]:
        """Given a text span (typically for a token), determines which annotators
        have also decided to mask it.

        Args:
            start_token (int): The starting token index of the span.
            end_token (int): The ending token index of the span.

        Returns:
            Set[str]: A (possibly empty) set of annotator names that have masked that span.
        """
        
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

    def split_by_tokens(self, start:int, end:int) -> Iterator[Tuple[int, int]]:
        """
        Generates the (start, end) boundaries of each token included in this span.

        Args:
            start (int): The starting index of the span.
            end (int): The ending index of the span.

        Returns:
            Iterator[Tuple[int, int]]: An iterator of (start, end) tuples for each token.
        """    

        for match in re.finditer(r"\w+", self.text[start:end]):
            start_token = start + match.start(0)
            end_token = start + match.end(0)
            yield start_token, end_token

    #endregion

class TokenWeighting:
    """Abstract class for token weighting schemes (i.e., `ICTokenWeighting` and `UniformTokenWeighting`)"""

    @abc.abstractmethod
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        """Given a text and a list of text spans, returns a NumPy array of numeric weights
        (of same length as the list of spans) corresponding to each span.

        Args:
            text (str): The input text.
            text_spans (List[Tuple[int,int]]): A list of text spans, where each span
                is represented as a tuple of (start_index, end_index).

        Returns:
            np.ndarray: A NumPy array of numeric weights, with the same length as
            `text_spans`.
        """

        return

class ICTokenWeighting(TokenWeighting):
    """Token weighting based on a BERT language model. 
    The weighting mechanism runs the model on a text in which the provided spans are masked. 
    The weight of each token is then defined as its information content:
    -log(probability of the actual token value)
    
    In other words, a token that is difficult to predict will have a high
    information content, and therefore a high weight, whereas a token which can
    be predicted from its content will received a low weight (closer to zero)"""

    max_segment_length:int
    model_name:str
    device:str

    model=None
    tokenizer=None
    
    def __init__(self, model_name:str, device:str, max_segment_length:int):
        """Initializes the `ICTokenWeighting`

        Args:
            model_name (str): The name of the BERT model to use (e.g., "bert-base-uncased").
            device (str): The device to run the model on (e.g., "cpu" or "cuda").
            max_segment_length (int): The maximum sequence length for the model.
        """

        self.max_segment_length = max_segment_length
        self.model_name = model_name
        self.device = device
    
    # TODO: Implement a batched version
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        """Returns an array of numeric information content weights, where each value
        corresponds to -log(probability of predicting the value of the text span
        according to the BERT model).

        If the span corresponds to several BERT tokens, the probability is the
        minimum of the probabilities for each token.

        Args:
            text (str): The input text.
            text_spans (List[Tuple[int,int]]): A list of text spans, where each span
                is represented as a tuple of (start_index, end_index).

        Returns:
            np.ndarray: A NumPy array of numeric weights, with the same length as
            `text_spans`. A weight close to 0 represents a span with low information
            content (i.e. which can be easily predicted from the remaining context),
            while a higher weight represents a high information content (which is
            difficult to predict from the context).
        """

        # STEP 0: Create model if it is not already created
        if self.model is None:
            self._create_model()
        
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

    def _create_model(self):
        """
        Initializes the BERT model and tokenizer from the pre-trained model name.
        Only executed the first time the get_weights method is invoked.
        """

        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _get_tokens_by_span(self, bert_token_spans:List[Tuple[int,int]],
                            text_spans:List[Tuple[int,int]], text:str) -> Dict[Tuple[int,int],List[int]]:
        """Given two lists of spans (one with the spans of the BERT tokens, and one with
        the text spans to weight), returns a dictionary where each text span is associated
        with the indices of the BERT tokens it corresponds to.

        Args:
            bert_token_spans (List[Tuple[int,int]]): A list of tuples, where each tuple
                represents the (start_index, end_index) of a BERT token within the text.
            text_spans (List[Tuple[int,int]]): A list of tuples, where each tuple
                represents the (start_index, end_index) of a text span to be weighted.
            text (str): The original text.

        Returns:
            Dict[Tuple[int,int],List[int]]: A dictionary where keys are text spans
            (tuples of start and end indices) and values are lists of BERT token indices
            that fall within the respective text span.
        """
        
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
    
    def _get_model_predictions(self, input_ids:Union[List[int], torch.Tensor],
                                attention_mask:Union[List[int], torch.Tensor]) -> torch.Tensor:
        """Given tokenised input identifiers and an associated attention mask (where the
        tokens to predict have a mask value set to 0), runs the BERT language and returns
        the (unnormalised) prediction scores for each token.

        If the input length is longer than max_segment_length, we split the document in
        small segments, and then concatenate the model predictions for each segment.

        Args:
            input_ids (Union[List[int], torch.Tensor]): The input token IDs.
            attention_mask (Union[List[int], torch.Tensor]): The attention mask, where
                0 indicates tokens to be predicted.

        Returns:
            torch.Tensor: The (unnormalised) prediction scores for each token.
        """
        
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
        """Method invoked when deleting the instance to dispose the model and the tokenizer
        (if these are already defined)"""

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
        """Given a text and a list of text spans, returns a NumPy array of uniform weights.

        Args:
            text (str): The input text.
            text_spans (List[Tuple[int,int]]): A list of text spans, where each span
                is represented as a tuple of (start_index, end_index).

        Returns:
            np.ndarray: A NumPy array with all weights set to 1.0, with the same length
                as `text_spans`.
        """

        return np.ones(len(text_spans))

#endregion


#region TAE

# TODO: Check all input and output typing (specially for private methods)
class TAE:
    """Text Anonymization Evaluator (TAE) class, defined for the utility and privacy assessment of a text anonymization corpus.
    It is instanciated for a particular corpus, and provides functions for several evaluation metrics.
    Optionally, the corpus can include gold annotations, used for precision and recall metrics."""

    #region Attributes

    documents:Dict[str, Document]
    spacy_nlp=None
    gold_annotations_ratio:int
    metrics_funcs:dict

    #endregion


    #region Initialization
    
    def __init__(self, corpus:List[Document], spacy_model_name:str=SPACY_MODEL_NAME):
        """
        Initializes the `TAE` with a given corpus and spaCy model.

        Args:
            corpus (List[Document]): The list of documents to be evaluated.
            spacy_model_name (str): The name of the spaCy model to load.
        """

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
            spacy_doc = self.spacy_nlp(doc[ORIGINAL_TEXT_KEY])

            # Get gold annotations (if present)
            gold_annotations = doc.get(GOLD_ANNOTATIONS_KEY, None)
            
            # Creating the actual document (identifier, text and gold annotations)
            new_doc = Document(doc[DOC_ID_KEY], doc[ORIGINAL_TEXT_KEY], spacy_doc, gold_annotations)
            self.documents[doc[DOC_ID_KEY]] = new_doc
            if len(new_doc.gold_annotated_entities) > 0:
                n_docs_with_annotations += 1
        
        # Notify the number and percentage of annotated documents
        self.gold_annotations_ratio = n_docs_with_annotations / len(self.documents)
        logging.info(f"Number of gold annotated documents: {n_docs_with_annotations} ({self.gold_annotations_ratio:.3%})")

        # Create dictionary of metric functions (used in _get_partial_metric_func)
        self.metrics_funcs = {PRECISION_METRIC_NAME:self.get_precision,
                              WEIGHTED_PRECISION_METRIC_NAME:self.get_weighted_precision,
                              RECALL_METRIC_NAME:self.get_recall,
                              RECALL_PER_ENTITY_METRIC_NAME:self.get_recall_per_entity_type,
                              TPI_METRIC_NAME:self.get_TPI,
                              TPS_METRIC_NAME:self.get_TPS,
                              NMI_METRIC_NAME:self.get_NMI,
                              TRIR_METRIC_NAME:self.get_TRIR}

    @classmethod
    def from_file_path(self, corpus_file_path:str, spacy_model_name:str=SPACY_MODEL_NAME):
        """
        Initializes the `TAE` from a JSON corpus file.

        Args:
            corpus_file_path (str): The path to the corpus JSON file.
            spacy_model_name (str): The name of the spaCy model to load.

        Returns:
            TAE: An instance of the TAE class.
        """
        with open(corpus_file_path, encoding="utf-8") as f:
            corpus = json.load(f)
        if type(corpus)!=list:
            raise RuntimeError("Corpus JSON file must be a list of documents")
        return TAE(corpus, spacy_model_name=spacy_model_name)

    #endregion


    #region Evaluation

    def evaluate(self, anonymizations:Dict[str, List[MaskedDocument]], metrics:Dict[str,dict], results_file_path:Optional[str]=None) -> dict:
        """
        Evaluates multiple anonymizations based on the specified metrics.

        Args:
            anonymizations (Dict[str, List[MaskedDocument]]): A dictionary where keys are anonymization names
                                                                and values are lists of masked documents.
            metrics (Dict[str, dict]): A dictionary where keys are metric names and values are their parameters.
            Metric names are splitted by underscores ("_"). The string before the first underscore must be one of those present in `METRIC_NAMES`.
            results_file_path (Optional[str]): The path to a file where results will be written.

        Returns:
            dict: A dictionary containing the evaluation results for each metric and anonymization.
        """
        
        results = {}

        # Initial checks
        self._eval_checks(anonymizations, metrics)

        # Write results file header
        if results_file_path:
            self._write_into_results(results_file_path, ["Metric/Anonymization"]+list(anonymizations.keys()))

        # For each metric
        for metric_name, metric_parameters in metrics.items():
            logging.info(f"########################### Computing {metric_name} metric ###########################")
            metric_key = metric_name.split("_")[0] # Text before first underscore is name of the metric, the rest is freely used
            partial_eval_func = self._get_partial_metric_func(metric_key, metric_parameters)
            #TODO: If any error/exception happens, notify and skip to the next metric

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
                        #logging.info(f"{metric_name} for {anon_name}: {metric_results[anon_name]}") #TODO: Check if remove this or store the results after each metric
            
            # Save results
            results[metric_name] = metric_results
            if not partial_eval_func is None: # TODO: Check if preserve this if
                if not results_file_path is None: #TODO: CSV is maybe a bad format for complex results such as those from recall_per_entity_type. Is JSON better instead? (worse for Excel)
                    self._write_into_results(results_file_path, [metric_name]+list(metric_results.values()))
            
            # Show results all together for easy comparison
            msg = f"Results for {metric_name}:"
            for name, value in results[metric_name].items():
                msg += f"\n\t\t\t\t\t{name}: {value}" #TODO: Check these tabs
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
                raise RuntimeError(f"Metric {name} depends on gold annotations, but these are not present for all documents (only for a {self.gold_annotations_ratio:.3%})")

    def _get_partial_metric_func(self, metric_name:str, parameters:dict) -> Optional[partial]:
        func = self.metrics_funcs.get(metric_name, None)
        partial_func = None if func is None else partial(func, **parameters)
        return partial_func # Result would be None if name is invalid

    def _write_into_results(self, results_file_path:str, values:list):
        # Create containing directory if it does not exist
        directory = os.path.dirname(results_file_path)
        if directory and not os.path.exists(directory): # If it does not exist
            os.makedirs(directory, exist_ok=True) # Create directory (including intermediate ones)

        # Store the row of results
        with open(results_file_path, "a+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([datetime_str]+values)
    
    #endregion


    #region Privacy metrics


    #region Recall

    def get_recall(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                    include_quasi:bool=RECALL_INCLUDE_QUASI, token_level:bool=RECALL_TOKEN_LEVEL) -> float:
        """
        Returns the mention or token-level recall of the masked spans when compared to the gold annotations. 
        This metric is used to assess privacy protection.
        This metric implementation was initially presented in [The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization](https://aclanthology.org/2022.cl-4.19/) (Pilán et al., CL 2022)

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            include_direct (bool): Whether to include direct identifiers in the metric.
            include_quasi (bool): Whether to include quasi identifiers in the metric.
            token_level (bool): Whether to compute the recall at the level of tokens or mentions.

        Returns:
            float: The recall score.
        """

        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        nb_masked_elements = sum(nb_masked_by_type.values())
        nb_elements = sum(nb_by_type.values())
                
        try:
            return nb_masked_elements / nb_elements
        except ZeroDivisionError:
            return 0
    
    def get_recall_per_entity_type(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                                   include_quasi:bool=RECALL_INCLUDE_QUASI, token_level:bool=RECALL_TOKEN_LEVEL) -> Dict[str,float]:
        """
        Returns the mention or token-level recall of the masked spans when compared
        to the gold annotations, and factored by entity type.
        This metric implementation was initially presented in [The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization](https://aclanthology.org/2022.cl-4.19/) (Pilán et al., CL 2022)

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the system.
            include_direct (bool): Whether to include direct identifiers in the metric.
            include_quasi (bool): Whether to include quasi identifiers in the metric.
            token_level (bool): Whether to compute the recall at the level of tokens or mentions.

        Returns:
            dict: A dictionary where keys are entity types and values are their corresponding recall scores.
        """
        
        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        return {ent_type:nb_masked_by_type[ent_type]/nb_by_type[ent_type]
                for ent_type in nb_by_type}
                
    def _get_mask_counts(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                                   include_quasi:bool=RECALL_INCLUDE_QUASI,
                                   token_level:bool=RECALL_TOKEN_LEVEL) -> Tuple[Dict[str,int],Dict[str,int]]:
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


    #region TRIR

    def get_TRIR(self, anonymizations:Dict[str, List[MaskedDocument]],
                 background_knowledge_file_path:str, output_folder_path:str, #TODO: Maybe, autogenerate output_folder_path
                 verbose:bool=True, **kwargs) -> Dict[str, float]: #TODO: Add verbose to each metric
        """
        Calculates the Text Re-Identification Risk (TRIR) for given anonymizations, simulating a re-identification attack on the same basis as record linkage.
        This metric is used to empirically assess privacy protection.
        This metric was proposed in [Evaluating the disclosure risk of anonymized documents via a machine learning-based re-identification attack](https://link.springer.com/article/10.1007/s10618-024-01066-3) (Manzanares-Salor et al., DAMI 2024)

        Args:
            anonymizations (Dict[str, List[MaskedDocument]]): A dictionary where keys are anonymization names
                                                                and values are lists of masked documents.
            background_knowledge_file_path (str): Path to the background knowledge JSON file.
            output_folder_path (str): Path to the folder where TRI outputs (e.g., trained model) will be stored.
            verbose (bool): Whether to print verbose output during execution.
            **kwargs: Additional keyword arguments to be passed to the TRI class constructor. Check `README.md` or the [Text Re-Identification repository](https://github.com/BenetManzanaresSalor/TextRe-Identification) for more information.

        Returns:
            dict: A dictionary where keys are anonymization names and values are their TRIR scores.
        """
        
        # Load corpora
        corpora = self._get_anonymization_corpora(anonymizations)

        # Load background knowledge and add it to the corpora
        with open(background_knowledge_file_path, "r", encoding="utf-8") as f:
            bk_dict = json.load(f)        
        for doc_id, bk in bk_dict.items():
            doc_dict = corpora.get(doc_id, {})
            doc_dict[BACKGROUND_KNOWLEDGE_KEY] = bk
            corpora[doc_id] = doc_dict #TODO: Test with BK that are supersets

        # Create dataframe from corpora
        dataframe = pd.DataFrame.from_dict(list(corpora.values())) #TODO: Why results change when compared with original TRI?
        #dataframe.to_json("dataframe.json", orient="records") #TODO: Remove this

        # Create and run TRI
        tri = TRI(
            dataframe=dataframe,
            background_knowledge_column=BACKGROUND_KNOWLEDGE_KEY,
            output_folder_path=output_folder_path,
            individual_name_column=DOC_ID_KEY,
            **kwargs)        
        results = tri.run(verbose=verbose)

        # Obtain TRIR
        results = {anon_name:values["eval_Accuracy"] for anon_name, values in results.items()}

        return results

    #endregion


    #endregion


    #region Utility metrics


    #region Precision
    
    #TODO: Make a per-entity version
    def get_precision(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=None,
                      weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH,
                      token_level:bool=PRECISION_TOKEN_LEVEL) -> float:
        """
        Returns the precision of the masked spans when compared to the gold annotations.
        This metric is used to assess utility preservation.
        Optionally, it can be weighted considering information content (with `ICTokenWeighting`) 
        and be computed at token-level or at mention-level.
        If annotations from several annotators are available for a given document,
        the precision corresponds to a micro-average over the annotators.
        This metric implementation was initially presented in [The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization](https://aclanthology.org/2022.cl-4.19/) (Pilán et al., CL 2022)

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            weighting_model_name (Optional[str]): Name of the model for `ICTokenWeighting`. If none, uniform token is used.
            weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`.
            token_level (bool): If token_level is set to True, the precision is computed at the level of tokens, 
                                otherwise the precision is at the mention-level.

        Returns:
            float: The precision score.
        """
        
        weighted_true_positives = 0.0
        weighted_system_masks = 0.0

        # Define token weighting
        if weighting_model_name is None:
            token_weighting = UniformTokenWeighting()
        
        else:
            token_weighting = ICTokenWeighting(model_name=weighting_model_name, device=DEVICE,
                                               max_segment_length=weighting_max_segment_length)
        
        # For each masked document
        for doc in masked_docs:
            gold_doc = self.documents[doc.doc_id]
            
            # We extract the list of spans (token- or mention-level)
            anonymization_masks = []
            for start, end in doc.masked_spans:
                if token_level:
                    anonymization_masks += list(gold_doc.split_by_tokens(start, end))
                else:
                    anonymization_masks += [(start,end)]
            
            # We compute the weights (information content) of each mask
            weights = token_weighting.get_weights(gold_doc.text, anonymization_masks)
            
            # We store the number of annotators in the gold standard document
            nb_annotators = len(set(entity.annotator for entity in gold_doc.gold_annotated_entities.values()))
            
            for (start, end), weight in zip(anonymization_masks, weights):
                
                # We extract the annotators that have also masked this token/span
                annotators = gold_doc.get_annotators_for_span(start, end)
                
                # And update the (weighted) counts
                weighted_true_positives += (len(annotators) * weight)
                weighted_system_masks += (nb_annotators * weight)
        
        # Dispose token weighting
        del token_weighting

        # Return results
        try:
            return weighted_true_positives / weighted_system_masks
        except ZeroDivisionError:
            return 0

    def get_weighted_precision(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
                      weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH,
                      token_level:bool=PRECISION_TOKEN_LEVEL) -> float:
        """
        Returns the precision of the masked spans, with Information Content (IC) weighting by default.
        This defines a wrapper around the `get_precision` method for avoiding the need to select the `weighting_model_name` for IC weighting.

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            weighting_model_name (Optional[str]): Name of the model for `ICTokenWeighting`. Defaults to `IC_WEIGHTING_MODEL_NAME`.
            weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`.
            token_level (bool): If token_level is set to True, the precision is computed at the level of tokens,
                                otherwise the precision is at the mention-level.
        """
        return self.get_precision(masked_docs, weighting_model_name=weighting_model_name,
                      weighting_max_segment_length=weighting_max_segment_length,
                      token_level=token_level)

    #endregion
    

    #region TPS and TPI
    
    def get_TPI(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
            weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH, 
            term_alterning:Union[int,str]=TPI_TERM_ALTERNING, use_chunking:bool=TPI_USE_CHUNKING,
            ICs_dict:Optional[Dict[str,np.ndarray]]=None) -> Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
        """
        Text Preserved Information (TPI) measures the percentage of information content (IC) still present in the masked documents.
        This metric is used to assess utility preservation. It employs `ICTokenWeighting` for measuring IC.
        It can be seen as an simplified/ablated version of Text Preserved Similarity (TPS), without taking into account replacements and their similarities.

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            weighting_model_name (Optional[str]): Name of the model for `ICTokenWeighting`. If None, uniform weighting is used (not intended for this metric).
            weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`.
            term_alterning (Union[int,str]): Parameter for term alternation in IC calculation.
                It can be an integer (e.g., N = 6) or the string "sentence." 
                When using an integer N, one of the N terms will be masked each round.
                A larger N value implies a more accurate IC estimation (up to a certain point), but slower computation because more rounds are required.
                If "sentence" is used, the text will be split into sentences, and one of the sentence terms will be masked at each round.
                This approach is significantly slower but may provide the most accurate IC estimation.
            use_chunking (bool): Whether to use chunking for term span extraction. It is recommended for a more precise IC calculation.
            ICs_dict (Optional[Dict[str,np.ndarray]]): Precomputed IC values for documents. 
            Used in `evaluate` to avoid recomputing, for each anonymization, the original document's ICs (which are always identical).

        Returns:
            Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
                - float: The average TPI for the corpus.
                - np.ndarray: An array of TPI values for each document.
                - Dict[str,np.ndarray]: A dictionary containing precomputed ICs (used for caching).
                - np.ndarray: An array of IC multipliers (i.e., IC of masked terms divided by IC of non-masked terms) for each document.
        """

        # Initialize outputs
        tpi_array = np.empty(len(masked_docs))
        if ICs_dict is None:
            ICs_dict = {}
        IC_multiplier_array = np.empty(len(masked_docs))

        # Define token weighting
        if weighting_model_name is None:
            token_weighting = UniformTokenWeighting()        
        else:
            token_weighting = ICTokenWeighting(model_name=weighting_model_name, device=DEVICE,
                                               max_segment_length=weighting_max_segment_length)

        # For each masked document
        for i, masked_doc in enumerate(masked_docs):
            doc = self.documents[masked_doc.doc_id]

            # Get terms spans and mask
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)
            masked_spans = self._filter_masked_spans(doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, masked_spans) # Non-masked=True(1), Masked=False(0)

            # Get IC for all spans
            if masked_doc.doc_id in ICs_dict:
                spans_IC = ICs_dict[masked_doc.doc_id] # Use precomputed ICs
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

        # Dispose token weighting
        del token_weighting

        # Get corpus TPI as the mean
        tpi = tpi_array.mean()

        return tpi, tpi_array, ICs_dict, IC_multiplier_array

    def get_TPS(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
            weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH, term_alterning=TPS_TERM_ALTERNING,
            similarity_model_name:str=TPS_SIMILARITY_MODEL_NAME, use_chunking:bool=TPS_USE_CHUNKING,
            ICs_dict:Optional[Dict[str,np.ndarray]]=None) -> Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
        """
        Text Preserved Similarity (TPS) measures the percentage of information content (IC) still present in the masked documents,
        weighted by the similarity of replacement terms.
        This metric is used to assess utility preservation for replacement-based masking (i.e., text sanitization).
        It employs `ICTokenWeighting` for measuring IC and a specified similarity model for replacement similarity.
        This metric was proposed in [Truthful Text Sanitization Guided by Inference Attacks](https://arxiv.org/abs/2412.12928) (Pilán et al., arXiv 2024)

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            weighting_model_name (Optional[str]): Name of the model for `ICTokenWeighting`. If None, uniform weighting is used.
            weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`.
            term_alterning: Parameter for term alternation in IC calculation.
                It can be an integer (e.g., N = 6) or the string "sentence." 
                When using an integer N, one of the N terms will be masked each round.
                A larger N value implies a more accurate IC estimation (up to a certain point), but slower computation because more rounds are required.
                If "sentence" is used, the text will be split into sentences, and one of the sentence terms will be masked at each round.
                This approach is significantly slower but may provide the most accurate IC estimation.
            similarity_model_name (str): The name of the embedding model to use for calculating text similarity.
            use_chunking (bool): Whether to use chunking for term span extraction. It is recommended for a more precise IC calculation.
            ICs_dict (Optional[Dict[str,np.ndarray]]): Precomputed IC values for documents. 
            Used in `evaluate` to avoid recomputing, for each anonymization, the original document's ICs (which are always identical).

        Returns:
            Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
                - float: The average TPS for the corpus.
                - np.ndarray: An array of TPS values for each document.
                - Dict[str,np.ndarray]: A dictionary containing precomputed ICs (used for caching).
                - np.ndarray: An array of similarities for replacements.
        """
        
        # Initialize outputs
        tps_array = np.empty(len(masked_docs))
        if ICs_dict is None:
            ICs_dict = {}
        similarity_array = []

        # Define token weighting
        if weighting_model_name is None:
            token_weighting = UniformTokenWeighting()
        
        else:
            token_weighting = ICTokenWeighting(model_name=weighting_model_name, device=DEVICE,
                                               max_segment_length=weighting_max_segment_length)
        
        # Load embedding model and function for similarity
        embedding_func, embedding_model = self._get_embedding_func(similarity_model_name)
        
        # Process each masked document
        for idx, masked_doc in enumerate(masked_docs):
            doc = self.documents[masked_doc.doc_id]

            # Get text spans
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)

            # Get IC for all spans
            if masked_doc.doc_id in ICs_dict:
                spans_IC = ICs_dict[masked_doc.doc_id] # Use precomputed ICs
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
        
        # Dispose token weighting
        del token_weighting

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
    
    def _get_spans_ICs(self, spans: List[Tuple[int,int]], doc:Document, token_weighting: TokenWeighting,
                        context_span:Optional[Tuple[int,int]]=None) -> np.ndarray:
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
    
    def _get_embedding_func(self, sim_model_name:str) -> Tuple:
        embedding_model = None

        if sim_model_name is None: # Default spaCy model
            embedding_func = lambda x: np.array([self.spacy_nlp(text).vector for text in x])
        else:   # Sentence Transformer
            embedding_model = SentenceTransformer(sim_model_name, trust_remote_code=True)
            embedding_func = lambda x: embedding_model.encode(x, show_progress_bar=False)
        
        return embedding_func, embedding_model
    
    def _get_replacements_info(self, masked_doc:MaskedDocument, doc:Document,
                               spans:List[Tuple[int, int]]) -> Tuple[List[str], List[str], List[List[int]]]:
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
                k_multiplier:int=NMI_K_MULTIPLIER, embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=MASKING_MARKS,
                n_clusterings:int=NMI_N_CLUSTERINGS,
                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[Dict[str,float], List[List[np.ndarray]], np.ndarray, int]:
        """
        Computes the Normalized Mutual Information (NMI) between the original corpus and anonymized corpora in document clustering.
        This metric is used to measure empirical utility preservation in a generic downstream task.        
        NMI measures how well the K-means++ clusters formed by the anonymized texts align with the clusters formed by the original texts.
        Clustering is repeated multiple times for minimizing the impact of randomness.
        This metric was proposed in [Truthful Text Sanitization Guided by Inference Attacks](https://arxiv.org/abs/2412.12928) (Pilán et al., arXiv 2024)
        For this particular implementation, clustering is carried out with multiple Ks,
        choosing as a result the one which provided the best silouhette score in original texts clustering.

        Args:
            anonymizations (Dict[str, List[MaskedDocument]]): A dictionary where keys are anonymization names and values are lists of masked documents.
            min_k (int): The minimum number of clusters `k` to consider.
            max_k (int): The maximum number of clusters `k` to consider.
            k_multiplier (int): The multiplier to increase `k` for each iteration.
                Iterations start with from `min_k` and ending when `max_k` is surpassed.
            embedding_model_name (str): The name of the embedding model to use for text vectorial representation.
            remove_mask_marks (bool): Whether to remove mask marks (e.g., "SENSITIVE" or "PERSON") from the text before embedding.
            mask_marks (List[str]): A list of mask marks to remove if `remove_mask_marks` is True (by default `MASKING_MARKS`).
            n_clusterings (int): The number of clusterings to perform for each `k`.
            n_tries_per_clustering (int): The number of tries for each clustering. 
            Total number of clusterings per `k` will be `n_clusterings`*`n_tries_per_clustering`

        Returns:
            Tuple[Dict[str,float], np.ndarray, np.ndarray, int]:
                - Dict[str,float]: A dictionary containing the NMI scores for each anonymization.
                - List[List[np.ndarray]]: A list of lists of clustering labels. 
                    For each of the `n_clusterings` for the best `k`, for each of the anonymizations.
                - np.ndarray: An array of silhouette scores for each evaluated `k`.
                - int: The best `k` value chosen based on silhouette score.
        """
        
        # Create the corpora
        orig_corpora = self._get_anonymization_corpora(anonymizations, include_original_text=True)
        nmi_corpora = [[doc_dict[ORIGINAL_TEXT_KEY] for doc_dict in orig_corpora.values()]] # Prepend original texts (ground truth)
        nmi_corpora += [[doc_dict[anon_name] for doc_dict in orig_corpora.values()] for anon_name in anonymizations.keys()]

        # Get the embeddings
        corpora_embeddings = self._get_corpora_embeddings(nmi_corpora, embedding_model_name,
                                                   remove_mask_marks=remove_mask_marks, mask_marks=mask_marks)
        
        # Clustering results based on the maximum silhouette
        values, all_corpora_labels, true_silhouettes, best_k = self._silhouette_based_NMI(corpora_embeddings, min_k=min_k, max_k=max_k, k_multiplier=k_multiplier,
                                                                      n_clusterings=n_clusterings, n_tries_per_clustering=n_tries_per_clustering)
        
        # Prepare results
        values = values[1:] # Remove result for the first corpus (ground truth defined by the original texts)
        results = {anon_name:value for anon_name, value in zip(anonymizations.keys(), values)}
        
        return results, all_corpora_labels, true_silhouettes, best_k

    def _get_corpora_embeddings(self, corpora:List[List[str]], embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                                 remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=MASKING_MARKS,
                                 device:str=DEVICE) -> List[np.ndarray]:
        corpora_embeddings = []

        # Load model
        model = SentenceTransformer(embedding_model_name, device=device)
        model.eval()
        
        # Collect embeddings
        mask_marks_re_pattern = "|".join([m.upper() for m in mask_marks])
        for corpus in tqdm(corpora, desc="Computing embeddings"):
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

    def _silhouette_based_NMI(self, corpora_embeddings:List[np.ndarray], min_k:int=NMI_MIN_K, max_k:int=NMI_MAX_K,
                k_multiplier:int=NMI_K_MULTIPLIER, n_clusterings:int=NMI_N_CLUSTERINGS, 
                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, List[List[np.ndarray]], np.ndarray, int]:
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

        logging.info(f"Clustering results for k={best_k} were selected because they correspond to the maximum silhouette ({max_silhouette:.3f})")
        values, all_corpora_labels, true_silhouettes = outputs_by_k[best_k]

        return values, all_corpora_labels, true_silhouettes, best_k

    def _get_corpora_multiclustering(self, corpora_embeddings:List[np.ndarray], k:int, n_clusterings:int=NMI_N_CLUSTERINGS,
                                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING
                                ) -> Tuple[np.ndarray, List[List[np.ndarray]], np.ndarray]:
        results = np.empty((n_clusterings, len(corpora_embeddings)))
        all_corpora_labels = []
        true_silhouettes = np.empty(n_clusterings)
        for clustering_idx in tqdm(range(n_clusterings), desc=f"Clustering k={k}"):
            true_labels, corpora_labels, true_silhouettes[clustering_idx] = self._get_corpora_clustering(corpora_embeddings, k,
                                                                                                        tries_per_clustering=n_tries_per_clustering)
            results[clustering_idx, :] = self._compare_clusterings(true_labels, corpora_labels)
            all_corpora_labels.append(corpora_labels)

        # Average for the n_clusterings
        results = results.mean(axis=0)

        return results, all_corpora_labels, true_silhouettes

    def _get_corpora_clustering(self, corpora_embeddings:List[np.ndarray], k:int,
                                 tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, List[np.ndarray], float]:
        corpora_labels = []

        # First corpus corresponds to the ground truth
        true_labels = self._get_corpus_clustering(corpora_embeddings[0], k, tries=tries_per_clustering)
        true_silhouette = silhouette_score(corpora_embeddings[0], true_labels, metric="cosine")

        # Clusterize for each corpus
        for corpus_embeddings in corpora_embeddings: # Repeating for the first one (ground truth) allows to check consistency
            labels = self._get_corpus_clustering(corpus_embeddings, k, tries=tries_per_clustering)            
            corpora_labels.append(labels)

        return true_labels, corpora_labels, true_silhouette

    def _get_corpus_clustering(self, corpus_embeddings, k:int, tries:int=NMI_N_TRIES_PER_CLUSTERING) -> np.ndarray:
        kmeanspp = KMeans(n_clusters=k, init="k-means++", n_init=tries)
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


    #endregion


    #region Auxiliar
    
    def _get_anonymization_corpora(self, anonymizations:Dict[str, List[MaskedDocument]],
                                   include_original_text:bool=False) -> Dict[str, Dict[str,str]]:
        corpora = {}
        
        # Transform list of masked docs into dictionaries for faster processing
        anon_dicts = {}
        for anon_name, masked_docs in anonymizations.items():
            anon_dicts[anon_name] = {masked_doc.doc_id:masked_doc for masked_doc in masked_docs}

        # Create a dictionary per document
        for doc_id, doc in self.documents.items():
            doc_dict = {DOC_ID_KEY:doc_id}
            if include_original_text:
                doc_dict[ORIGINAL_TEXT_KEY] = doc.text
            for anon_name, masked_docs_dict in anon_dicts.items():
                masked_doc = masked_docs_dict[doc_id]
                doc_dict[anon_name] = masked_doc.get_masked_text(doc.text)
            corpora[doc_id] = doc_dict

        return corpora

    #endregion

#endregion
