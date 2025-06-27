#region Imports

import json, re, abc, argparse, math, ntpath, os, csv
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from functools import partial

from tqdm.autonotebook import tqdm
import numpy as np

import spacy
import intervaltree
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# TODO: Manage imports from document clustering
from transformers import AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# TODO: Apply logging instead of prints
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

#endregion

#region Constants

# Configuration dictionary keys
CORPUS_CONFIG_KEY = "corpus_filepath"
ANONYMIZATIONS_CONFIG_KEY = "anonymizations"
RESULTS_CONFIG_KEY = "results_filepath"
METRICS_CONFIG_KEY = "metrics"
MANDATORY_CONFIG_KEYS = [CORPUS_CONFIG_KEY, ANONYMIZATIONS_CONFIG_KEY, RESULTS_CONFIG_KEY, METRICS_CONFIG_KEY]

# Corpus dictionary keys
DOC_ID_KEY = "doc_id"

# Metric names
PRECISION_METRIC_NAME = "Precision"
RECALL_METRIC_NAME = "Recall"
TPI_METRIC_NAME = "TPI"
TPS_METRIC_NAME = "TPS"
NMI_METRIC_NAME = "NMI"
TRIR_METRIC_NAME = "TRIR"
METRIC_NAMES = [PRECISION_METRIC_NAME, RECALL_METRIC_NAME, TPI_METRIC_NAME, TPS_METRIC_NAME, NMI_METRIC_NAME, TRIR_METRIC_NAME]


# POS tags, tokens or characters that can be ignored from the recall scores 
# (because they do not carry much semantic content, and there are discrepancies
# on whether to include them in the annotated spans or not)
POS_TO_IGNORE = {"ADP", "PART", "CCONJ", "DET"} 
TOKENS_TO_IGNORE = {"mr", "mrs", "ms", "no", "nr", "about"}
CHARACTERS_TO_IGNORE = " ,.-;:/&()[]–'\" ’“”"


#TODO: Manage constants from document clustering
# To avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3
os.environ["OMP_NUM_THREADS"] = "3"

MODEL_NAME = "bert-base-cased" # Other options: "distilbert-base-uncased", "distilbert-base-cased", "bert-base-uncased", "roberta-base"

MASK_MARKS = ["sensitive", "person", "dem", "loc",
                        "org", "datetime", "quantity", "misc",
                        "norp", "fac", "gpe", "product", "event",
                        "work_of_art", "law", "language", "date",
                        "time", "ordinal", "cardinal", "date_time", "datetime",
                        "nrp", "location", "organization", "\*\*\*"]

# Check for GPU with CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    DEVICE = torch.device("cpu")

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

    def get_masked_offsets(self):
        """Returns the character offsets that are masked"""
        if not hasattr(self, "masked_offsets"):
            self.masked_offsets = {i for start, end in self.masked_spans
                                   for i in range(start, end)}
        return self.masked_offsets

@dataclass
class MaskedDocumentList(List[MaskedDocument]):
    def __init__(self, masked_docs_filepath:str):
        """Given a file path for a JSON file containing the spans to be masked for
        each document, builds a list of MaskedDocument objects"""
        masked_docs_list = []
        
        with open(masked_docs_filepath, "r", encoding="utf-8") as fd:
            masked_docs_dict = json.load(fd)
        
        if type(masked_docs_dict)!= dict:
            raise RuntimeError(f"{masked_docs_filepath} must contain a mapping between document identifiers"
                                + " and lists of masked spans in this document")
        
        for doc_id, masked_spans in masked_docs_dict.items():
            doc = MaskedDocument(doc_id, [], [])
            if type(masked_spans)!=list:
                raise RuntimeError("Masked spans for the document must be a list of [start, end, replacement] tuples (replacement is optional)")
            
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
            raise RuntimeError("Direct identifiers must always be masked")

    @property
    def mentions_to_mask(self):
        return [mention for i, mention in enumerate(self.mentions)
                if self.mention_level_masking[i]]

class GoldDocument:
    """Representation of a gold standard annotated document"""

    doc_id:str
    text:str
    spacy_docs:spacy.tokens.Doc
    entities:dict

    #region Initialization
    
    def __init__(self, doc_id:str, text:str, annotations:Dict[str,List],
                 spacy_doc:spacy.tokens.Doc):
        """Creates a new annotated document with an identifier, a text content, and 
        a set of annotations (see guidelines)"""
        
        # The (unique) document identifier, its text and the spacy document
        self.doc_id = doc_id
        self.text = text
        self.spacy_doc = spacy_doc
        
        # Annotated entities (indexed by id)
        self.entities = {}        
        for annotator, ann_by_person in annotations.items():
            
            if "entity_mentions" not in ann_by_person:
                raise RuntimeError("Annotations must include entity_mentions")
            
            for entity in self._get_entities_from_mentions(ann_by_person["entity_mentions"]):
                
                # We require each entity_id to be specific for each annotator
                if entity.entity_id in self.entities:
                    raise RuntimeError(f"Entity ID {entity.entity_id} already used by another annotator")
                    
                entity.annotator = annotator
                entity.doc_id = doc_id
                self.entities[entity.entity_id] = entity
                    
    def _get_entities_from_mentions(self, entity_mentions):
        """Returns a set of entities based on the annotated mentions"""
        
        entities = {}
        
        for mention in entity_mentions:
                
            for key in ["entity_id", "identifier_type", "start_offset", "end_offset"]:
                if key not in mention:
                    raise RuntimeError("Unspecified key in entity mention: " + key)
                                   
            entity_id = mention["entity_id"]
            start = mention["start_offset"]
            end = mention["end_offset"]
                
            if start < 0 or end > len(self.text) or start >= end:
                raise RuntimeError(f"Invalid character offsets: [{start}-{end}]")
                
            if mention["identifier_type"] not in ["DIRECT", "QUASI", "NO_MASK"]:
                raise RuntimeError(f"Unspecified or invalid identifier type: {mention["identifier_type"]}")

            need_masking = mention["identifier_type"] in ["DIRECT", "QUASI"]
            is_direct = mention["identifier_type"]=="DIRECT"
            
                
            # We check whether the entity is already defined
            if entity_id in entities:
                    
                # If yes, we simply add a new mention
                current_entity = entities[entity_id]
                current_entity.mentions.append((start, end))
                current_entity.mention_level_masking.append(need_masking)
                    
            # Otherwise, we create a new entity with one single mention
            else:
                new_entity = AnnotatedEntity(entity_id, [(start, end)], need_masking, is_direct, 
                                             mention["entity_type"], [need_masking])
                entities[entity_id] = new_entity
                
        for entity in entities.values():
            if set(entity.mention_level_masking) != {entity.need_masking}:
                entity.need_masking = True
                #print(f"Warning: inconsistent masking of entity {entity.entity_id}: {entity.mention_level_masking}") # TODO: Check this
                
        return list(entities.values())
    
    #endregion

    #region Functions

    def is_masked(self, masked_doc:MaskedDocument, entity: AnnotatedEntity):
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
    
    def is_mention_masked(self, masked_doc:MaskedDocument, mention_start:int, mention_end:int):
        """Given a document with a set of masked text spans and a particular mention span,
        determine whether the mention is fully masked (taking into account special
        characters or tokens to skip)"""
        
        mention_to_mask = self.text[mention_start:mention_end].lower()
                
        # Computes the character offsets that must be masked
        offsets_to_mask = set(range(mention_start, mention_end))

        # We build the set of character offsets that are not covered
        non_covered_offsets = offsets_to_mask - masked_doc.get_masked_offsets()
            
        # If we have not covered everything, we also make sure punctuations
        # spaces, titles, etc. are ignored
        if len(non_covered_offsets) > 0:
            span = self.spacy_doc.char_span(mention_start, mention_end, alignment_mode = "expand")
            for token in span:
                if token.pos_ in POS_TO_IGNORE or token.lower_ in TOKENS_TO_IGNORE:
                    non_covered_offsets -= set(range(token.idx, token.idx+len(token)))
        for i in list(non_covered_offsets):
            if self.text[i] in set(CHARACTERS_TO_IGNORE):
                non_covered_offsets.remove(i)

        # If that set is empty, we consider the mention as properly masked
        return len(non_covered_offsets) == 0

    def get_entities_to_mask(self,  include_direct=True, include_quasi=True):
        """Return entities that should be masked, and satisfy the constraints 
        specified as arguments"""
        
        to_mask = []
        for entity in self.entities.values():     
             
            # We only consider entities that need masking and are the right type
            if not entity.need_masking:
                continue
            elif entity.is_direct and not include_direct:
                continue
            elif not entity.is_direct and not include_quasi:
                continue  
            to_mask.append(entity)
                
        return to_mask      
        
    def get_annotators_for_span(self, start_token: int, end_token: int):
        """Given a text span (typically for a token), determines which annotators 
        have also decided to mask it. Concretely, the method returns a (possibly
        empty) set of annotators names that have masked that span."""
        
        
        # We compute an interval tree for fast retrieval
        if not hasattr(self, "masked_spans"):
            self.masked_spans = intervaltree.IntervalTree()
            for entity in self.entities.values():
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

    def split_by_tokens(self, start: int, end: int):
        """Generates the (start, end) boundaries of each token included in this span"""
        
        for match in re.finditer(r"\w+", self.text[start:end]):
            start_token = start + match.start(0)
            end_token = start + match.end(0)
            yield start_token, end_token

    #endregion

class TokenWeighting:
    """Abstract class for token weighting schemes"""

    @abc.abstractmethod
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        """Given a text and a list of text spans, returns a list of numeric weights
        (of same length as the list of spans) representing the information content
        conveyed by each span.

        A weight close to 0 represents a span with low information content (i.e. which
        can be easily predicted from the remaining context), while a higher weight 
        represents a high information content (which is difficult to predict from the
        context)"""

        return

class BertTokenWeighting(TokenWeighting):
    """Token weighting based on a BERT language model. The weighting mechanism
    runs the BERT model on a text in which the provided spans are masked. The
    weight of each token is then defined as -log(probability of the actual token value).
    
    In other words, a token that is difficult to predict will have a high
    information content, and therefore a high weight, whereas a token which can
    be predicted from its content will received a low weight. """
    
    def __init__(self, max_segment_size = 100):
        """Initialises the BERT tokenizers and masked language model"""        
        self.tokeniser = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForMaskedLM.from_pretrained('google-bert/bert-base-uncased', trust_remote_code=True)
        self.model = self.model.to(self.device)
        
        self.max_segment_size = max_segment_size
        
        
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        """Returns a list of numeric information content weights, where each value
        corresponds to -log(probability of predicting the value of the text span
        according to the BERT model).
        
        If the span corresponds to several BERT tokens, the probability is the 
        mininum of the probabilities for each token."""
        
        # STEP 1: we tokenise the text
        bert_tokens = self.tokeniser(text, return_offsets_mapping=True)
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
                input_ids[token_idx] = self.tokeniser.mask_token_id
          
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
        
        return weights


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
                print(f"Warning: span ({span_start},{span_end}) without any token [{repr(text[span_start:span_end])}]")
        
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
        if nb_tokens > self.max_segment_size:
            nb_segments = math.ceil(nb_tokens/self.max_segment_size)
            
            # Split the input_ids (and add padding if necessary)
            split_pos = [self.max_segment_size * (i + 1) for i in range(nb_segments - 1)]
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

class UniformTokenWeighting(TokenWeighting):
    """Uniform weighting (all tokens assigned to a weight of 1.0)"""
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        return [1.0] * len(text_spans)


#endregion


#region Evaluator

class Evaluator:
    """Representation of a the corpus for text anonymization, extracted from a JSON file.
    Optionally, it can include gold annotations, standardly made by humans"""

    documents:dict
    nlp=None

    #region Initialization
    
    def __init__(self, corpus:list, spacy_model_name:str="en_core_web_md"):
        # Documents indexed by identifier
        self.documents = {}

        # Loading the spaCy model
        self.nlp = spacy.load(spacy_model_name, disable=["lemmatizer"])        

        for ann_doc in tqdm(corpus):
            for key in [DOC_ID_KEY, "text", "annotations"]: #TODO: Annotations only needed for precision and recall
                if key not in ann_doc:
                    raise RuntimeError(f"Annotated document is not well formed: Missing variable {key}")
            
            # Parsing the document with spaCy
            spacy_doc = self.nlp(ann_doc["text"]) #TODO: Only create this if needed
            
            # Creating the actual document (identifier, text and annotations)          
            new_doc = GoldDocument(ann_doc[DOC_ID_KEY], ann_doc["text"],
                                   ann_doc["annotations"], spacy_doc)
            self.documents[ann_doc[DOC_ID_KEY]] = new_doc

    #endregion


    #region Evaluation

    def evaluate(self, anonymizations:Dict[List[MaskedDocument]], metrics:dict, results_filepath:Optional[str]=None, verbose:bool=False):
        results = {}

        #TODO: Check each anonymization has a masked version of all the documents in the corpus

        if results_filepath:
            self._write_into_results(results_filepath, ["Metric/Anonymization"]+list(anonymizations.keys())) # First row for anonymizations names

        # For each metric
        for metric_name, parameters in metrics.items():
            partial_eval_func = self._get_partial_eval_func(metric_name, parameters)
            
            # For each anonymization
            metric_results = {}
            for anon_name, masked_docs in anonymizations.items(): # TODO: This approach is not valid for NMI
                # If is a valid metric, compute it
                if not partial_eval_func is None:
                    output = partial_eval_func(masked_docs)
                    value = output[0] # First of the results is the actual value of the metric
                    metric_results[anon_name] = value
                # Otherwise, there are no results for this metric
                else:
                    metric_results[anon_name] = None

                if verbose:
                    pass #TODO: Print results in a fancy way            

            # Save results
            results[metric_name] = metric_results
            if results_filepath:
                self._write_into_results([metric_name]+list(metric_results.values()))
        
        return results

    def _get_partial_eval_func(self, metric_name:str, parameters:dict) -> partial:
        partial_func = None

        if not metric_name in METRIC_NAMES:
            logging.warning(f"Unknown metric name {metric_name} | Available metrics: {METRIC_NAMES}")
        
        if metric_name == PRECISION_METRIC_NAME:
            partial_func = partial(self.get_precision, **parameters)
        elif metric_name == RECALL_METRIC_NAME:
            partial_func = partial(self.get_recall, **parameters)
        elif metric_name == TPI_METRIC_NAME:
            partial_func = partial(self.get_TPI, **parameters)
        elif metric_name == TPS_METRIC_NAME:
            partial_func = partial(self.get_TPS, **parameters)
        elif metric_name == NMI_METRIC_NAME:
            partial_func = partial(self.get_NMI, **parameters)
        elif metric_name == TRIR_METRIC_NAME:
            partial_func = partial(self.get_TRIR, **parameters)

        #TODO: Check if required data (e.g., manual annotations) is available

        return partial_func

    def _write_into_results(self, results_filepath:str, row:list):
        with open(results_filepath, 'a+') as csvfile: #TODO: Create path to results if it does not exist
            writer = csv.writer(csvfile)
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")        
            writer.writerow([datetime_str]+row)
    
    #endregion


    #region Precision
            
    def get_precision(self, masked_docs:List[MaskedDocument], token_weighting:TokenWeighting, 
                      token_level:bool=True):
        """Returns the weighted, token-level precision of the masked spans when compared 
        to the gold standard annotations. Arguments:
        - masked_docs: documents together with spans masked by the system
        - token_weighting: mechanism for weighting the information content of each token
        
        If token_level is set to true, the precision is computed at the level of tokens, 
        otherwise the precision is at the mention-level. The masked spans/tokens are weighted 
        by their information content, given the provided weighting scheme. If annotations from 
        several annotators are available for a given document, the precision corresponds to a 
        micro-average over the annotators."""      
        
        weighted_true_positives = 0.0
        weighted_system_masks = 0.0
                
        for doc in tqdm(masked_docs):
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
            nb_annotators = len(set(entity.annotator for entity in gold_doc.entities.values()))
            
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

    def get_recall(self, masked_docs:List[MaskedDocument], include_direct=True, 
                    include_quasi=True, token_level:bool=True):
        """Returns the mention or token-level recall of the masked spans when compared 
        to the gold standard annotations. 
        
        Arguments:
        - masked_docs: documents together with spans masked by the system
        - include_direct: whether to include direct identifiers in the metric
        - include_quasi: whether to include quasi identifiers in the metric
        - token_level: whether to compute the recall at the level of tokens or mentions
                
        If annotations from several annotators are available for a given document, the recall 
        corresponds to a micro-average over the annotators. """

        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        nb_masked_elements = sum(nb_masked_by_type.values())
        nb_elements = sum(nb_by_type.values())
                
        try:
            return nb_masked_elements / nb_elements
        except ZeroDivisionError:
            return 0

    def get_recall_per_entity_type(self, masked_docs:List[MaskedDocument], include_direct=True, 
                                   include_quasi=True, token_level:bool=True):
        """Returns the mention or token-level recall of the masked spans when compared 
        to the gold standard annotations, and factored by entity type. 
        
        Arguments:
        - masked_docs: documents together with spans masked by the system
        - include_direct: whether to include direct identifiers in the metric
        - include_quasi: whether to include quasi identifiers in the metric
        - token_level: whether to compute the recall at the level of tokens or mentions
                
        If annotations from several annotators are available for a given document, the recall 
        corresponds to a micro-average over the annotators. """
        
        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        return {ent_type:nb_masked_by_type[ent_type]/nb_by_type[ent_type]
                for ent_type in nb_by_type}
                
    def _get_mask_counts(self, masked_docs:List[MaskedDocument], include_direct=True, 
                                   include_quasi=True, token_level:bool=True):
        
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

    def get_TPS(self, masked_docs:List[MaskedDocument], token_weighting: TokenWeighting,
                word_alterning=6, sim_model_name="paraphrase-albert-base-v2", use_chunking=True) -> Tuple[float, np.ndarray, np.ndarray]:
        tps_array = np.empty(len(masked_docs))
        similarity_array = []
        
        # Load embedding model and function for similarity
        embedding_func = self._get_embedding_func(sim_model_name)
        
        # Process each masked document
        for i, masked_doc in enumerate(tqdm(masked_docs)):
            gold_doc = self.documents[masked_doc.doc_id]

            # Get text spans
            spans = self._get_terms_spans(gold_doc.spacy_doc, use_chunking=use_chunking)

            # Get IC for all spans
            spans_IC = self._get_ICs(spans, gold_doc, token_weighting, word_alterning)

            # Get replacements, corresponding masked texts and corresponding spans indexes
            repl_out = self._get_replacements_info(masked_doc, gold_doc, spans)
            (replacements, masked_texts, spans_idxs_per_replacement) = repl_out

            # Measure similarities of replacements
            masked_spans = self._filter_masked_spans(gold_doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, gold_doc, masked_spans) # Non-masked=True(1), Masked=False(0)
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

        # Get mean TPS
        tps = tps_array.mean()

        # All similarities to NumPy array
        similarity_array = np.array(similarity_array)

        return tps, tps_array, similarity_array 

    def get_TPI(self, masked_docs: List[MaskedDocument], token_weighting: TokenWeighting,
            word_alterning=6, use_chunking: bool=True) -> Tuple[float, np.ndarray, np.ndarray]:
        tpi_array = np.empty(len(masked_docs))
        IC_multiplier_array = np.empty(len(masked_docs))

        for i, masked_doc in enumerate(tqdm(masked_docs)):
            gold_doc = self.documents[masked_doc.doc_id]

            # Get terms spans and mask
            spans = self._get_terms_spans(gold_doc.spacy_doc, use_chunking=use_chunking)
            masked_spans = self._filter_masked_spans(gold_doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, gold_doc, masked_spans) # Non-masked=True(1), Masked=False(0)

            # Get IC for all spans
            spans_IC = self._get_ICs(spans, gold_doc, token_weighting, word_alterning)
            
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

    def _get_terms_spans(self, spacy_doc: spacy.tokens.Doc, use_chunking: bool=True) -> list:
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

    def _filter_masked_spans(self, gold_doc, masked_doc: MaskedDocument) -> list:
        filtered_masked_spans = []

        masking_array = np.zeros(len(gold_doc.spacy_doc.text), dtype=bool)
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

    def _get_spans_mask(self, spans: List[Tuple[int, int]], gold_doc, masked_spans: list) -> np.array:
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

    def _get_ICs(self, spans: List[Tuple[int, int]], gold_doc, token_weighting: TokenWeighting, word_alterning) -> np.array:
        spans_IC = np.empty(len(spans))
        if isinstance(word_alterning, int) and word_alterning > 1: # N Word Alterning
            # Get ICs by masking each N words, with all the document as context
            for i in range(word_alterning):
                spans_for_IC = spans[i::word_alterning]
                spans_IC[i::word_alterning] = self._get_spans_ICs(spans_for_IC, gold_doc, token_weighting)
        
        elif isinstance(word_alterning, str) and word_alterning == "sentence": # Sentence Word Alterning
            # Get masks by masking 1 word of each sentence, with the sentence as context
            # Get sentences spans
            sentences_spans = [[sent.start_char, sent.end_char] for sent in gold_doc.spacy_doc.sents]
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
                    original_info, masked_info, n_masked_terms = self._get_spans_ICs(gold_doc, [span], gold_doc,
                                                                                     token_weighting, context_span=sentence_span)
                    original_doc_info += original_info
                    masked_doc_info += masked_info
                    total_n_masked_terms += n_masked_terms
        else:
            raise Exception(f"Word alterning setting [{word_alterning}] is invalid")

        return spans_IC
    
    def _get_spans_ICs(self, spans: List[Tuple[int, int]], gold_doc, token_weighting: TokenWeighting, context_span=None) -> np.array:
        # By default, context span is all the document
        if context_span is None:
            context_span = (0, len(gold_doc.text))

        # Get context
        context_start, context_end = context_span
        context = gold_doc.text[context_start:context_end]

        # Adjust spans to the context
        in_context_spans = []
        for (start, end) in spans:
            in_context_spans.append((start - context_start, end - context_start))

        # Obtain the weights (Information Content) of each word
        ICs = token_weighting.get_weights(context, in_context_spans)
        ICs = np.array(ICs) # Transform to numpy

        return ICs
    
    def _get_embedding_func(self, sim_model_name:str):
        if sim_model_name is None: # Default spaCy model
            embedding_func = lambda x: np.array([self.nlp(text).vector for text in x])
        else:   # Sentence Transformer
            sim_model = SentenceTransformer(sim_model_name, trust_remote_code=True)
            embedding_func = lambda x : sim_model.encode(x)
        
        return embedding_func
    
    def _get_replacements_info(self, masked_doc: MaskedDocument, gold_doc, spans: list):
        replacements = []
        masked_texts = []
        spans_idxs_per_replacement = []
        
        for replacement, (masked_span_start, masked_span_end) in zip(masked_doc.replacements, masked_doc.masked_spans):
            if replacement is not None: # If there is a replacement
                replacements.append(replacement)
                masked_texts.append(gold_doc.text[masked_span_start:masked_span_end])
                replacement_spans_idxs = []
                for span_idx, (span_start, span_end) in enumerate(spans):
                    if span_start <= masked_span_start < span_end or span_start < masked_span_end <= span_end:
                        replacement_spans_idxs.append(span_idx)
                    elif span_start > masked_span_end:  # Break if candidate span starts too late
                        break
                spans_idxs_per_replacement.append(replacement_spans_idxs)
        
        return replacements, masked_texts, spans_idxs_per_replacement
    
    def _cos_sim(self, a:np.array, b:np.array) -> float:
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        sim = dot_product / (magnitude_a * magnitude_b)
        if np.isnan(sim):
            sim = 0
        return sim

    #endregion


    #region DocumentClustering

    def get_NMI(self, masked_docs:List[MaskedDocument]):
        # Get the embeddings
        all_embeddings = self._get_all_embeddings(all_documents)

        # Clustering
        k=4
        NMI_metrics, all_labels = self._multi_clustering_eval(all_embeddings, k=k)

        # Print results
        for elem in zip(docs_columns, NMI_metrics):
            print(elem)
        
        return NMI_metrics

    #region Embedding/feature extraction

    def _get_all_embeddings(self, all_documents, remove_mask_marks=False)->list:
        # Create BERT-based model and tokenizer
        model = AutoModel.from_pretrained(self.clustering_embedding_model_name, output_hidden_states=True) # Whether the model returns all hidden-states.
        model.to(DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.clustering_embedding_model_name)

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
                    corpus_embeddings[i] = self._bert_embedding(text, model, tokenizer)
                    pbar.update(1)

            all_embeddings.append(corpus_embeddings)

        return all_embeddings

    def _bert_embedding(self, texts, model, tokenizer, max_pooling=False):
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

    #endregion

    #region Clustering

    def _multi_clustering_eval(self, all_embeddings, k=None, n_clusterings=5, tries_per_clustering=50):
        results = np.empty((n_clusterings, len(all_embeddings)))
        for i in range(n_clusterings):
            true_labels, all_labels = self._get_all_clusterings(all_embeddings, k=k, tries=tries_per_clustering)
            results[i, :] = self._compare_clusterings(true_labels, all_labels, normalized_mutual_info_score)

        # Average per n_clusterings
        results = results.mean(axis=0)

        return results, all_labels

    def _get_all_clusterings(self, all_embeddings, k=None, tries=50):
        all_labels = []

        true_labels, inertia = self._clusterize(all_embeddings[0], k, tries=tries) # First used as groundtruth

        for embeddings in tqdm(all_embeddings):
            labels, inertia = self._clusterize(embeddings, k, tries=tries) # Repeating for the first allows to check the consistency of the groundtruth
            all_labels.append(labels)

        return true_labels, all_labels

    def _clusterize(self, embeddings, k, tries=50):
        inertia = 0
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=tries)
        labels = kmeans.fit_predict(embeddings)  # WikiActors and Wiki553
        inertia = kmeans.inertia_
        # Wiki553 bad manual | labels = DBSCAN(eps=1.1, min_samples=5, algorithm='kd_tree', metric='euclidean').fit(embeddings).labels_
        #labels = DBSCAN(eps=1.5, min_samples=5, algorithm='kd_tree', metric='euclidean').fit(embeddings).labels_ # Wiki533?
        #labels = OPTICS(min_samples=0.1, max_eps=1.25).fit(embeddings).labels_
        return labels, inertia

    def _compare_clusterings(self, true_labels, all_labels, eval_metric):
        metrics = []
        for labels in all_labels:
            metric = eval_metric(labels, true_labels)
            metrics.append(metric)
        return np.array(metrics)

    #endregion

    #endregion


    #region TRIR

    def get_TRIR(self, masked_docs:List[MaskedDocument]):
        return 0 #TODO

    #endregion

#endregion


#endregion


#region Main

if __name__ == "__main__":


    #region Arguments

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

    # Create evaluator from corpus
    corpus_filepath = config[CORPUS_CONFIG_KEY]
    with open(corpus_filepath, encoding="utf-8") as f:
        corpus = json.load(f)
    if type(corpus)!=list:
        raise RuntimeError("Corpus JSON file must be a list of documents")
    logging.info(f"Corpus with {len(corpus)} documents")
    evaluator = Evaluator(corpus)

    # Create masked documents from anonymizations
    anonymizations = {}
    anonymizations_config = config[ANONYMIZATIONS_CONFIG_KEY]
    for anon_name, anon_filepath in anonymizations_config.items():
        masked_docs = MaskedDocumentList(anon_filepath)
        anonymizations[anon_name] = masked_docs
    
    # Get metrics
    metrics = config[METRICS_CONFIG_KEY]

    # Get filepath for results CSV file
    results_filepath = config[RESULTS_CONFIG_KEY]

    #endregion


    #region Evaluation

    evaluator.evaluate(anonymizations, metrics, results_filepath)

    #endregion


#endregion