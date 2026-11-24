import copy
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

from sentence_transformers import SentenceTransformer

from .EmbeddingModels import BaseEmbeddingModel, SBertEmbeddingModel
from .SummarizationModels import (
    BaseSummarizationModel,
    GPT3TurboSummarizationModel,
)
from .tree_structures import Node, Tree
from .utils import (
    distances_from_embeddings,
    get_embeddings,
    get_str,
    indices_of_nearest_neighbors_from_distances,
    split_text,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_models=None,
        cluster_embedding_model=None,
    ):
        if tokenizer is None:
            #tokenizer = tiktoken.get_encoding("cl100k_base")
            tokenizer = SentenceTransformer('all-MiniLM-L6-v2').tokenizer
        self.tokenizer = tokenizer

        if max_tokens is None:
            max_tokens = 150
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        if summarization_model is None:
            summarization_model = GPT3TurboSummarizationModel()
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        self.summarization_model = summarization_model

        if embedding_models is None:
            #embedding_models = {"OpenAI": OpenAIEmbeddingModel()}
            embedding_models = {"SBert": SBertEmbeddingModel()}
        if not isinstance(embedding_models, dict):
            raise ValueError(
                "embedding_models must be a dictionary of model_name: instance pairs"
            )
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError(
                    "All embedding models must be an instance of BaseEmbeddingModel"
                )
        self.embedding_models = embedding_models

        if cluster_embedding_model is None:
            cluster_embedding_model = "SBert"
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )
        self.cluster_embedding_model = cluster_embedding_model

    def log_config(self):
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
        )
        return config_log


class TreeBuilder(ABC):
    """
    The TreeBuilder class is responsible for building a hierarchical text abstraction
    structure, known as a "tree," using summarization models and
    embedding models.
    """

    def __init__(self, config) -> None:
        """Initializes the tokenizer, maximum tokens, number of layers, top-k value, threshold, and selection mode."""

        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model

        logging.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )
        
 


    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        """Creates a new node with the given index, text, and (optionally) children indices.

        Args:
            index (int): The index of the new node.
            text (str): The text associated with the new node.
            children_indices (Optional[Set[int]]): A set of indices representing the children of the new node.
                If not provided, an empty set will be used.

        Returns:
            Tuple[int, Node]: A tuple containing the index and the newly created node.
        """
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))

          

    def create_embedding(self, text) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_models[self.cluster_embedding_model].create_embedding(
            text
        )

    def summarize(self, context, max_tokens=150) -> str:
        """
        Generates a summary of the input context using the specified summarization model.

        Args:
            context (str, optional): The context to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.o

        Returns:
            str: The generated summary.
        """
        return self.summarization_model.summarize(context, max_tokens)

    def get_relevant_nodes(self, current_node, list_nodes) -> List[Node]:
        """
        Retrieves the top-k most relevant nodes to the current node from the list of nodes
        based on cosine distance in the embedding space.

        Args:
            current_node (Node): The current node.
            list_nodes (List[Node]): The list of nodes.

        Returns:
            List[Node]: The top-k most relevant nodes.
        """
        embeddings = get_embeddings(list_nodes, self.cluster_embedding_model)
        distances = distances_from_embeddings(
            current_node.embeddings[self.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]

        elif self.selection_mode == "top_k":
            best_indices = indices[: self.top_k]

        nodes_to_add = [list_nodes[idx] for idx in best_indices]

        return nodes_to_add

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        """Creates leaf nodes using multithreading from the given list of text chunks.

        Args:
            chunks (List[str]): A list of text chunks to be turned into leaf nodes.

        Returns:
            Dict[int, Node]: A dictionary mapping node indices to the corresponding leaf nodes.
        """
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
        """Builds a golden tree from the input text, optionally using multithreading.

        Args:
            text (str): The input text.
            use_multithreading (bool, optional): Whether to use multithreading when creating leaf nodes.
                Default: True.

        Returns:
            Tree: The golden tree structure.
        """
        chunks = split_text(text, self.tokenizer, self.max_tokens)

        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, text in enumerate(chunks):
                __, node = self.create_node(index, text)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logging.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree
    

    
    def build_from_text_section(
        self,
        value: dict,
        max_tokens: Optional[int] = 100,
        chunk_group_size: int = 2,
        use_multithreading: bool = True,
    ) -> Tree:
        """Builds a hierarchical retrieval tree from a structured document (e.g., QASPER).

        Constructs three levels of granularity:
        - Root nodes: section-level summaries (plus title and abstract directly).
        - Intermediate nodes: summaries over groups of `chunk_group_size` consecutive chunks.
        - Leaf nodes: fine-grained text chunks of length `max_tokens`.

        Args:
            value (dict): Structured document containing "title", "abstract", "full_text",
                and optionally "figures_and_tables".
            max_tokens (Optional[int]): Token length for leaf chunks (l). Defaults to 100.
            chunk_group_size (int): Number of consecutive chunks per intermediate summary (r).
                Defaults to 2.
            use_multithreading (bool): Whether to use multithreading when creating nodes.

        Returns:
            Tree: The constructed hierarchical tree structure.
        """
        chunk_len = max_tokens if max_tokens is not None else self.max_tokens
        tree_chunks = []

        # Extract sections and optional figure/table captions
        section_texts = []
        for section in value["full_text"]:
            section_name = section["section_name"]
            full_paragraph = " ".join(
                paragraph for paragraph in section["paragraphs"] if paragraph
            )
            if full_paragraph:
                section_texts.append(f"{section_name}: {full_paragraph}")

        figures_and_tables = value.get("figures_and_tables", [])
        all_captions = " ".join(
            item["caption"] for item in figures_and_tables if "caption" in item
        )
        if all_captions:
            section_texts.append(all_captions)

        # Root-level section summaries
        section_summaries = [
            self.summarize(text, max_tokens=self.summarization_length)
            for text in section_texts
        ]
        tree_chunks.extend(section_summaries)

        # Include title and abstract directly without redundant summarization
        title_abstract = f"{value['title']} {value['abstract']}".strip()
        if title_abstract:
            tree_chunks.append(title_abstract)

        # Leaf-level chunks and intermediate-level grouped summaries
        leaf_chunks = []
        for text in section_texts:
            section_chunks = split_text(text, self.tokenizer, chunk_len)
            leaf_chunks.extend(section_chunks)

            chunk_groups = [
                section_chunks[i : i + chunk_group_size]
                for i in range(0, len(section_chunks), chunk_group_size)
            ]
            chunk_groups_texts = [get_str(group) for group in chunk_groups]
            intermediate_summaries = [
                self.summarize(group_text, max_tokens=self.summarization_length)
                for group_text in chunk_groups_texts
            ]
            tree_chunks.extend(intermediate_summaries)

        tree_chunks.extend(leaf_chunks)

        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(tree_chunks)
        else:
            leaf_nodes = {}
            for index, text in enumerate(tree_chunks):
                __, node = self.create_node(index, text)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")
        logging.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)
        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree

    def new_build_from_text(
        self, text: str, chunknum: int = 2, use_multithreading: bool = True
    ) -> Tree:
        """Builds a hierarchical retrieval tree from unstructured plain text.

        Args:
            text (str): The input text.
            chunknum (int): Number of consecutive chunks grouped for intermediate summaries.
            use_multithreading (bool, optional): Whether to use multithreading when creating leaf nodes.
                Default: True.

        Returns:
            Tree: The constructed tree structure.
        """
        chunks = split_text(text, self.tokenizer, self.max_tokens)
        chunk_groups = [
            chunks[i : i + chunknum] for i in range(0, len(chunks), chunknum)
        ]
        chunk_groups_texts = [get_str(group) for group in chunk_groups]
        summaries = [
            self.summarize(group_text, max_tokens=self.summarization_length)
            for group_text in chunk_groups_texts
        ]
        chunks.extend(summaries)

        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, chunk_text in enumerate(chunks):
                __, node = self.create_node(index, chunk_text)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")
        logging.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)
        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree

    @abstractmethod
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        Constructs the hierarchical tree structure layer by layer by iteratively summarizing groups
        of relevant nodes and updating the current_level_nodes and all_tree_nodes dictionaries at each step.

        Args:
            current_level_nodes (Dict[int, Node]): The current set of nodes.
            all_tree_nodes (Dict[int, Node]): The dictionary of all nodes.
            layer_to_nodes (Dict[int, List[Node]]): Mapping from layer index to nodes.
            use_multithreading (bool): Whether to use multithreading to speed up the process.

        Returns:
            Dict[int, Node]: The final set of root nodes.
        """
        pass
