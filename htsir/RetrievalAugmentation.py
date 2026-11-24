import logging
import os
import pickle
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel, GPT3TurboQAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Map supported tree builders to their respective config classes
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,
        qa_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model="SBert",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="SBert",
        # Reranker arguments
        use_reranker=True,
        reranker_type="local",
        reranker_model_name="BAAI/bge-reranker-large",
        reranker_top_k=4,
    ):
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]

        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
            )
        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT3TurboQAModel()
        self.tree_builder_type = tree_builder_type
        self.use_reranker = use_reranker
        self.reranker_type = reranker_type
        self.reranker_model_name = reranker_model_name
        self.reranker_top_k = reranker_top_k

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}

            {tree_retriever_config}

            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
            Use Reranker: {use_reranker} ({reranker_type}, top_k={reranker_top_k})
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
            use_reranker=self.use_reranker,
            reranker_type=self.reranker_type,
            reranker_top_k=self.reranker_top_k,
        )
        return config_summary


class Reranker:
    """Reranks candidate documents using the Cohere Rerank API."""

    def __init__(self, model_name="rerank-english-v3.0", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")

    def rerank_documents(self, query, docs, top_n=4):
        import cohere
        from cohere.errors import TooManyRequestsError

        if not self.api_key:
            raise ValueError(
                "Cohere API key not found. Set the COHERE_API_KEY environment variable or pass api_key."
            )

        retries = 10
        co = cohere.Client(self.api_key)
        for attempt in range(retries):
            try:
                results = co.rerank(
                    model=self.model_name, query=query, documents=docs, top_n=top_n
                )
                reranker_list = [docs[result.index] for result in results.results]
                return "\n".join(reranker_list)
            except TooManyRequestsError:
                if attempt < retries - 1:
                    time.sleep(60)
                else:
                    raise


class RerankerLocal:
    """Reranks candidate documents locally using a HuggingFace cross-encoder (e.g., BGE-Reranker)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def rerank_documents(
        self, question: str, selected_nodes_text: list, topk: int = 4
    ) -> list:
        if not selected_nodes_text:
            return []

        pairs = [[question, doc] for doc in selected_nodes_text]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu()

        scored_docs = list(zip(scores.tolist(), selected_nodes_text))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:topk]]


class RetrievalAugmentation:
    """
    HTSIR Retrieval Augmentation pipeline combining hierarchical tree construction,
    collapsed tree retrieval, optional reranking, and question answering.
    """

    def __init__(self, config=None, tree=None):
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model
        self.use_reranker = config.use_reranker
        self.reranker_type = config.reranker_type
        self.reranker_model_name = config.reranker_model_name
        self.reranker_top_k = config.reranker_top_k
        self._reranker = None

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def _get_reranker(self):
        if self._reranker is None:
            if self.reranker_type == "local":
                self._reranker = RerankerLocal(model_name=self.reranker_model_name)
            elif self.reranker_type == "cohere":
                self._reranker = Reranker(model_name=self.reranker_model_name)
            else:
                raise ValueError(
                    f"Unsupported reranker_type '{self.reranker_type}'. Expected 'local' or 'cohere'."
                )
        return self._reranker

    def add_documents(self, docs, max_tokens: int = 100, chunk_group_size: int = 2):
        """
        Adds a document to the HTSIR tree and initializes the TreeRetriever.

        Args:
            docs (str | dict): Plain text string for unstructured documents, or a
                structured dict (e.g., QASPER paper entry with 'title', 'abstract',
                'full_text') for hierarchical documents.
            max_tokens (int): Leaf chunk length l. Defaults to 100.
            chunk_group_size (int): Number of chunks r grouped for intermediate summaries.
                Defaults to 2.
        """
        if isinstance(docs, dict):
            self.tree = self.tree_builder.build_from_text_section(
                value=docs,
                max_tokens=max_tokens,
                chunk_group_size=chunk_group_size,
            )
        elif isinstance(docs, str):
            self.tree = self.tree_builder.new_build_from_text(
                text=docs, chunknum=chunk_group_size
            )
        else:
            raise ValueError("docs must be a str or a structured document dict.")

        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """
        Retrieves relevant context from the HTSIR tree for a given question.
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
        reranker_top_k: int = None,
    ):
        """
        Retrieves relevant nodes, optionally reranks them, and generates an answer.
        """
        context, selected_nodes_text = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, False
        )

        if self.use_reranker and selected_nodes_text:
            reranker = self._get_reranker()
            k = reranker_top_k if reranker_top_k is not None else self.reranker_top_k
            if self.reranker_type == "local":
                reranked_docs = reranker.rerank_documents(
                    question, selected_nodes_text, topk=k
                )
                qa_context = "\n\n".join(reranked_docs)
            else:
                qa_context = reranker.rerank_documents(
                    question, selected_nodes_text, top_n=k
                )
        else:
            qa_context = context

        answer = self.qa_model.answer_question(qa_context, question)

        if return_layer_information:
            return answer

        return qa_context, answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")



