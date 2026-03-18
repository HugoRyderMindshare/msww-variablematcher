"""
Embedding encoder module.

This module provides the EmbeddingEncoder class for encoding survey variables
and values using the Google GenAI embedding API.
"""

import functools
from typing import Sequence

from google import genai

from .config import GCPConfig
from .models import Variable


class EmbeddingEncoder:
    """
    Encoder for generating embeddings from survey variables and values.

    Uses Google GenAI's text embedding models to generate vector representations
    of variable labels/questions and value labels/statements.

    Parameters
    ----------
    gcp_config : GCPConfig or None, optional
        GCP configuration object. If None, creates a default GCPConfig.
    batch_size : int, default 250
        Maximum number of texts to embed in a single API call.

    Attributes
    ----------
    gcp_config : GCPConfig
        The GCP configuration being used.
    client : genai.Client
        Cached GenAI client (lazy loaded).

    See Also
    --------
    Survey : Data structure for survey data.
    Variable : Data model for survey variables.

    Examples
    --------
    >>> from variablematcher import EmbeddingEncoder, Survey, GCPConfig
    >>> config = GCPConfig(project_id='my-project')
    >>> encoder = EmbeddingEncoder(gcp_config=config)
    >>> import pyreadstat
    >>> df, meta = pyreadstat.read_sav('data.sav')
    >>> survey = Survey.from_sav(df, meta)
    >>> encoder.encode_survey(survey)
    """

    def __init__(
        self,
        gcp_config: GCPConfig | None = None,
        batch_size: int = 250,
    ) -> None:
        """
        Initialise the EmbeddingEncoder.

        Parameters
        ----------
        gcp_config : GCPConfig or None, optional
            GCP configuration object. If None, creates a default GCPConfig.
        batch_size : int, default 250
            Maximum number of texts to embed in a single API call.
        """
        self.gcp_config = gcp_config or GCPConfig()
        self.batch_size = min(batch_size, 250)

    @functools.cached_property
    def client(self) -> genai.Client:
        """
        Get the GenAI client (lazy loaded and cached).

        Returns
        -------
        genai.Client
            Configured GenAI client for Vertex AI.
        """
        return self.gcp_config.get_genai_client()

    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Encode a list of texts into embeddings.

        Parameters
        ----------
        texts : Sequence of str
            Texts to encode.

        Returns
        -------
        list of list of float
            List of embedding vectors, one per input text.

        Examples
        --------
        >>> embeddings = encoder.encode_texts(['What is your age?', 'Gender'])
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        768
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = self.client.models.embed_content(
                model=self.gcp_config.embedding_model,
                contents=batch,
                config={
                    "output_dimensionality": self.gcp_config.embedding_dimensions,
                },
            )
            all_embeddings.extend(
                [e.values for e in response.embeddings]
            )

        return all_embeddings

    def encode_variables(
        self,
        variables: Sequence[Variable],
        include_values: bool = True,
    ) -> list[list[float]]:
        """
        Encode a list of variables into embeddings.

        Creates a text representation of each variable including its label/question
        and optionally its values, then generates embeddings.

        Parameters
        ----------
        variables : Sequence of Variable
            Variables to encode.
        include_values : bool, default True
            Whether to include value labels in the text representation.

        Returns
        -------
        list of list of float
            List of embedding vectors, one per variable.

        Examples
        --------
        >>> embeddings = encoder.encode_variables(survey.variables)
        >>> len(embeddings) == len(survey.variables)
        True
        """
        texts = []
        for var in variables:
            text = self._variable_to_text(var, include_values=include_values)
            texts.append(text)

        return self.encode_texts(texts)

    def encode_survey(
        self,
        survey: object,
        include_values: bool = True,
    ) -> dict[str, list[float]]:
        """
        Encode all variables in a survey and return embeddings.

        Parameters
        ----------
        survey : Survey
            Survey whose variables to encode.
        include_values : bool, default True
            Whether to include value labels in the text representation.

        Returns
        -------
        dict of str to list of float
            Mapping of variable name to embedding vector.
        """
        variables = list(survey.variables)
        embeddings = self.encode_variables(
            variables, include_values=include_values
        )

        return {
            var.name: emb
            for var, emb in zip(
                variables, embeddings, strict=True
            )
        }

    def _variable_to_text(
        self,
        variable: Variable,
        include_values: bool = True,
    ) -> str:
        """
        Convert a variable to a text representation for embedding.

        Parameters
        ----------
        variable : Variable
            Variable to convert.
        include_values : bool, default True
            Whether to include value labels.

        Returns
        -------
        str
            Text representation of the variable.
        """
        # Start with the best available text for the variable
        text = variable.question or variable.label or variable.name

        # Add value context if available and requested
        if include_values and variable.values:
            value_texts = [
                v.statement or v.label for v in variable.values
            ]
            # Limit to first 10 values to avoid overly long texts
            if len(value_texts) > 10:
                value_texts = value_texts[:10] + ["..."]
            values_str = ", ".join(value_texts)
            text = f"{text} [Values: {values_str}]"

        return text

    def encode_value_pairs(
        self,
        source_values: Sequence[str],
        target_values: Sequence[str],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Encode value labels from two surveys for comparison.

        Parameters
        ----------
        source_values : Sequence of str
            Value labels from the source survey.
        target_values : Sequence of str
            Value labels from the target survey.

        Returns
        -------
        tuple of (list of list of float, list of list of float)
            Tuple of (source_embeddings, target_embeddings).

        Examples
        --------
        >>> s_emb, t_emb = encoder.encode_value_pairs(
        ...     ['Yes', 'No'],
        ...     ['Yes', 'No', 'Maybe']
        ... )
        """
        all_values = list(source_values) + list(target_values)
        all_embeddings = self.encode_texts(all_values)

        n_source = len(source_values)
        source_embeddings = all_embeddings[:n_source]
        target_embeddings = all_embeddings[n_source:]

        return source_embeddings, target_embeddings
