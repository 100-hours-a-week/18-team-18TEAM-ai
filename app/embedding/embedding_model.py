"""BGE-m3-ko 임베딩 모델 싱글톤"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """dragonkue/BGE-m3-ko 모델을 싱글톤으로 로드하고 텍스트를 1024차원 벡터로 변환한다."""

    _instance: EmbeddingModel | None = None
    _model = None
    DIMENSION = 1024
    MODEL_NAME = "dragonkue/BGE-m3-ko"

    @classmethod
    def get_instance(cls) -> EmbeddingModel:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if EmbeddingModel._model is None:
            logger.info("임베딩 모델 로딩 중: %s", self.MODEL_NAME)
            try:
                from sentence_transformers import SentenceTransformer

                EmbeddingModel._model = SentenceTransformer(self.MODEL_NAME)
                logger.info("임베딩 모델 로딩 완료 (dim=%d)", self.DIMENSION)
            except Exception as e:
                logger.error("임베딩 모델 로딩 실패: %s", e)
                raise

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(self, text: str) -> List[float]:
        """단일 텍스트를 1024차원 벡터로 인코딩한다."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 인코딩한다."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
