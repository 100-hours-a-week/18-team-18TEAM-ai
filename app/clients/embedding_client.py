"""임베딩 클라이언트 — 로컬 모델 + Milvus 직접 호출"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.embedding.embedding_model import EmbeddingModel
from app.embedding.milvus_client import MilvusManager

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """임베딩 모델과 Milvus를 직접 사용하는 클라이언트."""

    # ── 임베딩 ──

    async def embed(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩 벡터로 변환한다."""
        model = EmbeddingModel.get_instance()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.encode, text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩한다."""
        model = EmbeddingModel.get_instance()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.encode_batch, texts)

    # ── 컬렉션 관리 ──

    async def create_collection(
        self,
        name: str,
        dimension: int = 1024,
        description: str = "",
    ) -> Dict[str, str]:
        """Milvus 컬렉션을 생성한다."""
        milvus = await MilvusManager.get_instance()
        result = await milvus.create_collection(name=name, dim=dimension, description=description)
        return {"collection": result["collection"], "status": result["status"]}

    async def list_collections(self) -> List[str]:
        """모든 컬렉션 목록을 반환한다."""
        milvus = await MilvusManager.get_instance()
        return await milvus.list_collections()

    # ── 데이터 삽입 ──

    async def insert(
        self,
        collection_name: str,
        items: List[Dict[str, Any]],
        auto_embed: bool = True,
    ) -> Dict[str, Any]:
        """벡터 + 메타데이터를 컬렉션에 삽입한다."""
        if auto_embed:
            model = EmbeddingModel.get_instance()
            loop = asyncio.get_event_loop()
            texts = [item["text"] for item in items]
            vectors = await loop.run_in_executor(None, model.encode_batch, texts)
            items = [
                {**item, "embedding": vector}
                for item, vector in zip(items, vectors)
            ]

        milvus = await MilvusManager.get_instance()
        return await milvus.insert(collection_name=collection_name, data=items)

    # ── 검색 ──

    async def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """텍스트로 유사도 검색한다."""
        model = EmbeddingModel.get_instance()
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, model.encode, query)

        milvus = await MilvusManager.get_instance()
        raw = await milvus.search(
            collection_name=collection_name,
            query_vectors=[vector],
            limit=limit,
            output_fields=output_fields,
        )

        # raw[0]: 첫 번째 쿼리의 결과 목록, entity 필드를 평탄화
        hits = raw[0] if raw else []
        return [
            {
                "id": hit["id"],
                "distance": hit["distance"],
                "text": hit["entity"].get("text", ""),
                "category": hit["entity"].get("category", ""),
                "metadata": hit["entity"].get("metadata", {}),
            }
            for hit in hits
        ]

    async def search_by_vector(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """벡터로 직접 유사도 검색한다."""
        milvus = await MilvusManager.get_instance()
        raw = await milvus.search(
            collection_name=collection_name,
            query_vectors=[query_vector],
            limit=limit,
            output_fields=output_fields,
        )

        hits = raw[0] if raw else []
        return [
            {
                "id": hit["id"],
                "distance": hit["distance"],
                "text": hit["entity"].get("text", ""),
                "category": hit["entity"].get("category", ""),
                "metadata": hit["entity"].get("metadata", {}),
            }
            for hit in hits
        ]
