"""임베딩 서비스 HTTP 클라이언트"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """별도 임베딩 서비스(port 8100)와 HTTP로 통신하는 클라이언트."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "EMBEDDING_SERVICE_URL", "http://localhost:8100"
        )
        self.timeout = float(os.getenv("EMBEDDING_SERVICE_TIMEOUT", "3.0"))

    async def health(self) -> Dict[str, Any]:
        """임베딩 서비스 헬스체크."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    # ── 임베딩 ──

    async def embed(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩 벡터로 변환한다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/embed",
                json={"text": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩한다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/embed/batch",
                json={"texts": texts},
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]

    # ── 컬렉션 관리 ──

    async def create_collection(
        self,
        name: str,
        dimension: int = 1024,
        description: str = "",
    ) -> Dict[str, str]:
        """Milvus 컬렉션을 생성한다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/collection/create",
                json={
                    "name": name,
                    "dimension": dimension,
                    "description": description,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def list_collections(self) -> List[str]:
        """모든 컬렉션 목록을 반환한다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/collection/list")
            resp.raise_for_status()
            return resp.json()["collections"]

    # ── 데이터 삽입 ──

    async def insert(
        self,
        collection_name: str,
        items: List[Dict[str, Any]],
        auto_embed: bool = True,
    ) -> Dict[str, Any]:
        """벡터 + 메타데이터를 컬렉션에 삽입한다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/collection/{collection_name}/insert",
                json={"items": items, "auto_embed": auto_embed},
            )
            resp.raise_for_status()
            return resp.json()

    # ── 검색 ──

    async def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """텍스트로 유사도 검색한다."""
        payload: Dict[str, Any] = {"query": query, "limit": limit}
        if output_fields:
            payload["output_fields"] = output_fields

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/collection/{collection_name}/search",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["results"]

    async def search_by_vector(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """벡터로 직접 유사도 검색한다."""
        payload: Dict[str, Any] = {
            "query_vector": query_vector,
            "limit": limit,
        }
        if output_fields:
            payload["output_fields"] = output_fields

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/collection/{collection_name}/search/vector",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["results"]
