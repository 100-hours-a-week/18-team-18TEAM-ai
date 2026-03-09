"""Milvus 벡터 DB 연결 및 컬렉션 관리"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from pymilvus import AsyncMilvusClient, DataType

logger = logging.getLogger(__name__)


class MilvusManager:
    """Milvus AsyncMilvusClient 싱글톤 래퍼."""

    _instance: MilvusManager | None = None

    @classmethod
    async def get_instance(cls) -> MilvusManager:
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.connect()
        return cls._instance

    def __init__(self) -> None:
        self.client: AsyncMilvusClient | None = None

    async def connect(self) -> None:
        """Milvus 서버에 비동기 연결한다."""
        uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        token = os.getenv("MILVUS_TOKEN", "")
        logger.info("Milvus 연결 중: %s", uri)
        try:
            kwargs: Dict[str, Any] = {"uri": uri}
            if token:
                kwargs["token"] = token
            self.client = AsyncMilvusClient(**kwargs)
            await self.client.list_collections()  # 실제 연결 확인
            logger.info("Milvus 연결 완료")
        except Exception as e:
            logger.error("Milvus 연결 실패: %s", e)
            self.client = None
            raise

    async def create_collection(
        self,
        name: str,
        dim: int = 1024,
        description: str = "",
    ) -> Dict[str, Any]:
        """컬렉션을 생성한다. 이미 존재하면 스킵."""
        collections = await self.client.list_collections()
        if name in collections:
            logger.info("컬렉션 '%s' 이미 존재함 - 스킵", name)
            return {"collection": name, "status": "already_exists"}

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)

        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=2048,
            description="원본 텍스트",
        )
        schema.add_field(
            field_name="category",
            datatype=DataType.VARCHAR,
            max_length=64,
            description="분류 카테고리",
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON,
            description="추가 메타데이터",
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
            description=f"{dim}차원 임베딩 벡터",
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        await self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            description=description,
        )

        logger.info("컬렉션 '%s' 생성 완료 (dim=%d)", name, dim)
        return {"collection": name, "status": "created"}

    async def insert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """벡터 + 메타데이터를 컬렉션에 삽입한다. 컬렉션이 없으면 자동 생성."""
        collections = await self.client.list_collections()
        if collection_name not in collections:
            logger.info("컬렉션 '%s' 없음 - 자동 생성", collection_name)
            await self.create_collection(name=collection_name)
        result = await self.client.insert(
            collection_name=collection_name,
            data=data,
        )
        logger.info(
            "컬렉션 '%s'에 %d건 삽입 완료",
            collection_name,
            len(data),
        )
        return {"insert_count": len(data), "ids": result.get("ids", [])}

    async def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """KNN 유사도 검색을 수행한다. 컬렉션이 없으면 ValueError 발생."""
        collections = await self.client.list_collections()
        if collection_name not in collections:
            raise ValueError(f"컬렉션 '{collection_name}'이 존재하지 않습니다.")

        if output_fields is None:
            output_fields = ["text", "category", "metadata"]

        results = await self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE"},
        )

        formatted = []
        for hits in results:
            hit_list = []
            for hit in hits:
                hit_list.append({
                    "id": hit["id"],
                    "distance": float(hit["distance"]),
                    "entity": hit["entity"],
                })
            formatted.append(hit_list)

        return formatted

    async def list_collections(self) -> List[str]:
        """모든 컬렉션 목록을 반환한다."""
        return await self.client.list_collections()

    async def drop_collection(self, name: str) -> Dict[str, str]:
        """컬렉션을 삭제한다."""
        await self.client.drop_collection(collection_name=name)
        logger.info("컬렉션 '%s' 삭제 완료", name)
        return {"collection": name, "status": "dropped"}
