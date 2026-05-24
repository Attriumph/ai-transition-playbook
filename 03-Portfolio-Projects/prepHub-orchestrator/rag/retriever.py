import os
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

class pgvectorRetriever:
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        
    def _get_connection(self):
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is missing.")
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    def retrieve_custom_context(
        self, 
        user_id: str, 
        query_embedding: List[float], 
        top_k: int = 5,
        target_role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a secure, tenant-isolated vector search using the pgvector L2 distance (<->) 
        or Cosine distance (<=>) operators. Row-level filters restrict results strictly to the 
        active user's database scope.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # SQL Schema Expectation:
        # Table: 'document_embeddings'
        # Columns: id (UUID), user_id (VARCHAR), content (TEXT), role_tag (VARCHAR), embedding (VECTOR)
        
        # Base query structure applying inner-product/cosine-distance operator (<=>)
        query = """
            SELECT 
                id, 
                content, 
                role_tag,
                (embedding <=> %s::vector) AS cosine_distance
            FROM document_embeddings
            WHERE user_id = %s
        """
        params = [query_embedding, user_id]
        
        # Add metadata pre-filtering dynamically to restrict vectors safely
        if target_role:
            query += " AND role_tag = %s"
            params.append(target_role)
            
        # Complete query with distance sorting and result capping
        query += " ORDER BY cosine_distance ASC LIMIT %s;"
        params.append(top_k)
        
        try:
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.commit()
            return [dict(row) for row in results]
        except Exception as e:
            conn.rollback()
            print(f"❌ Database error executing pgvector search query: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()

    def hybrid_search_rrf(
        self,
        user_id: str,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes a local multi-retrieval pipeline:
        1. BM25 exact keyword matching (via Postgres full-text indexing).
        2. pgvector semantic embedding matching.
        Combines outputs using Reciprocal Rank Fusion (RRF) with smoothing.
        """
        # Step 1: Retrieve via semantic distance
        dense_results = self.retrieve_custom_context(user_id, query_embedding, top_k=top_k * 2)
        
        # Step 2: Retrieve via standard SQL Full-Text Search (Sparse Keyword)
        conn = self._get_connection()
        cursor = conn.cursor()
        
        sparse_query = """
            SELECT 
                id, 
                content, 
                ts_rank_cd(to_tsvector('english', content), query) AS rank
            FROM document_embeddings, plainto_tsquery('english', %s) query
            WHERE user_id = %s AND to_tsvector('english', content) @@ query
            ORDER BY rank DESC
            LIMIT %s;
        """
        
        sparse_results = []
        try:
            cursor.execute(sparse_query, [query_text, user_id, top_k * 2])
            sparse_results = cursor.fetchall()
        except Exception as e:
            print(f"❌ Database error executing BM25 text search: {str(e)}")
        finally:
            cursor.close()
            conn.close()
            
        # Step 3: Run Reciprocal Rank Fusion (RRF)
        # Convert lists of Dicts into simple lists of IDs for scoring
        dense_ids = [str(row["id"]) for row in dense_results]
        sparse_ids = [str(row["id"]) for row in sparse_results]
        
        # Build dictionary to hold scores
        rrf_scores: Dict[str, float] = {}
        id_to_content: Dict[str, str] = {}
        
        # Map IDs to original contents for reconstruction
        for row in dense_results:
            id_to_content[str(row["id"])] = row["content"]
        for row in sparse_results:
            id_to_content[str(row["id"])] = row["content"]
            
        k = 60
        # Score Dense
        for rank, doc_id in enumerate(dense_ids, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
            
        # Score Sparse
        for rank, doc_id in enumerate(sparse_ids, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
            
        # Reconstruct and sort final merged outputs
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        fused_results = []
        for doc_id, score in sorted_ids:
            fused_results.append({
                "id": doc_id,
                "content": id_to_content[doc_id],
                "rrf_score": score
            })
            
        return fused_results
