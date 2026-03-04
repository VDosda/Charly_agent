# Agent Memory Flow (v1)

Ce document decrit le flux memoire actuel entre runtime, distillation et stockage.

```mermaid
flowchart TD
  U["User message"] --> RH["core/runtime.py::handle_message"]

  subgraph INIT["Database init / migrations"]
    MIG["db/migrate.py::migrate"] --> M1["0001 chat_history"]
    MIG --> M2["0002 episodes + episode_sources"]
    MIG --> M3["0003 memory_items"]
    MIG --> M4["0004 vec tables (1024)"]
  end

  RH --> PUSER["_persist_turn(role=user)"]
  PUSER --> CH[(chat_history)]

  RH --> BLM["_build_llm_messages()"]
  BLM --> RST["_read_recent_turns(session, limit)"]
  RST --> CH
  RST -->|"WHERE archived=0 (fallback if missing column)" BLM

  RH --> BMB["_build_memory_bundle(user_message)"]

  subgraph RET["Smart retrieval path"]
    BMB --> RLT["retrieve_smart.retrieve_lt_smart(...)"]
    BMB --> RMT["retrieve_smart.retrieve_mt_smart(...)"]

    RLT --> SID1["_read_scoped_item_ids(user_id)"]
    RMT --> SID2["_read_scoped_episode_ids(user_id, session_id)"]
    SID1 --> MI[(memory_items)]
    SID2 --> EP[(episodes)]

    RLT --> EQ1["vector_store.embed_query()"]
    RMT --> EQ2["vector_store.embed_query()"]
    EQ1 --> EMB["EmbeddingProvider.embed()"]
    EQ2 --> EMB

    RLT --> QI["query_topk_item_ids(..., allowed_ids=scoped_ids)"]
    RMT --> QE["query_topk_episode_ids(..., allowed_ids=scoped_ids)"]
    QI --> MIV[(memory_items_vec)]
    QE --> EV[(episodes_vec)]

    RLT --> JI["Join rows from memory_items"]
    RMT --> JE["Join rows from episodes"]
    JI --> WDI["List[Tuple[MemoryItem, distance]]"]
    JE --> WDE["List[Tuple[Episode, distance]]"]
  end

  WDI --> RR1["scoring.rerank_items(...)"]
  WDE --> RR2["scoring.rerank_episodes(...)"]
  RR1 --> MB["models.MemoryBundle"]
  RR2 --> MB

  MB --> CTX["context.build_memory_context_blocks(bundle)"]
  CTX --> INS["messages[1:1] = mem_blocks"]
  INS --> LLM["LLM generate + optional tool loop"]

  LLM --> PASS["_persist_turn(role=assistant)"]
  PASS --> CH

  RH --> MCE["_maybe_create_episode(correlation_id, user_id, session_id)"]

  subgraph DISTILL["Distillation path"]
    MCE --> DMT["distill_mt.maybe_create_episode(...)"]
    DMT --> SMT["store_mt: count/read/insert episode"]
    SMT --> CH
    SMT --> EP
    SMT --> ES[(episode_sources)]

    DMT --> SLLM["LLM summary JSON"]
    DMT --> EPMB["pack_f32 + embedding_blob in episodes"]
    DMT --> UEV["vector_store.upsert_episode_vec(episode_id, vec)"]
    UEV --> EV

    DMT --> DLT["distill_lt.maybe_distill_profile_from_episode(...)"]
    DLT --> EPR["read episode payload"]
    EPR --> EP
    DLT --> LLLM["LLM LT items JSON"]
    DLT --> LTB["pack_f32 + embedding_blob in memory_items"]
    DLT --> ULT["store_lt.upsert_memory_item(...)"]
    ULT --> MI
    DLT --> UIV["vector_store.upsert_item_vec(item_id, vec)"]
    UIV --> MIV
  end

  RH --> CLN["_cleanup(session_id)"]
  CLN --> CST["cleanup.cleanup_st(...)"]
  CST --> CH
```
