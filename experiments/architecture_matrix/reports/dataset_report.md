# Jarvis 1.0B Token Pre-Training Dataset Acquisition Report

## STATUS: **READY FOR 0.8–1.0B TOKEN TRAINING**

### 1. Executive Summary & Provenance
- **Corpus Name:** FineWeb-Edu (Sample-10BT curated subset)
- **License:** `Open Data Commons Attribution (ODC-By 1.0) / CC-BY-4.0`
- **Tokenizer:** `tiktoken / gpt2` (`vocab_size=50257`)
- **Token Target:** 1,000,000,000 tokens
- **Total Usable Tokens Acquired:** **1,001,588,263 tokens**
- **Total Shards Emitted:** 21 binary shards

### 2. Token & Shard Breakdown

| Split | Tokens | Percentage | Binary Shards | Format |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | **1,000,000,000** | 99.84% | **20** | uint16 binary array (50M tokens/shard) |
| **Validation** | **1,588,263** | 0.16% | **1** | uint16 binary array |
| **Total Usable** | **1,001,588,263** | **100.00%** | **21** | **1.87 GB total** |

### 3. Verification & Integrity Checklist
- [x] Correct tokenizer verified (`gpt2` tiktoken, vocab 50257)
- [x] Document boundary delimiter preserved (`<|endoftext|>`, token ID 50256)
- [x] Exactly 20 full training shards (50,000,000 tokens / 100,000,000 bytes each)
- [x] Dedicated validation shard (zero document leakage)
- [x] Shard streaming dataloader verified (`data/streaming_dataloader.py`)

**Conclusion:** The pre-training corpus meets all scientific requirements and is strictly prepared for the upcoming billion-token training sprint.
