# 2ndBrain v2

Local, offline-first engineering knowledge base, persistent memory ledger, and CPU System 1 router for Windows (Python 3.13+).

> **Upgrading from v1?** The previous v1 monorepo is archived at tag [`v1-archive`](../../tree/v1-archive). v2 separates the public engine repository from your external Obsidian Vault and local SQLite/ONNX runtime state so raw source documents and personal memories never enter Git.

## Architecture (Zero-Leak Vault Isolation)

```mermaid
flowchart LR
    Repo["2ndBrain (Git Repo)<br/>src/ + scripts/setup.py + training/"] -->|"scripts/setup.py"| Venv[".venv + pip install -e ."]
    Venv -->|"brain init"| Vault["External Live Vault<br/>(%USERPROFILE%\\2ndBrain_Vault)"]
    Venv -->|"brain model pull"| State["Local Runtime State<br/>(%LOCALAPPDATA%\\2ndBrain\\state)"]
    State --> Daemon["braind Daemon<br/>(Windows Named Pipe + SQLite FTS5/vec0)"]
    Daemon <-->|"FD-1 isolated stdio"| MCP["brain-mcp.exe<br/>(Antigravity / Claude / Cursor)"]
```

- **Engine Repo (`2ndBrain/`):** Contains source code, `scripts/setup.py`, generic starter templates (`src/second_brain/templates/vault/`), and Laya fine-tuning scripts (`training/`).
- **External Vault (`%USERPROFILE%\2ndBrain_Vault`):** Holds your raw PDFs/docs (`raw/`), `specialists/*.yaml`, `memory/*.md`, `ontology/taxonomy.yaml`, and compiled `domains/` outside the engine repo.
- **Local State (`%LOCALAPPDATA%\2ndBrain\state`):** Holds `brain.sqlite`, parsed text chunks, visual crops, logs, and ONNX models (`laya_stock`, FastEmbed embedder/reranker). Never committed to Git.

---

## Requirements

- **OS:** Windows 10/11 (`win32` Named Pipe IPC + Windows ACLs)
- **Python:** Python 3.13+ (full installer from python.org, not the Microsoft Store stub)
- **Disk:** ~700 MB free for stock ONNX models (`laya_stock`, `BAAI/bge-small-en-v1.5`, `Xenova/ms-marco-MiniLM-L-6-v2`)

---

## Quickstart (Auto-Detect Setup)

1. Clone the repository:
   ```powershell
   git clone https://github.com/Fabbbbbbb99/2ndBrain.git
   cd 2ndBrain
   ```
2. Run the auto-detect installer (creates `.venv`, installs dependencies, initializes an external Vault at `%USERPROFILE%\2ndBrain_Vault`, writes `%LOCALAPPDATA%\2ndBrain\config.toml`, pulls & SHA-256-verifies stock models, configures Git leak-guard hooks, generates ready-to-paste MCP snippets in `configs/generated/`, and runs `brain doctor --stdio-check`):
   ```powershell
   py -3.13 scripts/setup.py
   ```
   Or via PowerShell wrapper:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```
3. *(Optional)* Auto-register Antigravity MCP & export starter skills (creates a timestamped backup of `~/.gemini/config/mcp_config.json` before merging `second-brain` and runs `brain export-skills --no-prune`):
   ```powershell
   py -3.13 scripts/setup.py --register-mcp
   ```

### Useful Setup Flags
- `--vault <PATH>` / `--state <PATH>` / `--config <PATH>`: Override default locations.
- `--skip-models`: Skip model downloads during setup (run `.venv\Scripts\brain.exe model pull` later before querying).
- `--skip-fastembed`: Download `laya_stock` only (`~506 MB`) and let FastEmbed download on first use (`~157 MB`).
- `--allow-onedrive`: Allow OneDrive-synced Vault/state paths (disabled by default due to file-locking risks).
- `--force`: Overwrite an existing `config.toml` that points to a different Vault.

---

## Connecting AI Clients

By default, `scripts/setup.py` writes ready-to-paste JSON configs with your detected paths under `configs/generated/` (gitignored):
- `configs/generated/antigravity.local.json`
- `configs/generated/claude_desktop.local.json`
- `configs/generated/cursor.local.json`

For Claude Desktop or Cursor custom instructions, see [configs/AGENT_INSTRUCTIONS.example.md](configs/AGENT_INSTRUCTIONS.example.md).

---

## Using Your 2ndBrain

### 1. Define a Domain Specialist
Rename or duplicate `specialists/example_domain.yaml` inside your external Vault (`%USERPROFILE%\2ndBrain_Vault\specialists\<your_domain>.yaml`) and register matching domain aliases in `ontology/taxonomy.yaml`, then sync the manifest hash:
```powershell
.venv\Scripts\brain.exe init --sync-manifest
```

### 2. Ingest Documents
Place raw documents inside `<VAULT>\raw\<your_domain>\` and ingest them into SQLite:
```powershell
.venv\Scripts\brain.exe ingest pdf "<VAULT>\raw\<domain>\<file>.pdf" --domain <domain> --source-id <SOURCE_ID> --version <VER>
.venv\Scripts\brain.exe ingest docs "<VAULT>\raw\<domain>\docs" --domain <domain> --source-id <SOURCE_ID> --version <VER>
```

### 3. Manage Daemon, Skills & Memory
```powershell
.venv\Scripts\brain.exe doctor --stdio-check
.venv\Scripts\brain.exe daemon start
.venv\Scripts\brain.exe query "your technical question"
.venv\Scripts\brain.exe certify
.venv\Scripts\brain.exe export-skills
.venv\Scripts\brain.exe graph-export --open
```

---

## Optional: Purpose-Training Laya (Multi-Domain Routing)

With 1 specialist, System 1 short-circuits routing at `0 ms`. Once you have 2+ specialists, stock zero-shot Laya routes across them automatically, or you can fine-tune a custom INT8 ONNX checkpoint on Kaggle/Colab:

1. Build JSONL splits (`train.jsonl`, `val.jsonl`, `calibration.jsonl`):
   ```powershell
   .venv\Scripts\python.exe training/build_dataset.py --out-dir data/laya_splits
   ```
2. Fine-tune using [training/laya_finetune_kaggle.ipynb](training/laya_finetune_kaggle.ipynb) and export via `training/export_int8_onnx.py`.
3. Swap the signed checkpoint in locally:
   ```powershell
   .venv\Scripts\brain.exe model swap --model-dir "<PATH_TO_EXPORTED_BUNDLE>"
   ```

---

## License

[MIT](LICENSE)
