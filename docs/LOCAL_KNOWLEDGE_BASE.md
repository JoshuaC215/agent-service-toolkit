# Local Knowledge Base RAG

The toolkit now includes a local document ingestion path for the `local-rag-agent`.
It is separate from the existing AWS Knowledge Base agent and the example Chroma tool.

## Flow

```text
UploadFile
  -> PDF/DOCX/Markdown/TXT extraction
  -> RecursiveCharacterTextSplitter
  -> Chroma collection per knowledge_base_id
  -> local-rag-agent retrieval
  -> answer with source and page footer
```

## Configuration

The default storage directory is `data/knowledge_bases`. The default embedding model is
OpenAI `text-embedding-3-small`.

For an offline demo or test run, set:

```env
USE_FAKE_MODEL=true
```

This uses LangChain's deterministic fake embedding and is not suitable for production
semantic search. For real retrieval, set `USE_FAKE_MODEL=false` and provide
`OPENAI_API_KEY`.

The service limits documents to 10 MB by default and uses 1000-character chunks with a
150-character overlap. These values can be changed with the `KNOWLEDGE_*` settings.

## Upload a document

Start the FastAPI service, then upload a document with PowerShell:

```powershell
$form = @{
  file = Get-Item .\docs\employee-handbook.md
}
Invoke-RestMethod `
  -Uri http://127.0.0.1:8080/knowledge-bases/handbook/documents `
  -Method Post `
  -Form $form
```

The response includes the knowledge-base ID, document ID, file size, and number of
indexed chunks.

## Inspect retrieval

```powershell
Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8080/knowledge-bases/handbook/search?q=remote%20work&k=4'
```

Search results include the chunk content, distance score, document ID, source filename,
and page number.

## Ask the Agent

Pass the knowledge-base ID through `agent_config` and select `local-rag-agent`:

```powershell
$body = @{
  message = "What is the remote work policy?"
  user_id = "demo-user"
  thread_id = "demo-thread"
  agent_config = @{
    knowledge_base_id = "handbook"
    top_k = 4
  }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri http://127.0.0.1:8080/local-rag-agent/invoke `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body
```

The Agent receives only the retrieved document chunks in its context. It appends a
deterministic `Sources` section to the answer so the response remains traceable even
when the model does not format citations itself.

## Supported formats and API

- `POST /knowledge-bases/{knowledge_base_id}/documents`: upload and index one file.
- `GET /knowledge-bases/{knowledge_base_id}/search?q=...&k=...`: inspect retrieved chunks.
- `POST /local-rag-agent/invoke`: answer using a knowledge base.
- `POST /local-rag-agent/stream`: stream the grounded answer over SSE.
- Supported files: `.txt`, `.md`, `.pdf`, and `.docx`.
