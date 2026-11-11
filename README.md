# CV Match Agent

MVP de API para comparar um currículo PDF com uma vaga usando RAG, Gemini, PostgreSQL 18 e pgvector.

## Arquitetura

- **FastAPI** recebe e valida requisições HTTP.
- **Services** extraem o texto PDF, criam chunks, chamam embeddings/LLM e calculam o score.
- **SQLAlchemy + Alembic** persistem currículos e chunks. A migration ativa a extensão vector.
- **pgvector** calcula distância cosseno no banco, filtrando pelo currículo e limitando a TOP_K.
- **Pydantic** valida configurações, requests e resposta JSON do modelo.

## Fluxo RAG

1. POST /api/v1/cvs/upload valida o PDF e extrai texto com pypdf.
2. O texto é dividido por limite de caracteres em chunks com sobreposição configurável.
3. gemini-embedding-001 produz embeddings de documento em EMBEDDING_DIMENSIONS (3072 por padrão).
4. Chunks e vetores são persistidos no PostgreSQL.
5. POST /api/v1/analysis embeda a vaga como consulta, recupera os TOP_K trechos mais próximos por distância cosseno e fornece apenas esses trechos ao Gemini.
6. gemini-3.7-flash retorna requisitos estruturados. A API valida evidências e calcula o score: found=1, partial=0.5, not_found=0, média dos requisitos, arredondada para 0–100. Se o modelo não identificar requisitos, o score é 0.

Os embeddings do Gemini são vetores densos. pgvector armazena-os e executa a ordenação no banco, sem carregar todos os vetores na aplicação. O MVP usa busca exata sem índice aproximado; isso mantém o ranking direto e é adequado a volumes pequenos. A dimensão 3072 é a saída padrão documentada para gemini-embedding-001; ela cabe no tipo vector, mas não em índices HNSW/IVFFlat tradicionais de vector, por isso nenhum índice aproximado é criado.

## Configuração

1. Python 3.12 ou superior.
2. Instale as dependências: python -m pip install -r requirements.txt.
3. Preencha GEMINI_API_KEY no .env existente. Não compartilhe nem registre o segredo.
4. Confira DATABASE_URL: o arquivo .env atual neste checkout aponta para a porta 5432. O Compose abaixo publica o banco em 5433. Para conectar a API ao container, ajuste manualmente somente o número da porta no seu .env para 5433, preservando as demais variáveis. O .env foi mantido intacto durante a implementação.

O .env.example mostra os nomes e valores locais de exemplo; ele não é carregado automaticamente nem substitui .env. A aplicação valida GEMINI_API_KEY, DATABASE_URL, faixas de TOP_K, CHUNK_SIZE, CHUNK_OVERLAP e EMBEDDING_DIMENSIONS ao iniciar. É possível configurar os modelos por GEMINI_MODEL e GEMINI_EMBEDDING_MODEL.

## PostgreSQL e Docker

O docker-compose.yml usa pgvector/pgvector:0.8.7-pg18-trixie, publica apenas em 127.0.0.1:5433, persiste dados no volume nomeado pgdata e monta o volume no caminho compatível com PostgreSQL 18 (/var/lib/postgresql).

    docker compose up -d
    docker compose ps
    docker compose exec db pg_isready -U postgres -d cv_match
    docker compose exec db psql -U postgres -d cv_match -c "CREATE EXTENSION IF NOT EXISTS vector;"
    docker compose exec db psql -U postgres -d cv_match -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"

O compose cria somente o serviço deste projeto. Não use docker compose down -v se deseja preservar os dados.

## Migrations e API

Execute a partir da raiz do projeto:

    python -m alembic upgrade head
    uvicorn app.main:app --reload

Health check: GET http://localhost:8000/health. Documentação interativa: http://localhost:8000/docs.

## Endpoints

### Upload de currículo

    curl.exe -X POST http://localhost:8000/api/v1/cvs/upload -F "file=@curriculo.pdf"

Resposta:

    {
      "resume_id": "uuid",
      "filename": "curriculo.pdf",
      "chunks_created": 4
    }

### Análise

    curl.exe -X POST http://localhost:8000/api/v1/analysis -H "Content-Type: application/json" -d '{"resume_id":"UUID_DO_UPLOAD","job_description":"Buscamos engenheiro backend com experiência em Python, APIs REST, PostgreSQL e testes automatizados."}'

A resposta contém compatibility_score, resumo, pontos fortes, requisitos sem evidência, análise requisito a requisito com evidências e recomendações. “Não encontrado” descreve somente o texto do currículo e não afirma que o candidato não possui a competência.

## pgAdmin

Conecte a localhost, porta 5433, banco cv_match, usuário postgres, senha postgres. Essas credenciais são somente para desenvolvimento local.

## Testes

    pytest

Os testes unitários não precisam chamar a API real. Para validar persistência e recuperação em integração, inicie o Compose, ajuste DATABASE_URL para 5433 e execute as migrations antes de chamar os endpoints.

## Decisões e limites conhecidos

- Sem LangChain/LlamaIndex: o fluxo de chunking, embeddings, retrieval e geração fica explícito.
- Sem guardar o PDF original: apenas o nome e chunks são persistidos.
- Embedding em 3072 dimensões conforme saída padrão verificada do modelo; ajuste essa dimensão antes de criar a primeira migration se precisar reduzir armazenamento. Uma mudança posterior requer reprocessar currículos e migrar a coluna.
- Score transparente com peso igual por requisito identificado pelo LLM; não é uma métrica estatística de contratação.
- A saída estruturada é validada por Pydantic, e requisitos found/partial precisam incluir evidência.
- Para produção, revisar autenticação, limites de upload, logging/retention, política de privacidade e estratégia de índice/volume.
