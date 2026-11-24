# Extração Automática de Endereços — Applied Computer Vision (Substitutiva GS 2026.2)

- **Instituição:** FIAP — Engenharia de Software (4º Ano | Turma 4ESPR)
- **Disciplina:** Applied Computer Vision
- **Integrante:** Lucas de Freitas Pagung — **RM 553242**
- **Aplicação publicada:** https://sub-cv-zelafacil-rm553242-4s4tczifgpb95dwxmhvgkn.streamlit.app/
- **Repositório Git:** https://github.com/lfspagung/sub-cv-zelafacil-rm553242


## Visão Geral
Sistema completo de Visão Computacional e OCR para extração automática do endereço do titular a partir de fotografias de comprovantes de residência de **3 empresas reais** (**Enel Distribuição**, **Sabesp Saneamento** e **Claro Telecomunicações**), utilizando **documentos 100% sintéticos** em conformidade com a LGPD (carimbados com `"DOCUMENTO SINTÉTICO"` e sem nenhum dado pessoal real).

### Etapas do Pipeline (`src/pipeline.py`)
1. **Localização do documento e correção de perspectiva (OpenCV):** normalização CLAHE + limiarização Otsu/Canny + detecção do quadrilátero principal + transformação homográfica (`cv2.warpPerspective`).
2. **Pré-processamento OpenCV automático:** upscaling para largura ótima de OCR, normalização de iluminação por divisão de fundo Gaussiano (remoção de sombras), filtro bilateral para redução de ruído, CLAHE e binarização de Otsu.
3. **OCR com confiança por token:** `pytesseract.image_to_data` em português (`lang="por"`, `--oem 3 --psm 6`).
4. **Identificação automática da empresa:** pontuação por palavras-chave características do layout (sem entrada manual).
5. **Isolamento do endereço do titular:** descarte explícito de endereços concorrentes (sede corporativa, agência/loja de atendimento, endereço de cobrança) e extração estruturada dos campos (`logradouro`, `numero`, `complemento`, `bairro`, `cidade`, `uf`, `cep`).
6. **Validação por regras e sinalização de campos duvidosos:** verificação de obrigatoriedade, formato de CEP (`00000-000`), lista das 27 UFs válidas e confiança média dos tokens OCR por campo.

## Estrutura do Repositório
```
├── app.py                        # Aplicação Web Streamlit (upload, destaque de dúvidas, confirmação JSON)
├── src/
│   └── pipeline.py               # Pipeline completo OpenCV + Tesseract OCR + Validação/Confiança
├── scripts/
│   ├── gerar_dataset.py          # Gera 30 imagens sintéticas (15 dev + 15 test) com gabaritos JSON
│   └── avaliar_modelo.py         # Mede métricas no conjunto de teste (com vs. sem pré-processamento e confiança)
├── dataset/
│   ├── dev/                      # 15 imagens de desenvolvimento (5 por empresa) + gabarito.json
│   └── test/                     # 15 imagens de teste (5 por empresa, inéditas) + gabarito.json
├── resultados/                   # Relatório JSON, resumo Markdown e gráficos da avaliação
├── requirements.txt              # Dependências Python
└── packages.txt                  # Dependências de sistema (Tesseract OCR para Streamlit Community Cloud)
```

## Como Executar Localmente

### 1. Instalar o Tesseract OCR (com idioma português)
- **macOS (Homebrew):**
  ```bash
  brew install tesseract tesseract-lang
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-por
  ```

### 2. Criar ambiente virtual e instalar dependências
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Opcional) Regerar o dataset sintético (15 dev + 15 test)
```bash
python scripts/gerar_dataset.py
```

### 4. Executar a avaliação quantitativa no conjunto de teste
```bash
python scripts/avaliar_modelo.py
```

### 5. Executar a aplicação web Streamlit
```bash
streamlit run app.py
```
