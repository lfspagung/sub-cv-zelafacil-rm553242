"""
Pipeline de Visão Computacional e OCR para Extração Automática de Endereços
Disciplina: Applied Computer Vision (Substitutiva GS 2026.2)
Autor: Lucas de Freitas Pagung — RM 553242
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None
    Output = None


UFS_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

CAMPOS_OBRIGATORIOS = ["logradouro", "numero", "bairro", "cidade", "uf", "cep"]
TODOS_CAMPOS = ["logradouro", "numero", "complemento", "bairro", "cidade", "uf", "cep"]


def _ordenar_pontos(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def _transformar_perspectiva(img_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _ordenar_pontos(pts.reshape(4, 2).astype("float32"))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(int(height_a), int(height_b))

    if max_w < 200 or max_h < 200:
        return img_bgr

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, matrix, (max_w, max_h))
    return warped


def detectar_e_corrigir_perspectiva(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Localiza automaticamente a folha completa do documento na foto (incluindo
    cabeçalhos coloridos escuros como o azul da Enel e o vermelho da Claro) e
    aplica correção de perspectiva homográfica.
    Retorna: (imagem_corrigida_bgr, imagem_debug_contorno_bgr, perspectiva_aplicada)
    """
    h, w = img_bgr.shape[:2]
    debug_img = img_bgr.copy()

    escala = 900.0 / max(h, w) if max(h, w) > 900 else 1.0
    small = cv2.resize(img_bgr, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    # Estratégia 1 (principal): distância de cor CIELAB em relação às bordas (fundo da mesa).
    # Preserva faixas/cabeçalhos coloridos escuros (azul Enel, vermelho Claro, azul Sabesp)
    # que poderiam se confundir com o fundo em luminância pura.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(8, min(sh, sw) // 35)
    amostras_fundo = np.concatenate(
        [
            lab[:b, :, :].reshape(-1, 3),
            lab[-b:, :, :].reshape(-1, 3),
            lab[:, :b, :].reshape(-1, 3),
            lab[:, -b:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    mediana_fundo = np.median(amostras_fundo, axis=0)
    # Pondera mais os canais de croma (a*, b*) para que cabeçalhos coloridos escuros
    # (ex.: azul Enel, vermelho Claro) se destaquem nitidamente da superfície de fundo.
    pesos = np.array([1.0, 1.8, 1.8], dtype=np.float32)
    dist = np.linalg.norm((lab - mediana_fundo[None, None, :]) * pesos[None, None, :], axis=2)
    dist_borda = np.linalg.norm((amostras_fundo - mediana_fundo[None, :]) * pesos[None, :], axis=1)
    limiar_fundo = max(22.0, float(np.percentile(dist_borda, 99)) * 1.65)
    dist_blur = cv2.GaussianBlur(dist, (5, 5), 0)
    mask_lab = np.where(dist_blur > limiar_fundo, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask_lab = cv2.morphologyEx(mask_lab, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_lab = cv2.morphologyEx(mask_lab, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    # Estratégia 2 (fallback): bordas Canny no canal de distância de cor + cinza
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)

    melhor_quad = None
    melhor_area = 0.0

    for mascara in (mask_lab, edges):
        contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        for c in contours:
            area = cv2.contourArea(c)
            frac = area / float(sh * sw)
            if frac < 0.22 or frac > 0.985:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.015, 0.02, 0.03, 0.04, 0.05):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if area > melhor_area:
                        melhor_area = area
                        melhor_quad = approx
                    break
        if melhor_quad is not None:
            break

    if melhor_quad is None:
        contours, _ = cv2.findContours(mask_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            frac = cv2.contourArea(c) / float(sh * sw)
            if 0.22 <= frac <= 0.985:
                box = cv2.boxPoints(cv2.minAreaRect(c))
                melhor_quad = np.int32(box).reshape(4, 1, 2)

    if melhor_quad is not None:
        pts_orig = melhor_quad.reshape(4, 2).astype("float32") / escala
        pts_orig[:, 0] = np.clip(pts_orig[:, 0], 0, w - 1)
        pts_orig[:, 1] = np.clip(pts_orig[:, 1], 0, h - 1)
        cv2.drawContours(debug_img, [pts_orig.astype(np.int32)], -1, (0, 200, 70), 4)
        warped = _transformar_perspectiva(img_bgr, pts_orig)
        return warped, debug_img, True

    return img_bgr, debug_img, False


def preprocessar_imagem(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pré-processamento OpenCV automático (sem ajuste manual por imagem):
    1) Redimensionamento para resolução ótima de OCR (largura mínima ~1800 px)
    2) Escala de cinza
    3) Normalização de iluminação (divisão pelo fundo Gaussiano para remover sombras)
    4) Filtro bilateral suave (reduz ruído preservando bordas das letras)
    5) CLAHE + binarização Otsu + borda branca
    """
    h, w = img_bgr.shape[:2]
    if w < 1800:
        fator = min(2.5, 1900.0 / max(w, 1))
        img_bgr = cv2.resize(img_bgr, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    ksize = max(31, (min(gray.shape[:2]) // 25) | 1)
    fundo = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    fundo = np.clip(fundo, 1, 255).astype(np.float32)
    normalizada = np.clip((gray.astype(np.float32) / fundo) * 245.0, 0, 255).astype(np.uint8)

    suavizada = cv2.bilateralFilter(normalizada, d=5, sigmaColor=40, sigmaSpace=40)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    realcada = clahe.apply(suavizada)

    _, binarizada = cv2.threshold(realcada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    com_borda = cv2.copyMakeBorder(binarizada, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
    return com_borda


def executar_ocr(img_input: np.ndarray, psm: int = 6) -> Tuple[str, List[Dict[str, Any]], float]:
    """
    Executa Tesseract OCR e retorna:
    - texto_completo
    - lista de tokens com confiança individual [{'text': str, 'conf': float}, ...]
    - confiança média dos tokens válidos (0 a 100)
    """
    if pytesseract is None:
        raise RuntimeError("pytesseract não está instalado neste ambiente.")

    cfg = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(img_input, lang="por", config=cfg, output_type=Output.DICT)
    except Exception:
        data = pytesseract.image_to_data(img_input, config=cfg, output_type=Output.DICT)

    linhas_dict: Dict[Tuple[int, int, int], List[str]] = {}
    tokens: List[Dict[str, Any]] = []
    confs_validas: List[float] = []

    n = len(data["text"])
    for i in range(n):
        txt = str(data["text"][i]).strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if txt:
            chave = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            linhas_dict.setdefault(chave, []).append(txt)
            if conf >= 0:
                tokens.append({"text": txt, "conf": conf})
                confs_validas.append(conf)

    linhas = [" ".join(palavras) for _, palavras in sorted(linhas_dict.items())]
    texto_completo = "\n".join(linhas)
    conf_media = float(np.mean(confs_validas)) if confs_validas else 0.0
    return texto_completo, tokens, round(conf_media, 2)


def normalizar_texto(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.upper().strip()


def identificar_empresa(texto_ocr: str) -> Tuple[str, float]:
    """
    Identifica automaticamente a empresa emissora entre as 3 cadastradas,
    sem nenhuma entrada manual do usuário.
    """
    tu = normalizar_texto(texto_ocr)
    scores = {
        "Enel Distribuição": 0,
        "Sabesp Saneamento": 0,
        "Claro Telecomunicações": 0,
    }

    for kw, peso in [
        ("ENEL", 4), ("ENERGIA ELETRICA", 3), ("KWH", 2),
        ("DISTRIBUICAO SAO PAULO", 3), ("CONTA DE ENERGIA", 2), ("INSTALACAO", 1),
    ]:
        if kw in tu:
            scores["Enel Distribuição"] += peso

    for kw, peso in [
        ("SABESP", 4), ("SANEAMENTO BASICO", 3), ("AGUA E ESGOTO", 3),
        ("HIDROMETRO", 2), ("RGI", 2), ("CONSUMO M3", 1), ("M3", 1),
    ]:
        if kw in tu:
            scores["Sabesp Saneamento"] += peso

    for kw, peso in [
        ("CLARO", 4), ("TELECOMUNICACOES", 3), ("FATURA MENSAL", 2),
        ("BANDA LARGA", 2), ("TELEFONIA", 2), ("ASSINANTE", 1), ("MOVEL", 1),
    ]:
        if kw in tu:
            scores["Claro Telecomunicações"] += peso

    melhor_empresa = max(scores, key=scores.get)
    melhor_score = scores[melhor_empresa]
    if melhor_score == 0:
        return "Não identificada", 0.0
    confianca = min(100.0, 55.0 + melhor_score * 8.0)
    return melhor_empresa, round(confianca, 1)


def _limpar_valor(v: str) -> str:
    v = re.sub(r"\s+", " ", (v or "")).strip(" :;-|,.")
    return v


def _extrair_cep_texto(texto: str) -> str:
    m = re.search(r"CEP[:\s\-]*([0-9OoIl|BbSsZz\-\.\s–—]{8,14})", texto, re.IGNORECASE)
    if m:
        bruto = m.group(1).translate(str.maketrans("OoIl|BbSsZz", "00111885522"))
        digitos = re.sub(r"\D", "", bruto)
        if len(digitos) >= 8:
            return f"{digitos[:5]}-{digitos[5:8]}"
    m2 = re.search(r"\b(\d{5})[\s\-\.–—]?(\d{3})\b", texto)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"
    return ""


def _filtrar_bloco_titular(texto_ocr: str, empresa: str) -> str:
    """
    Remove endereços concorrentes (sede corporativa, agência, loja, cobrança,
    central de atendimento, rodapés institucionais) e isola o bloco do titular.
    """
    linhas = [l.strip() for l in texto_ocr.splitlines() if l.strip()]

    termos_descarte = [
        "SEDE CORPORATIVA", "MATRIZ", "CNPJ", "AGENCIA DE ATENDIMENTO",
        "AGENCIA REGIONAL", "LOJA DE ATENDIMENTO", "POSTO DE ATENDIMENTO",
        "ENDERECO DE COBRANCA", "ENDERECO DA SEDE", "OUVIDORIA",
        "CENTRAL DE RELACIONAMENTO", "BARUERI", "COSTA CARVALHO",
        "HENRI DUNANT", "NACOES UNIDAS", "SUMIDOURO",
        "PAULISTA, 1106", "PINHEIROS", "CHACARA SANTO ANTONIO",
        "DOCUMENTO SINTETICO", "SEM DADOS PESSOAIS",
    ]

    ancoras_inicio = [
        "DADOS DO TITULAR", "ENDERECO DO TITULAR", "ENDERECO DA UNIDADE",
        "LOCAL DE FORNECIMENTO", "ENDERECO DO IMOVEL", "ENDERECO DE INSTALACAO",
        "TITULAR:", "CLIENTE:", "ASSINANTE:",
    ]
    ancoras_fim = [
        "RESUMO DA FATURA", "DETALHAMENTO", "HISTORICO DE CONSUMO",
        "ENDERECO DE COBRANCA", "AGENCIA DE ATENDIMENTO", "LOJA DE ATENDIMENTO",
        "SEDE CORPORATIVA", "DADOS DA EMPRESA", "TOTAL A PAGAR",
        "CODIGO DE BARRAS", "DOCUMENTO SINTETICO",
    ]

    idx_ini = -1
    for i, l in enumerate(linhas):
        ln = normalizar_texto(l)
        if any(a in ln for a in ancoras_inicio):
            idx_ini = i
            break

    if idx_ini != -1:
        secao = []
        for l in linhas[idx_ini:]:
            ln = normalizar_texto(l)
            if len(secao) >= 2 and any(f in ln for f in ancoras_fim):
                break
            if not any(t in ln for t in termos_descarte):
                secao.append(l)
        if secao:
            return "\n".join(secao)

    filtradas = [
        l for l in linhas
        if not any(t in normalizar_texto(l) for t in termos_descarte)
    ]
    return "\n".join(filtradas)


def _separar_logradouro_numero_complemento(linha_end: str) -> Tuple[str, str, str]:
    txt = _limpar_valor(linha_end)
    txt = re.sub(
        r"^(?:ENDERE[CÇ]O(?:\s+DO\s+\w+)?|LOGRADOURO)[:\s\-]*",
        "",
        txt,
        flags=re.IGNORECASE,
    ).strip()

    complemento = ""
    m_comp_rot = re.search(r"(?:COMPL(?:EMENTO)?\.?[:\s\-]+)([^|/\n]+)$", txt, re.IGNORECASE)
    if m_comp_rot:
        complemento = _limpar_valor(m_comp_rot.group(1))
        txt = txt[:m_comp_rot.start()].strip(" ,;-|")

    m = re.search(
        r"^(.+?)[,\s]+(?:N[º°o.]?\s*)?(\d{1,5}[A-Z]?)(?:\s*[-,|/]\s*(.+))?$",
        txt,
        re.IGNORECASE,
    )
    if m:
        logradouro = _limpar_valor(m.group(1))
        numero = _limpar_valor(m.group(2))
        if m.group(3) and not complemento:
            comp_cand = _limpar_valor(m.group(3))
            if not re.search(r"\b(?:BAIRRO|CEP|CIDADE)\b", comp_cand, re.IGNORECASE):
                complemento = comp_cand
        return logradouro, numero, complemento

    return txt, "", complemento


def extrair_campos_endereco(texto_ocr: str, empresa: str) -> Tuple[Dict[str, str], str]:
    """
    Extrai os campos estruturados do endereço do titular a partir do texto OCR,
    garantindo descarte de endereços concorrentes.
    """
    bloco = _filtrar_bloco_titular(texto_ocr, empresa)
    campos = {c: "" for c in TODOS_CAMPOS}

    m_end = re.search(
        r"(?m)^(?:ENDERE[CÇ]O(?:\s+DO\s+(?:TITULAR|IM[OÓ]VEL|CLIENTE))?|LOGRADOURO|LOCAL\s+DE\s+FORNECIMENTO)\s*[:;]\s*([^\n]+)",
        bloco,
        re.IGNORECASE,
    )
    if m_end:
        log, num, comp = _separar_logradouro_numero_complemento(m_end.group(1))
        campos["logradouro"], campos["numero"], campos["complemento"] = log, num, comp
    else:
        m_via = re.search(
            r"\b((?:RUA|R\.|AVENIDA|AV\.?|ALAMEDA|AL\.?|TRAVESSA|TV\.?|PRA[CÇ]A|P[CÇ]\.?|ESTRADA|RODOVIA)\s+[^\n]+)",
            bloco,
            re.IGNORECASE,
        )
        if m_via:
            log, num, comp = _separar_logradouro_numero_complemento(m_via.group(1))
            campos["logradouro"], campos["numero"], campos["complemento"] = log, num, comp

    if not campos["numero"]:
        m_num = re.search(r"(?:N[UÚ]MERO|N[º°])[:\s\-]*(\d{1,5}[A-Z]?)\b", bloco, re.IGNORECASE)
        if m_num:
            campos["numero"] = _limpar_valor(m_num.group(1))

    if not campos["complemento"]:
        m_comp = re.search(r"COMPLEMENTO[:\s\-]*([^\n|]+)", bloco, re.IGNORECASE)
        if m_comp:
            val = _limpar_valor(m_comp.group(1))
            if normalizar_texto(val) not in ("", "-", "N/A", "NAO INFORMADO", "SEM COMPLEMENTO"):
                campos["complemento"] = val

    # Bairro, Cidade e UF
    m_bcu = re.search(
        r"BAIRRO[:\s\-]*([^|\n/\-]+?)\s*(?:[|\-–]|CIDADE[:\s]*)\s*(?:CIDADE[:\s]*)?([A-Za-zÀ-ÿ\s\.]+?)\s*[/\-]\s*([A-Z]{2})\b",
        bloco,
        re.IGNORECASE,
    )
    if m_bcu:
        campos["bairro"] = _limpar_valor(m_bcu.group(1))
        campos["cidade"] = _limpar_valor(m_bcu.group(2))
        campos["uf"] = m_bcu.group(3).upper().strip()
    else:
        m_bairro = re.search(r"BAIRRO[:\s\-]*([^\n|]+)", bloco, re.IGNORECASE)
        if m_bairro:
            bruto_b = _limpar_valor(m_bairro.group(1))
            bruto_b = re.split(r"\b(?:CIDADE|MUNIC[IÍ]PIO|CEP|UF)\b", bruto_b, flags=re.IGNORECASE)[0]
            campos["bairro"] = _limpar_valor(bruto_b)

        m_cid_uf = re.search(
            r"(?:CIDADE|MUNIC[IÍ]PIO)[:\s\-]*([A-Za-zÀ-ÿ\s\.]+?)\s*[/\-]\s*([A-Z]{2})\b",
            bloco,
            re.IGNORECASE,
        )
        if m_cid_uf:
            campos["cidade"] = _limpar_valor(m_cid_uf.group(1))
            campos["uf"] = m_cid_uf.group(2).upper().strip()
        else:
            m_cid = re.search(r"(?:CIDADE|MUNIC[IÍ]PIO)[:\s\-]*([^\n|/]+)", bloco, re.IGNORECASE)
            if m_cid:
                bruto_c = re.split(r"\b(?:UF|ESTADO|CEP)\b", m_cid.group(1), flags=re.IGNORECASE)[0]
                campos["cidade"] = _limpar_valor(bruto_c)
            m_uf = re.search(r"(?:UF|ESTADO)[:\s\-]*([A-Z]{2})\b", bloco, re.IGNORECASE)
            if m_uf:
                campos["uf"] = m_uf.group(1).upper().strip()
            else:
                m_cuf = re.search(r"\b([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]{2,28})\s*[/\-]\s*([A-Z]{2})\b", bloco)
                if m_cuf and m_cuf.group(2).upper() in UFS_VALIDAS:
                    if not campos["cidade"]:
                        campos["cidade"] = _limpar_valor(m_cuf.group(1))
                    if not campos["uf"]:
                        campos["uf"] = m_cuf.group(2).upper()

    campos["cidade"] = re.sub(
        r"^(?:CIDADE|MUNIC[IÍ]PIO)[:\s\-]*", "", campos["cidade"], flags=re.IGNORECASE
    ).strip()

    campos["cep"] = _extrair_cep_texto(bloco)
    return campos, bloco


def _confianca_campo_por_tokens(valor: str, tokens: List[Dict[str, Any]], conf_global: float) -> float:
    if not valor:
        return 0.0
    partes = [p for p in re.findall(r"[A-Z0-9]+", normalizar_texto(valor)) if len(p) >= 1]
    if not partes or not tokens:
        return conf_global

    matched_confs: List[float] = []
    for p in partes:
        melhores = []
        for tk in tokens:
            tn = re.sub(r"[^A-Z0-9]", "", normalizar_texto(tk["text"]))
            if not tn:
                continue
            if p == tn or (len(p) >= 3 and (p in tn or tn in p)):
                melhores.append(tk["conf"])
        if melhores:
            matched_confs.append(float(np.mean(melhores)))

    if not matched_confs:
        return round(max(0.0, conf_global - 15.0), 1)
    return round(float(np.mean(matched_confs)), 1)


def validar_e_avaliar_confianca(
    campos: Dict[str, str],
    tokens: List[Dict[str, Any]],
    conf_global: float,
    empresa: str,
    limiar_confianca: float = 82.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Valida regras de formato/obrigatoriedade e decide quais campos são duvidosos.
    Retorna por campo: {valor, valido, confianca_ocr, duvidoso, motivos}
    """
    avaliacao: Dict[str, Dict[str, Any]] = {}

    for campo in TODOS_CAMPOS:
        val = (campos.get(campo) or "").strip()
        motivos: List[str] = []
        valido = True

        if campo in CAMPOS_OBRIGATORIOS and not val:
            valido = False
            motivos.append("Campo obrigatório ausente")

        if val:
            if campo == "cep":
                if not re.fullmatch(r"\d{5}-\d{3}", val):
                    valido = False
                    motivos.append("Formato de CEP inválido (esperado 00000-000)")
            elif campo == "uf":
                if val.upper() not in UFS_VALIDAS:
                    valido = False
                    motivos.append(f"UF '{val}' não reconhecida")
            elif campo == "numero":
                if not re.fullmatch(r"\d{1,5}[A-Za-z]?", val):
                    valido = False
                    motivos.append("Número fora do padrão esperado")
            elif campo == "logradouro":
                if len(val) < 5 or not re.search(r"[A-Za-zÀ-ÿ]{3,}", val):
                    valido = False
                    motivos.append("Logradouro muito curto ou suspeito")
                if not re.match(
                    r"^(?:RUA|R\.|AVENIDA|AV\.?|ALAMEDA|AL\.?|TRAVESSA|TV\.?|PRA[CÇ]A|P[CÇ]\.?|ESTRADA|RODOVIA)\b",
                    val,
                    re.IGNORECASE,
                ):
                    motivos.append("Tipo de logradouro ausente ou incomum")
            elif campo in ("bairro", "cidade"):
                if len(val) < 3 or re.search(r"\d", val):
                    valido = False
                    motivos.append("Contém dígitos ou texto muito curto")

        if campo == "complemento" and not val:
            conf_campo = round(conf_global, 1)
        else:
            conf_campo = _confianca_campo_por_tokens(val, tokens, conf_global)
            if val and conf_campo < limiar_confianca:
                motivos.append(f"Baixa confiança do OCR ({conf_campo:.1f}% < {limiar_confianca:.0f}%)")

        if empresa == "Não identificada" and campo in CAMPOS_OBRIGATORIOS:
            motivos.append("Empresa emissora não identificada com certeza")

        if val and re.search(r"[|_~@#$%*+={}\[\]<>]", val):
            motivos.append("Possível ruído de caractere no OCR")

        duvidoso = (not valido) or (len(motivos) > 0)
        avaliacao[campo] = {
            "valor": val,
            "valido": valido,
            "confianca_ocr": conf_campo,
            "duvidoso": duvidoso,
            "motivos": motivos,
        }

    return avaliacao


def processar_comprovante(
    imagem: Image.Image | np.ndarray,
    usar_preprocessamento: bool = True,
    limiar_confianca: float = 82.0,
) -> Dict[str, Any]:
    """
    Executa o pipeline completo ponta a ponta.
    Quando usar_preprocessamento=False, executa OCR direto na imagem original
    para permitir a comparação quantitativa exigida no Bloco 4.3.
    """
    if isinstance(imagem, Image.Image):
        img_rgb = np.array(imagem.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = imagem.copy()

    if usar_preprocessamento:
        img_warped, img_contorno, perspectiva_ok = detectar_e_corrigir_perspectiva(img_bgr)
        img_proc = preprocessar_imagem(img_warped)
        texto_ocr, tokens, conf_media = executar_ocr(img_proc, psm=6)
    else:
        img_warped = img_bgr
        img_contorno = img_bgr
        perspectiva_ok = False
        img_proc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        texto_ocr, tokens, conf_media = executar_ocr(img_proc, psm=6)

    empresa, conf_empresa = identificar_empresa(texto_ocr)
    campos, bloco_titular = extrair_campos_endereco(texto_ocr, empresa)
    validacao = validar_e_avaliar_confianca(
        campos, tokens, conf_media, empresa, limiar_confianca=limiar_confianca
    )

    campos_duvidosos = [c for c, info in validacao.items() if info["duvidoso"]]
    todos_obrigatorios_validos = all(validacao[c]["valido"] for c in CAMPOS_OBRIGATORIOS)

    return {
        "empresa_identificada": empresa,
        "confianca_empresa": conf_empresa,
        "perspectiva_corrigida": perspectiva_ok,
        "confianca_media_ocr": conf_media,
        "campos": campos,
        "validacao_campos": validacao,
        "campos_duvidosos": campos_duvidosos,
        "todos_obrigatorios_validos": todos_obrigatorios_validos,
        "bloco_titular_isolado": bloco_titular,
        "texto_ocr_bruto": texto_ocr,
        "imagens_intermediarias": {
            "contorno_bgr": img_contorno,
            "perspectiva_bgr": img_warped,
            "preprocessada": img_proc,
        },
    }
