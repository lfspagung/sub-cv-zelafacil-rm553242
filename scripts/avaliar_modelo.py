"""
Avaliação Quantitativa do Pipeline no Conjunto de Teste (dataset/test)
Calcula todas as métricas exigidas na Substitutiva GS de Applied Computer Vision:
- Acerto por campo, por empresa e do endereço completo (6 campos obrigatórios)
- Comparação quantitativa COM vs. SEM pré-processamento nas mesmas 15 imagens de teste
- Avaliação da sinalização de campos duvidosos (erros sinalizados vs. erros sem aviso)
- Geração de gráficos e relatório JSON/Markdown em resultados/

Autor: Lucas de Freitas Pagung — RM 553242
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src.pipeline import (
    CAMPOS_OBRIGATORIOS,
    TODOS_CAMPOS,
    processar_comprovante,
)


def _norm_base(texto: str) -> str:
    s = unicodedata.normalize("NFKD", texto or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_logradouro(texto: str) -> str:
    s = _norm_base(texto)
    substituicoes = [
        (r"^R\s+", "RUA "),
        (r"^AV\s+", "AVENIDA "),
        (r"^AL\s+", "ALAMEDA "),
        (r"^TV\s+", "TRAVESSA "),
        (r"^PC\s+", "PRACA "),
        (r"\bDR\b", "DOUTOR"),
        (r"\bCEL\b", "CORONEL"),
        (r"\bMAL\b", "MARECHAL"),
        (r"\bGEN\b", "GENERAL"),
    ]
    for padrao, rep in substituicoes:
        s = re.sub(padrao, rep, s)
    return re.sub(r"\s+", " ", s).strip()


def comparar_campo(campo: str, extraido: str, gabarito: str) -> bool:
    """
    Regra 5 da avaliação:
    Aceita diferenças de caixa, espaços, acentuação, pontuação, abreviações usuais
    (R., Av., Pç.) e CEP com ou sem hífen.
    """
    if campo == "cep":
        d_ext = re.sub(r"\D", "", extraido or "")
        d_gab = re.sub(r"\D", "", gabarito or "")
        return d_ext == d_gab and len(d_gab) == 8
    if campo == "logradouro":
        return _norm_logradouro(extraido) == _norm_logradouro(gabarito)
    if campo == "uf":
        return _norm_base(extraido) == _norm_base(gabarito) and len(_norm_base(gabarito)) == 2
    if campo == "numero":
        return _norm_base(extraido) == _norm_base(gabarito)
    return _norm_base(extraido) == _norm_base(gabarito)


def avaliar_modo(
    itens_teste: List[Dict[str, Any]],
    pasta_teste: str,
    usar_preprocessamento: bool,
    limiar_confianca: float = 82.0,
) -> Dict[str, Any]:
    resultados_imgs: List[Dict[str, Any]] = []
    acertos_por_campo = {c: 0 for c in TODOS_CAMPOS}
    total_por_campo = {c: 0 for c in TODOS_CAMPOS}

    por_empresa: Dict[str, Dict[str, int]] = {}
    empresa_identificada_correta = 0
    endereco_completo_correto = 0

    # Métricas de sinalização de confiança (Bloco 3.3 — campos obrigatórios)
    erros_sinalizados = 0
    erros_sem_aviso = 0
    acertos_sinalizados = 0
    acertos_sem_aviso = 0
    confs_medias: List[float] = []

    for item in itens_teste:
        caminho = os.path.join(pasta_teste, item["arquivo"])
        img_bgr = cv2.imread(caminho)
        res = processar_comprovante(
            img_bgr,
            usar_preprocessamento=usar_preprocessamento,
            limiar_confianca=limiar_confianca,
        )
        confs_medias.append(res["confianca_media_ocr"])

        emp_gab = item["empresa"]
        emp_ok = _norm_base(res["empresa_identificada"]) == _norm_base(emp_gab)
        if emp_ok:
            empresa_identificada_correta += 1

        por_empresa.setdefault(
            emp_gab,
            {"total_imgs": 0, "empresa_ok": 0, "completo_ok": 0, "campos_ok": 0, "campos_total": 0},
        )
        por_empresa[emp_gab]["total_imgs"] += 1
        if emp_ok:
            por_empresa[emp_gab]["empresa_ok"] += 1

        detalhes_campos: Dict[str, Any] = {}
        todos_obrig_ok = True

        for campo in TODOS_CAMPOS:
            gab_val = item["endereco"].get(campo, "")
            ext_val = res["campos"].get(campo, "")
            ok = comparar_campo(campo, ext_val, gab_val)
            duvidoso = res["validacao_campos"][campo]["duvidoso"]

            total_por_campo[campo] += 1
            if ok:
                acertos_por_campo[campo] += 1

            if campo in CAMPOS_OBRIGATORIOS:
                por_empresa[emp_gab]["campos_total"] += 1
                if ok:
                    por_empresa[emp_gab]["campos_ok"] += 1
                else:
                    todos_obrig_ok = False

                # Contabilização da sinalização de dúvida (critério 3.3)
                if not ok and duvidoso:
                    erros_sinalizados += 1
                elif not ok and not duvidoso:
                    erros_sem_aviso += 1
                elif ok and duvidoso:
                    acertos_sinalizados += 1
                else:
                    acertos_sem_aviso += 1

            detalhes_campos[campo] = {
                "gabarito": gab_val,
                "extraido": ext_val,
                "correto": ok,
                "duvidoso": duvidoso,
                "confianca_ocr": res["validacao_campos"][campo]["confianca_ocr"],
                "motivos_duvida": res["validacao_campos"][campo]["motivos"],
            }

        if todos_obrig_ok:
            endereco_completo_correto += 1
            por_empresa[emp_gab]["completo_ok"] += 1

        resultados_imgs.append(
            {
                "id": item["id"],
                "arquivo": item["arquivo"],
                "empresa_gabarito": emp_gab,
                "empresa_identificada": res["empresa_identificada"],
                "empresa_correta": emp_ok,
                "perspectiva_corrigida": res["perspectiva_corrigida"],
                "confianca_media_ocr": res["confianca_media_ocr"],
                "endereco_completo_correto": todos_obrig_ok,
                "condicoes_captura": item.get("condicoes_captura", {}),
                "campos": detalhes_campos,
            }
        )

    n_imgs = len(itens_teste)
    pct_por_campo = {
        c: round(100.0 * acertos_por_campo[c] / max(total_por_campo[c], 1), 2)
        for c in TODOS_CAMPOS
    }
    resumo_empresas = {}
    for emp, d in por_empresa.items():
        resumo_empresas[emp] = {
            "total_imagens": d["total_imgs"],
            "identificacao_empresa_pct": round(100.0 * d["empresa_ok"] / max(d["total_imgs"], 1), 2),
            "acerto_campos_obrigatorios_pct": round(100.0 * d["campos_ok"] / max(d["campos_total"], 1), 2),
            "endereco_completo_correto": d["completo_ok"],
            "endereco_completo_pct": round(100.0 * d["completo_ok"] / max(d["total_imgs"], 1), 2),
        }

    total_erros = erros_sinalizados + erros_sem_aviso
    taxa_deteccao_erros = round(100.0 * erros_sinalizados / total_erros, 2) if total_erros > 0 else 100.0
    total_campos_obrig = n_imgs * len(CAMPOS_OBRIGATORIOS)
    acertos_obrig = acertos_sinalizados + acertos_sem_aviso

    return {
        "usar_preprocessamento": usar_preprocessamento,
        "total_imagens_teste": n_imgs,
        "confianca_media_ocr_geral": round(float(np.mean(confs_medias)), 2) if confs_medias else 0.0,
        "identificacao_empresa_correta": empresa_identificada_correta,
        "identificacao_empresa_pct": round(100.0 * empresa_identificada_correta / max(n_imgs, 1), 2),
        "endereco_completo_correto": endereco_completo_correto,
        "endereco_completo_pct": round(100.0 * endereco_completo_correto / max(n_imgs, 1), 2),
        "acerto_medio_campos_obrigatorios_pct": round(100.0 * acertos_obrig / max(total_campos_obrig, 1), 2),
        "acerto_por_campo_pct": pct_por_campo,
        "acerto_por_empresa": resumo_empresas,
        "avaliacao_confianca": {
            "total_campos_obrigatorios_avaliados": total_campos_obrig,
            "total_erros": total_erros,
            "erros_sinalizados": erros_sinalizados,
            "erros_sem_aviso": erros_sem_aviso,
            "taxa_deteccao_erros_pct": taxa_deteccao_erros,
            "acertos_sinalizados_falso_alarme": acertos_sinalizados,
            "acertos_sem_aviso": acertos_sem_aviso,
        },
        "detalhes_por_imagem": resultados_imgs,
    }


def gerar_graficos(res_com: Dict[str, Any], res_sem: Dict[str, Any], pasta_resultados: str) -> None:
    campos = CAMPOS_OBRIGATORIOS + ["completo"]
    labels = ["Logradouro", "Número", "Bairro", "Cidade", "UF", "CEP", "Endereço\nCompleto"]
    vals_com = [res_com["acerto_por_campo_pct"][c] for c in CAMPOS_OBRIGATORIOS] + [res_com["endereco_completo_pct"]]
    vals_sem = [res_sem["acerto_por_campo_pct"][c] for c in CAMPOS_OBRIGATORIOS] + [res_sem["endereco_completo_pct"]]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    b1 = ax.bar(x - w / 2, vals_sem, w, label="Sem pré-processamento (OCR direto)", color="#94A3B8")
    b2 = ax.bar(x + w / 2, vals_com, w, label="Com pré-processamento OpenCV + Perspectiva", color="#0F766E")

    ax.set_ylabel("Acerto no Conjunto de Teste (%)")
    ax.set_title("Comparação Quantitativa: Com vs. Sem Pré-processamento OpenCV (n = 15 imagens de teste)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(pasta_resultados, "grafico_comparacao_preprocessamento.png"))
    plt.close(fig)


def gerar_imagem_etapas(pasta_teste: str, pasta_resultados: str) -> None:
    amostra = os.path.join(pasta_teste, "test_enel_01.jpg")
    if not os.path.exists(amostra):
        return
    img_bgr = cv2.imread(amostra)
    res = processar_comprovante(img_bgr, usar_preprocessamento=True)
    inter = res["imagens_intermediarias"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=150)
    etapas = [
        ("1. Foto Original (Inclinada/Sombra)", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
        ("2. Contorno Detectado (OpenCV)", cv2.cvtColor(inter["contorno_bgr"], cv2.COLOR_BGR2RGB)),
        ("3. Perspectiva Corrigida", cv2.cvtColor(inter["perspectiva_bgr"], cv2.COLOR_BGR2RGB)),
        ("4. Pré-processada p/ OCR", inter["preprocessada"]),
    ]
    for ax, (titulo, im) in zip(axes, etapas):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(pasta_resultados, "pipeline_etapas_exemplo.png"))
    plt.close(fig)


def main() -> None:
    pasta_teste = os.path.join(RAIZ, "dataset", "test")
    pasta_resultados = os.path.join(RAIZ, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)

    with open(os.path.join(pasta_teste, "gabarito.json"), encoding="utf-8") as f:
        itens_teste = json.load(f)

    print("Avaliando conjunto de teste COM pré-processamento OpenCV...")
    res_com = avaliar_modo(itens_teste, pasta_teste, usar_preprocessamento=True)

    print("Avaliando conjunto de teste SEM pré-processamento (baseline OCR direto)...")
    res_sem = avaliar_modo(itens_teste, pasta_teste, usar_preprocessamento=False)

    saida = {
        "autor": "Lucas de Freitas Pagung",
        "rm": "553242",
        "conjunto_avaliado": "dataset/test (15 imagens inéditas: 5 Enel, 5 Sabesp, 5 Claro)",
        "com_preprocessamento": res_com,
        "sem_preprocessamento": res_sem,
    }

    caminho_json = os.path.join(pasta_resultados, "avaliacao_teste.json")
    with open(caminho_json, "w", encoding="utf-8") as fj:
        json.dump(saida, fj, indent=2, ensure_ascii=False)

    gerar_graficos(res_com, res_sem, pasta_resultados)
    gerar_imagem_etapas(pasta_teste, pasta_resultados)

    # Imprime resumo claro no terminal e salva em Markdown
    conf = res_com["avaliacao_confianca"]
    linhas_md = [
        "# Resumo da Avaliação — Conjunto de Teste (15 imagens)",
        "",
        f"- **Identificação de empresa (Com Pré-proc):** {res_com['identificacao_empresa_correta']}/15 ({res_com['identificacao_empresa_pct']}%)",
        f"- **Endereço completo correto (Com Pré-proc):** {res_com['endereco_completo_correto']}/15 ({res_com['endereco_completo_pct']}%)",
        f"- **Endereço completo correto (Sem Pré-proc):** {res_sem['endereco_completo_correto']}/15 ({res_sem['endereco_completo_pct']}%)",
        f"- **Acerto médio nos 6 campos obrigatórios:** {res_com['acerto_medio_campos_obrigatorios_pct']}% (Com) vs. {res_sem['acerto_medio_campos_obrigatorios_pct']}% (Sem)",
        "",
        "## Acerto por Campo (Com vs. Sem Pré-processamento)",
        "| Campo | Com Pré-processamento | Sem Pré-processamento |",
        "| :--- | :---: | :---: |",
    ]
    for c in TODOS_CAMPOS:
        linhas_md.append(f"| {c} | {res_com['acerto_por_campo_pct'][c]}% | {res_sem['acerto_por_campo_pct'][c]}% |")

    linhas_md += [
        "",
        "## Acerto por Empresa (Com Pré-processamento)",
        "| Empresa | Identificação | Campos Obrigatórios | Endereço Completo |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for emp, d in res_com["acerto_por_empresa"].items():
        linhas_md.append(
            f"| {emp} | {d['identificacao_empresa_pct']}% | {d['acerto_campos_obrigatorios_pct']}% | {d['endereco_completo_correto']}/{d['total_imagens']} ({d['endereco_completo_pct']}%) |"
        )

    linhas_md += [
        "",
        "## Avaliação da Sinalização de Campos Duvidosos (Critério 3.3)",
        f"- **Total de campos obrigatórios avaliados:** {conf['total_campos_obrigatorios_avaliados']}",
        f"- **Total de erros de extração:** {conf['total_erros']}",
        f"- **Erros sinalizados como duvidosos (com aviso):** {conf['erros_sinalizados']} ({conf['taxa_deteccao_erros_pct']}%)",
        f"- **Erros que passaram sem aviso:** {conf['erros_sem_aviso']}",
        f"- **Acertos sinalizados (falsos alarmes):** {conf['acertos_sinalizados_falso_alarme']}",
        f"- **Acertos sem aviso:** {conf['acertos_sem_aviso']}",
    ]

    md_texto = "\n".join(linhas_md)
    with open(os.path.join(pasta_resultados, "resumo_metricas.md"), "w", encoding="utf-8") as fm:
        fm.write(md_texto)

    print("\n" + md_texto)
    print(f"\nArquivos salvos em: {pasta_resultados}/")


if __name__ == "__main__":
    main()
