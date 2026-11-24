"""
Aplicação Web de Demonstração — Extração Automática de Endereços (Applied Computer Vision)
Fluxo completo:
1) Upload da foto do comprovante (ou seleção de amostra sintética)
2) Identificação automática da empresa + correção de perspectiva + pré-processamento OpenCV + OCR
3) Formulário preenchido automaticamente com destaque visual dos campos duvidosos e seus motivos
4) Confirmação ou correção pelo cidadão + exibição do endereço final confirmado em JSON
   indicando exatamente quais campos foram corrigidos.

Autor: Lucas de Freitas Pagung — RM 553242
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from src.pipeline import (
    CAMPOS_OBRIGATORIOS,
    TODOS_CAMPOS,
    processar_comprovante,
)

st.set_page_config(
    page_title="ZelaFácil CV — Extração Automática de Endereços",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
        .doubt-badge {
            display: inline-block;
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #F59E0B;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-left: 6px;
        }
        .ok-badge {
            display: inline-block;
            background-color: #DCFCE7;
            color: #166534;
            border: 1px solid #22C55E;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-left: 6px;
        }
        .card-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Extração Automática de Endereços — Applied Computer Vision")
st.caption("Substitutiva GS 2026.2 · FIAP Engenharia de Software · Lucas de Freitas Pagung (RM 553242)")

aba_demo, aba_metricas = st.tabs(["1. Demonstração do Pipeline", "2. Métricas e Avaliação de Confiança (Teste)"])

with aba_demo:
    st.markdown(
        """
        Envie a foto de um comprovante de residência sintético (**Enel**, **Sabesp** ou **Claro**).
        O sistema detecta o documento, corrige a perspectiva com **OpenCV**, identifica a empresa automaticamente,
        extrai apenas o endereço do titular (descartando sede/agência/cobrança) e destaca os **campos duvidosos**
        para revisão humana.
        """
    )

    col_up, col_sample = st.columns([2, 1])
    with col_up:
        arquivo_enviado = st.file_uploader(
            "Envie a foto do comprovante (JPG ou PNG)",
            type=["jpg", "jpeg", "png"],
        )

    amostra_escolhida = None
    with col_sample:
        pasta_dev = os.path.join("dataset", "dev")
        opcoes_amostras = ["Nenhuma (usar upload)"]
        if os.path.isdir(pasta_dev):
            imgs_dev = sorted([f for f in os.listdir(pasta_dev) if f.lower().endswith((".jpg", ".png"))])
            opcoes_amostras += imgs_dev[:6]
        escolha = st.selectbox("Ou teste com uma amostra sintética (dev):", opcoes_amostras)
        if escolha != "Nenhuma (usar upload)":
            amostra_escolhida = os.path.join(pasta_dev, escolha)

    usar_preproc = st.toggle("Ativar correção de perspectiva e pré-processamento OpenCV", value=True)

    imagem_pil = None
    origem_chave = None
    if arquivo_enviado is not None:
        imagem_pil = Image.open(arquivo_enviado)
        origem_chave = f"upload_{arquivo_enviado.name}_{arquivo_enviado.size}_{usar_preproc}"
    elif amostra_escolhida is not None and os.path.exists(amostra_escolhida):
        imagem_pil = Image.open(amostra_escolhida)
        origem_chave = f"sample_{amostra_escolhida}_{usar_preproc}"

    if imagem_pil is not None:
        if st.session_state.get("ultima_chave") != origem_chave:
            with st.spinner("Executando pipeline OpenCV + Tesseract OCR..."):
                st.session_state["resultado_atual"] = processar_comprovante(
                    imagem_pil,
                    usar_preprocessamento=usar_preproc,
                )
                st.session_state["ultima_chave"] = origem_chave
                st.session_state["confirmacao_json"] = None

        res = st.session_state["resultado_atual"]
        inter = res["imagens_intermediarias"]

        # Cabeçalho de resultado automático
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Empresa identificada", res["empresa_identificada"])
        m2.metric("Confiança média OCR", f"{res['confianca_media_ocr']:.1f}%")
        m3.metric("Perspectiva corrigida", "Sim" if res["perspectiva_corrigida"] else "Não")
        m4.metric("Campos duvidosos", str(len(res["campos_duvidosos"])))

        # Visualização das etapas OpenCV
        with st.expander("Visualizar etapas de Visão Computacional (OpenCV)", expanded=True):
            c_img1, c_img2, c_img3 = st.columns(3)
            c_img1.image(
                cv2.cvtColor(inter["contorno_bgr"], cv2.COLOR_BGR2RGB),
                caption="1. Detecção do documento",
                width="stretch",
            )
            c_img2.image(
                cv2.cvtColor(inter["perspectiva_bgr"], cv2.COLOR_BGR2RGB),
                caption="2. Correção de perspectiva",
                width="stretch",
            )
            c_img3.image(
                inter["preprocessada"],
                caption="3. Pré-processamento p/ OCR",
                width="stretch",
            )

        st.divider()
        st.subheader("Conferência e Correção do Endereço Extraído")

        if res["campos_duvidosos"]:
            st.warning(
                f"Atenção: **{len(res['campos_duvidosos'])} campo(s)** foram sinalizados como duvidosos "
                f"({', '.join(res['campos_duvidosos'])}). Revise os itens destacados abaixo antes de confirmar."
            )
        else:
            st.success("Todos os campos obrigatórios foram extraídos com alta confiança e aprovados nas regras de validação.")

        rotulos = {
            "logradouro": "Logradouro *",
            "numero": "Número *",
            "complemento": "Complemento (opcional)",
            "bairro": "Bairro *",
            "cidade": "Cidade *",
            "uf": "UF *",
            "cep": "CEP *",
        }

        def _render_label(campo_nome: str) -> None:
            info = res["validacao_campos"][campo_nome]
            if info["duvidoso"]:
                motivos_txt = " · ".join(info["motivos"])
                st.markdown(
                    f"<strong>{rotulos[campo_nome]}</strong> "
                    f"<span class='doubt-badge'>⚠ Duvidoso ({info['confianca_ocr']:.0f}%) — {motivos_txt}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<strong>{rotulos[campo_nome]}</strong> "
                    f"<span class='ok-badge'>✓ Confiável ({info['confianca_ocr']:.0f}%)</span>",
                    unsafe_allow_html=True,
                )

        with st.form("form_confirmacao_endereco"):
            r1c1, r1c2, r1c3 = st.columns([2.4, 0.8, 1.2])
            with r1c1:
                _render_label("logradouro")
                v_log = st.text_input("logradouro", value=res["campos"]["logradouro"], label_visibility="collapsed")
            with r1c2:
                _render_label("numero")
                v_num = st.text_input("numero", value=res["campos"]["numero"], label_visibility="collapsed")
            with r1c3:
                _render_label("complemento")
                v_comp = st.text_input("complemento", value=res["campos"]["complemento"], label_visibility="collapsed")

            r2c1, r2c2, r2c3, r2c4 = st.columns([1.6, 1.6, 0.6, 1.0])
            with r2c1:
                _render_label("bairro")
                v_bairro = st.text_input("bairro", value=res["campos"]["bairro"], label_visibility="collapsed")
            with r2c2:
                _render_label("cidade")
                v_cidade = st.text_input("cidade", value=res["campos"]["cidade"], label_visibility="collapsed")
            with r2c3:
                _render_label("uf")
                v_uf = st.text_input("uf", value=res["campos"]["uf"], max_chars=2, label_visibility="collapsed")
            with r2c4:
                _render_label("cep")
                v_cep = st.text_input("cep", value=res["campos"]["cep"], label_visibility="collapsed")

            confirmado = st.form_submit_button("Confirmar Endereço Final", type="primary")

        if confirmado:
            confirmados = {
                "logradouro": v_log.strip(),
                "numero": v_num.strip(),
                "complemento": v_comp.strip(),
                "bairro": v_bairro.strip(),
                "cidade": v_cidade.strip(),
                "uf": v_uf.strip().upper(),
                "cep": v_cep.strip(),
            }
            originais = res["campos"]
            campos_corrigidos = [
                c for c in TODOS_CAMPOS
                if (confirmados[c] or "").strip() != (originais.get(c) or "").strip()
            ]
            diferencas = {
                c: {"extraido_ocr": originais.get(c, ""), "confirmado_usuario": confirmados[c]}
                for c in campos_corrigidos
            }

            st.session_state["confirmacao_json"] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "empresa_identificada": res["empresa_identificada"],
                "endereco_confirmado": confirmados,
                "campos_sinalizados_como_duvidosos": res["campos_duvidosos"],
                "houve_correcao_manual": len(campos_corrigidos) > 0,
                "campos_corrigidos": campos_corrigidos,
                "detalhe_correcoes": diferencas,
            }

        if st.session_state.get("confirmacao_json"):
            st.subheader("Endereço Final Confirmado (JSON)")
            payload = st.session_state["confirmacao_json"]
            if payload["campos_corrigidos"]:
                st.info(f"Campos corrigidos pelo usuário: **{', '.join(payload['campos_corrigidos'])}**")
            else:
                st.success("Endereço confirmado sem necessidade de correções manuais.")
            st.json(payload)

        with st.expander("Inspecionar texto bruto do OCR e bloco isolado do titular"):
            t1, t2 = st.columns(2)
            t1.markdown("**Bloco isolado do titular (após descarte de endereços concorrentes):**")
            t1.code(res["bloco_titular_isolado"] or "(vazio)")
            t2.markdown("**Texto bruto completo extraído pelo Tesseract OCR:**")
            t2.code(res["texto_ocr_bruto"] or "(vazio)")

with aba_metricas:
    caminho_av = os.path.join("resultados", "avaliacao_teste.json")
    if os.path.exists(caminho_av):
        with open(caminho_av, encoding="utf-8") as f:
            av = json.load(f)
        rc = av["com_preprocessamento"]
        rs = av["sem_preprocessamento"]
        conf = rc["avaliacao_confianca"]

        st.subheader("Resultados no Conjunto de Teste (15 imagens inéditas)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Endereço completo (Com Pré-proc)", f"{rc['endereco_completo_pct']}%", f"{rc['endereco_completo_correto']}/15")
        k2.metric("Endereço completo (Sem Pré-proc)", f"{rs['endereco_completo_pct']}%", f"{rs['endereco_completo_correto']}/15")
        k3.metric("Identificação da Empresa", f"{rc['identificacao_empresa_pct']}%", f"{rc['identificacao_empresa_correta']}/15")
        k4.metric("Erros sinalizados (Confiança)", f"{conf['erros_sinalizados']}/{conf['total_erros']}", f"{conf['erros_sem_aviso']} sem aviso")

        df_campos = pd.DataFrame(
            [
                {
                    "Campo": c,
                    "Com Pré-processamento (%)": rc["acerto_por_campo_pct"][c],
                    "Sem Pré-processamento (%)": rs["acerto_por_campo_pct"][c],
                }
                for c in TODOS_CAMPOS
            ]
        )
        st.markdown("### Acerto por Campo")
        st.dataframe(df_campos, width="stretch", hide_index=True)

        df_emp = pd.DataFrame(
            [
                {
                    "Empresa": emp,
                    "Identificação (%)": d["identificacao_empresa_pct"],
                    "Campos Obrigatórios (%)": d["acerto_campos_obrigatorios_pct"],
                    "Endereço Completo (%)": d["endereco_completo_pct"],
                }
                for emp, d in rc["acerto_por_empresa"].items()
            ]
        )
        st.markdown("### Acerto por Empresa (Com Pré-processamento)")
        st.dataframe(df_emp, width="stretch", hide_index=True)

        graf = os.path.join("resultados", "grafico_comparacao_preprocessamento.png")
        if os.path.exists(graf):
            st.image(graf, caption="Comparação quantitativa com e sem pré-processamento OpenCV", width="stretch")
    else:
        st.info("Execute `python scripts/avaliar_modelo.py` localmente para gerar o arquivo `resultados/avaliacao_teste.json` e exibir as métricas aqui.")
