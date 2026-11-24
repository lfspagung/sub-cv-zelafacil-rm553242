"""
Gerador de Dataset Sintético de Comprovantes de Residência (LGPD)
Gera 30 imagens (15 dev + 15 test) para 3 empresas reais (Enel, Sabesp, Claro),
com endereços concorrentes, marca "DOCUMENTO SINTÉTICO" e variações reais de
perspectiva, iluminação, enquadramento e ruído fotográfico.

Autor: Lucas de Freitas Pagung — RM 553242
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _carregar_fonte(tamanho: int, negrito: bool = False) -> ImageFont.FreeTypeFont:
    candidatos = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        if negrito
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for caminho in candidatos:
        if os.path.exists(caminho):
            try:
                return ImageFont.truetype(caminho, tamanho)
            except Exception:
                pass
    return ImageFont.load_default()


# ------------------------------------------------------------------
# Dados 100% sintéticos e disjuntos entre DEV (15) e TEST (15)
# ------------------------------------------------------------------
REGISTROS_DEV: List[Dict[str, Any]] = [
    # ENEL (5 dev)
    {"id": "dev_enel_01", "empresa": "Enel Distribuição", "titular": "MARIA CLARA MENDES (SINTETICO)", "logradouro": "Rua General Francisco Glicério", "numero": "1020", "complemento": "Casa 2", "bairro": "Centro", "cidade": "Suzano", "uf": "SP", "cep": "08674-000"},
    {"id": "dev_enel_02", "empresa": "Enel Distribuição", "titular": "JOAO PEDRO ALMEIDA (SINTETICO)", "logradouro": "Avenida Antônio Marques Figueira", "numero": "485", "complemento": "Apto 34", "bairro": "Vila Figueira", "cidade": "Suzano", "uf": "SP", "cep": "08676-000"},
    {"id": "dev_enel_03", "empresa": "Enel Distribuição", "titular": "ANA BEATRIZ NOGUEIRA (SINTETICO)", "logradouro": "Rua Doutor Deodato Wertheimer", "numero": "1540", "complemento": "", "bairro": "Centro", "cidade": "Mogi das Cruzes", "uf": "SP", "cep": "08710-430"},
    {"id": "dev_enel_04", "empresa": "Enel Distribuição", "titular": "CARLOS EDUARDO ROCHA (SINTETICO)", "logradouro": "Avenida Brasil", "numero": "890", "complemento": "Bloco B", "bairro": "Vila Nova", "cidade": "Poá", "uf": "SP", "cep": "08557-100"},
    {"id": "dev_enel_05", "empresa": "Enel Distribuição", "titular": "FERNANDA LIMA COSTA (SINTETICO)", "logradouro": "Rua Voluntários da Pátria", "numero": "2310", "complemento": "Apto 112", "bairro": "Santana", "cidade": "São Paulo", "uf": "SP", "cep": "02010-900"},
    # SABESP (5 dev)
    {"id": "dev_sabesp_01", "empresa": "Sabesp Saneamento", "titular": "ROBERTO SILVA TAVARES (SINTETICO)", "logradouro": "Rua Dom Pedro II", "numero": "315", "complemento": "", "bairro": "Centro", "cidade": "Santo André", "uf": "SP", "cep": "09080-110"},
    {"id": "dev_sabesp_02", "empresa": "Sabesp Saneamento", "titular": "JULIANA FERREIRA DIAS (SINTETICO)", "logradouro": "Avenida Marechal Tito", "numero": "4120", "complemento": "Fundos", "bairro": "Itaim Paulista", "cidade": "São Paulo", "uf": "SP", "cep": "08115-000"},
    {"id": "dev_sabesp_03", "empresa": "Sabesp Saneamento", "titular": "MARCOS ANTONIO PEREIRA (SINTETICO)", "logradouro": "Rua Campos Salles", "numero": "642", "complemento": "Sala 4", "bairro": "Centro", "cidade": "Guarulhos", "uf": "SP", "cep": "07011-010"},
    {"id": "dev_sabesp_04", "empresa": "Sabesp Saneamento", "titular": "PATRICIA SOUZA RAMOS (SINTETICO)", "logradouro": "Alameda Santos", "numero": "1280", "complemento": "Apto 71", "bairro": "Cerqueira César", "cidade": "São Paulo", "uf": "SP", "cep": "01418-100"},
    {"id": "dev_sabesp_05", "empresa": "Sabesp Saneamento", "titular": "RICARDO OLIVEIRA PINTO (SINTETICO)", "logradouro": "Rua Barão de Jaceguai", "numero": "750", "complemento": "", "bairro": "Centro", "cidade": "Mogi das Cruzes", "uf": "SP", "cep": "08710-160"},
    # CLARO (5 dev)
    {"id": "dev_claro_01", "empresa": "Claro Telecomunicações", "titular": "HELENA MARTINS LOPES (SINTETICO)", "logradouro": "Rua Vergueiro", "numero": "2490", "complemento": "Apto 85", "bairro": "Vila Mariana", "cidade": "São Paulo", "uf": "SP", "cep": "04102-000"},
    {"id": "dev_claro_02", "empresa": "Claro Telecomunicações", "titular": "EDUARDO CASTRO GOMES (SINTETICO)", "logradouro": "Avenida Francisco Matarazzo", "numero": "780", "complemento": "Conj 12", "bairro": "Água Branca", "cidade": "São Paulo", "uf": "SP", "cep": "05001-000"},
    {"id": "dev_claro_03", "empresa": "Claro Telecomunicações", "titular": "CAMILA BARBOSA VIEIRA (SINTETICO)", "logradouro": "Rua Quinze de Novembro", "numero": "410", "complemento": "", "bairro": "Centro", "cidade": "Santos", "uf": "SP", "cep": "11010-150"},
    {"id": "dev_claro_04", "empresa": "Claro Telecomunicações", "titular": "FELIPE ARAUJO MOURA (SINTETICO)", "logradouro": "Avenida Kennedy", "numero": "920", "complemento": "Torre 1", "bairro": "Jardim do Mar", "cidade": "São Bernardo do Campo", "uf": "SP", "cep": "09726-253"},
    {"id": "dev_claro_05", "empresa": "Claro Telecomunicações", "titular": "RENATA TEIXEIRA MELO (SINTETICO)", "logradouro": "Rua Tiradentes", "numero": "530", "complemento": "Casa 1", "bairro": "Vila Itapura", "cidade": "Campinas", "uf": "SP", "cep": "13023-190"},
]

REGISTROS_TEST: List[Dict[str, Any]] = [
    # ENEL (5 test — endereços 100% inéditos)
    {"id": "test_enel_01", "empresa": "Enel Distribuição", "titular": "LUCIA HELENA CARVALHO (SINTETICO)", "logradouro": "Rua Benjamin Constant", "numero": "845", "complemento": "Apto 21", "bairro": "Centro", "cidade": "Suzano", "uf": "SP", "cep": "08674-010"},
    {"id": "test_enel_02", "empresa": "Enel Distribuição", "titular": "PAULO CESAR RIBEIRO (SINTETICO)", "logradouro": "Avenida Armando Sales de Oliveira", "numero": "1670", "complemento": "", "bairro": "Parque Suzano", "cidade": "Suzano", "uf": "SP", "cep": "08673-000"},
    {"id": "test_enel_03", "empresa": "Enel Distribuição", "titular": "BEATRIZ FREITAS CUNHA (SINTETICO)", "logradouro": "Rua José Bonifácio", "numero": "512", "complemento": "Casa 3", "bairro": "Centro", "cidade": "Mogi das Cruzes", "uf": "SP", "cep": "08710-070"},
    {"id": "test_enel_04", "empresa": "Enel Distribuição", "titular": "GUSTAVO HENRIQUE DIAS (SINTETICO)", "logradouro": "Avenida Vital Brasil", "numero": "930", "complemento": "Apto 64", "bairro": "Butantã", "cidade": "São Paulo", "uf": "SP", "cep": "05503-001"},
    {"id": "test_enel_05", "empresa": "Enel Distribuição", "titular": "SIMONE APARECIDA LIMA (SINTETICO)", "logradouro": "Rua Coronel Cardoso de Siqueira", "numero": "1185", "complemento": "", "bairro": "Vila Oliveira", "cidade": "Mogi das Cruzes", "uf": "SP", "cep": "08780-000"},
    # SABESP (5 test)
    {"id": "test_sabesp_01", "empresa": "Sabesp Saneamento", "titular": "ANDRE LUIZ FONSECA (SINTETICO)", "logradouro": "Rua Coronel Oliveira Lima", "numero": "478", "complemento": "Apto 15", "bairro": "Centro", "cidade": "Santo André", "uf": "SP", "cep": "09010-000"},
    {"id": "test_sabesp_02", "empresa": "Sabesp Saneamento", "titular": "TATIANA MOREIRA SALES (SINTETICO)", "logradouro": "Avenida Goiás", "numero": "1950", "complemento": "", "bairro": "Barcelona", "cidade": "São Caetano do Sul", "uf": "SP", "cep": "09550-050"},
    {"id": "test_sabesp_03", "empresa": "Sabesp Saneamento", "titular": "DANIEL AUGUSTO BORGES (SINTETICO)", "logradouro": "Rua Domingos de Morais", "numero": "1420", "complemento": "Bloco A", "bairro": "Vila Mariana", "cidade": "São Paulo", "uf": "SP", "cep": "04010-200"},
    {"id": "test_sabesp_04", "empresa": "Sabesp Saneamento", "titular": "VANESSA CARDOSO NEVES (SINTETICO)", "logradouro": "Avenida Salgado Filho", "numero": "625", "complemento": "Casa 4", "bairro": "Maia", "cidade": "Guarulhos", "uf": "SP", "cep": "07115-000"},
    {"id": "test_sabesp_05", "empresa": "Sabesp Saneamento", "titular": "SERGIO ROBERTO DUARTE (SINTETICO)", "logradouro": "Rua Padre João", "numero": "290", "complemento": "", "bairro": "Penha", "cidade": "São Paulo", "uf": "SP", "cep": "03637-000"},
    # CLARO (5 test)
    {"id": "test_claro_01", "empresa": "Claro Telecomunicações", "titular": "ALINE CRISTINA FARIAS (SINTETICO)", "logradouro": "Avenida Brigadeiro Luís Antônio", "numero": "2750", "complemento": "Apto 92", "bairro": "Jardim Paulista", "cidade": "São Paulo", "uf": "SP", "cep": "01402-000"},
    {"id": "test_claro_02", "empresa": "Claro Telecomunicações", "titular": "THIAGO MENDONCA PRADO (SINTETICO)", "logradouro": "Rua Haddock Lobo", "numero": "595", "complemento": "Conj 41", "bairro": "Cerqueira César", "cidade": "São Paulo", "uf": "SP", "cep": "01414-001"},
    {"id": "test_claro_03", "empresa": "Claro Telecomunicações", "titular": "LARISSA NASCIMENTO REIS (SINTETICO)", "logradouro": "Avenida Ana Costa", "numero": "460", "complemento": "", "bairro": "Gonzaga", "cidade": "Santos", "uf": "SP", "cep": "11060-002"},
    {"id": "test_claro_04", "empresa": "Claro Telecomunicações", "titular": "RAFAEL SANTANA MORAES (SINTETICO)", "logradouro": "Rua Marechal Deodoro", "numero": "1340", "complemento": "Apto 53", "bairro": "Centro", "cidade": "São Bernardo do Campo", "uf": "SP", "cep": "09710-002"},
    {"id": "test_claro_05", "empresa": "Claro Telecomunicações", "titular": "CAROLINA PIRES GUEDES (SINTETICO)", "logradouro": "Avenida Doutor Moraes Salles", "numero": "810", "complemento": "", "bairro": "Centro", "cidade": "Campinas", "uf": "SP", "cep": "13010-001"},
]


def _desenhar_fatura_enel(reg: Dict[str, Any]) -> Image.Image:
    W, H = 1400, 1000
    img = Image.new("RGB", (W, H), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    f_title = _carregar_fonte(36, True)
    f_sub = _carregar_fonte(20, False)
    f_sec = _carregar_fonte(24, True)
    f_body = _carregar_fonte(24, False)
    f_small = _carregar_fonte(18, False)
    f_stamp = _carregar_fonte(22, True)

    # Cabeçalho Enel (com endereço concorrente 1: Sede Corporativa)
    draw.rectangle([0, 0, W, 130], fill=(22, 58, 122))
    draw.text((45, 22), "ENEL DISTRIBUICAO SAO PAULO S.A.", fill=(255, 255, 255), font=f_title)
    draw.text((45, 72), "CONTA DE ENERGIA ELETRICA  |  SEDE CORPORATIVA: Av. das Nacoes Unidas, 14401 - Barueri/SP - CEP 04794-000", fill=(225, 235, 250), font=f_small)
    draw.text((45, 98), "CNPJ: 61.695.227/0001-93  |  DOCUMENTO SINTETICO - SEM DADOS PESSOAIS REAIS", fill=(255, 220, 110), font=f_small)

    # Carimbo visível no topo direito
    draw.rectangle([1010, 145, 1360, 195], outline=(190, 30, 30), width=3)
    draw.text((1030, 158), "DOCUMENTO SINTETICO", fill=(190, 30, 30), font=f_stamp)

    # Bloco do Titular (endereço alvo)
    draw.rectangle([45, 215, 1355, 560], outline=(60, 60, 60), width=2, fill=(255, 255, 255))
    draw.rectangle([45, 215, 1355, 260], fill=(235, 240, 248))
    draw.text((65, 224), "DADOS DO TITULAR E ENDERECO DA UNIDADE CONSUMIDORA", fill=(20, 40, 80), font=f_sec)

    end_linha = f"{reg['logradouro']}, {reg['numero']}"
    if reg["complemento"]:
        end_linha += f" - {reg['complemento']}"

    y = 285
    draw.text((65, y), f"TITULAR: {reg['titular']}", fill=(15, 15, 15), font=f_body); y += 52
    draw.text((65, y), f"ENDERECO: {end_linha}", fill=(10, 10, 10), font=f_body); y += 52
    draw.text((65, y), f"BAIRRO: {reg['bairro']}", fill=(15, 15, 15), font=f_body); y += 52
    draw.text((65, y), f"CIDADE: {reg['cidade']}/{reg['uf']}", fill=(15, 15, 15), font=f_body); y += 52
    draw.text((65, y), f"CEP: {reg['cep']}", fill=(10, 10, 10), font=f_sec)

    # Resumo de consumo
    draw.rectangle([45, 585, 1355, 735], outline=(130, 130, 130), width=2)
    draw.text((65, 600), "RESUMO DA FATURA DE ENERGIA", fill=(30, 30, 30), font=f_sec)
    draw.text((65, 645), "INSTALACAO: 004928174  |  MES DE REFERENCIA: 11/2026  |  CONSUMO: 168 kWh", fill=(40, 40, 40), font=f_sub)
    draw.text((65, 685), "TOTAL A PAGAR: R$ 154,90  |  VENCIMENTO: 25/11/2026", fill=(20, 20, 20), font=f_body)

    # Endereço concorrente 2: Agência de Atendimento
    draw.rectangle([45, 765, 1355, 920], outline=(160, 160, 160), width=1, fill=(246, 246, 246))
    draw.text((65, 780), "AGENCIA DE ATENDIMENTO PRESENCIAL (NAO E O ENDERECO DO TITULAR)", fill=(80, 80, 80), font=f_sub)
    draw.text((65, 815), "Rua Voluntarios da Patria, 500 - Santana - Sao Paulo/SP - CEP 02010-000", fill=(70, 70, 70), font=f_sub)
    draw.text((65, 855), "ENDERECO DE COBRANCA: Av. Paulista, 1106 - Bela Vista - Sao Paulo/SP - CEP 01310-100", fill=(70, 70, 70), font=f_small)
    draw.text((65, 945), "DOCUMENTO SINTETICO PARA FINS ACADEMICOS (FIAP) - NENHUM DADO PESSOAL REAL", fill=(110, 110, 110), font=f_small)
    return img


def _desenhar_fatura_sabesp(reg: Dict[str, Any]) -> Image.Image:
    W, H = 1400, 1000
    img = Image.new("RGB", (W, H), (251, 253, 255))
    draw = ImageDraw.Draw(img)
    f_title = _carregar_fonte(36, True)
    f_sub = _carregar_fonte(20, False)
    f_sec = _carregar_fonte(24, True)
    f_body = _carregar_fonte(24, False)
    f_small = _carregar_fonte(18, False)
    f_stamp = _carregar_fonte(22, True)

    # Cabeçalho Sabesp (layout diferente: faixa lateral azul + cabeçalho limpo)
    draw.rectangle([0, 0, 28, H], fill=(0, 132, 198))
    draw.rectangle([28, 0, W, 120], fill=(232, 244, 252))
    draw.line([(28, 120), (W, 120)], fill=(0, 132, 198), width=4)
    draw.text((60, 22), "SABESP - SANEAMENTO BASICO DO ESTADO DE SAO PAULO", fill=(0, 88, 145), font=f_title)
    draw.text((60, 75), "CONTA DE AGUA E ESGOTO  |  MATRIZ: Rua Costa Carvalho, 300 - Pinheiros - Sao Paulo/SP - CEP 05429-900", fill=(40, 70, 95), font=f_small)

    # Bloco esquerdo: resumo / hidrômetro; Bloco direito: carimbo + agência concorrente
    draw.rectangle([60, 145, 760, 290], outline=(0, 132, 198), width=2)
    draw.text((80, 160), "RGI: 10982734-5  |  HIDROMETRO: A22B9081", fill=(15, 45, 70), font=f_sec)
    draw.text((80, 205), "CONSUMO M3: 14 m3  |  REFERENCIA: 11/2026", fill=(30, 30, 30), font=f_sub)
    draw.text((80, 245), "TOTAL A PAGAR: R$ 98,40  |  VENCIMENTO: 28/11/2026", fill=(10, 10, 10), font=f_body)

    draw.rectangle([790, 145, 1350, 290], outline=(180, 30, 30), width=2, fill=(255, 248, 248))
    draw.text((820, 165), "DOCUMENTO SINTETICO", fill=(180, 30, 30), font=f_stamp)
    draw.text((820, 205), "POSTO DE ATENDIMENTO REGIONAL:", fill=(70, 70, 70), font=f_small)
    draw.text((820, 232), "Rua do Sumidouro, 448 - Pinheiros", fill=(70, 70, 70), font=f_small)
    draw.text((820, 256), "Sao Paulo/SP - CEP 05428-010", fill=(70, 70, 70), font=f_small)

    # Bloco principal do Titular / Imóvel (mais abaixo, layout diferente da Enel)
    draw.rectangle([60, 330, 1350, 720], outline=(35, 35, 35), width=2, fill=(255, 255, 255))
    draw.rectangle([60, 330, 1350, 378], fill=(0, 132, 198))
    draw.text((80, 340), "ENDERECO DO IMOVEL / DADOS DO TITULAR", fill=(255, 255, 255), font=f_sec)

    y = 405
    draw.text((80, y), f"CLIENTE: {reg['titular']}", fill=(15, 15, 15), font=f_body); y += 54
    draw.text((80, y), f"LOGRADOURO: {reg['logradouro']}, {reg['numero']}", fill=(10, 10, 10), font=f_body); y += 54
    comp_txt = reg["complemento"] if reg["complemento"] else "SEM COMPLEMENTO"
    draw.text((80, y), f"COMPLEMENTO: {comp_txt}", fill=(20, 20, 20), font=f_body); y += 54
    draw.text((80, y), f"BAIRRO: {reg['bairro']}  |  CIDADE: {reg['cidade']} - {reg['uf']}", fill=(15, 15, 15), font=f_body); y += 54
    draw.text((80, y), f"CEP: {reg['cep']}", fill=(10, 10, 10), font=f_sec)

    # Rodapé com endereço de cobrança concorrente
    draw.rectangle([60, 760, 1350, 910], outline=(160, 160, 160), width=1, fill=(244, 247, 250))
    draw.text((80, 778), "SEDE CORPORATIVA / CNPJ 43.776.517/0001-80", fill=(60, 60, 60), font=f_sub)
    draw.text((80, 815), "ENDERECO DE COBRANCA: Rua Costa Carvalho, 300 - Pinheiros - Sao Paulo/SP - CEP 05429-900", fill=(70, 70, 70), font=f_small)
    draw.text((80, 855), "DOCUMENTO SINTETICO - USO EXCLUSIVO EM TESTES DE VISAO COMPUTACIONAL (LGPD)", fill=(120, 40, 40), font=f_small)
    return img


def _desenhar_fatura_claro(reg: Dict[str, Any]) -> Image.Image:
    W, H = 1400, 1000
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    f_title = _carregar_fonte(38, True)
    f_sub = _carregar_fonte(20, False)
    f_sec = _carregar_fonte(24, True)
    f_body = _carregar_fonte(24, False)
    f_small = _carregar_fonte(18, False)
    f_stamp = _carregar_fonte(22, True)

    # Cabeçalho Claro (vermelho institucional + layout 2 colunas no topo)
    draw.rectangle([0, 0, W, 115], fill=(204, 24, 30))
    draw.text((50, 20), "CLARO TELECOMUNICACOES S.A.", fill=(255, 255, 255), font=f_title)
    draw.text((50, 72), "FATURA MENSAL - BANDA LARGA E TELEFONIA MOVEL  |  DOCUMENTO SINTETICO", fill=(255, 235, 235), font=f_sub)

    # Faixa superior de endereço concorrente (Sede / Loja)
    draw.rectangle([45, 135, 1355, 225], outline=(180, 180, 180), width=1, fill=(248, 248, 248))
    draw.text((65, 148), "SEDE CORPORATIVA: Rua Henri Dunant, 780 - Chacara Santo Antonio - Sao Paulo/SP - CEP 04709-110", fill=(70, 70, 70), font=f_small)
    draw.text((65, 182), "LOJA DE ATENDIMENTO: Av. Paulista, 1106 - Bela Vista - Sao Paulo/SP - CEP 01310-100  |  CNPJ: 40.432.544/0001-47", fill=(70, 70, 70), font=f_small)

    # Carimbo DOCUMENTO SINTÉTICO
    draw.rectangle([45, 245, 420, 292], outline=(204, 24, 30), width=3)
    draw.text((65, 256), "DOCUMENTO SINTETICO", fill=(204, 24, 30), font=f_stamp)

    # Bloco do Assinante (endereço alvo)
    draw.rectangle([45, 315, 1355, 690], outline=(40, 40, 40), width=2, fill=(255, 255, 255))
    draw.line([(45, 368), (1355, 368)], fill=(204, 24, 30), width=3)
    draw.text((65, 328), "ENDERECO DE INSTALACAO / DADOS DO ASSINANTE", fill=(180, 20, 25), font=f_sec)

    y = 392
    draw.text((65, y), f"ASSINANTE: {reg['titular']}", fill=(15, 15, 15), font=f_body); y += 54
    end_txt = f"{reg['logradouro']}, N {reg['numero']}"
    if reg["complemento"]:
        end_txt += f" | COMPL: {reg['complemento']}"
    draw.text((65, y), f"ENDERECO DO TITULAR: {end_txt}", fill=(10, 10, 10), font=f_body); y += 54
    draw.text((65, y), f"BAIRRO: {reg['bairro']}", fill=(15, 15, 15), font=f_body); y += 54
    draw.text((65, y), f"MUNICIPIO: {reg['cidade']}  |  UF: {reg['uf']}", fill=(15, 15, 15), font=f_body); y += 54
    draw.text((65, y), f"CEP: {reg['cep']}", fill=(10, 10, 10), font=f_sec)

    # Resumo financeiro + Endereço de cobrança concorrente no rodapé
    draw.rectangle([45, 720, 1355, 860], outline=(140, 140, 140), width=2)
    draw.text((65, 738), "RESUMO DA FATURA  |  CONTRATO: 88921044  |  TOTAL A PAGAR: R$ 119,90", fill=(20, 20, 20), font=f_body)
    draw.text((65, 785), "ENDERECO DE COBRANCA: Rua Henri Dunant, 780 - Sao Paulo/SP - CEP 04709-110", fill=(80, 80, 80), font=f_small)
    draw.text((65, 818), "CENTRAL DE RELACIONAMENTO 10621  |  DOCUMENTO SINTETICO SEM DADOS PESSOAIS REAIS", fill=(100, 100, 100), font=f_small)
    return img


def _simular_foto_realista(doc_pil: Image.Image, rng: random.Random, modo_dificil: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Aplica variações fotográficas controladas:
    - Mesa/superfície escura texturizada ao redor do documento (variação de enquadramento)
    - Distorção de perspectiva + inclinação realistas
    - Gradiente de iluminação / sombra diagonal / variação de brilho
    - Leve ruído de sensor e leve desfoque óptico (quando aplicável)
    """
    doc = cv2.cvtColor(np.array(doc_pil), cv2.COLOR_RGB2BGR)
    dh, dw = doc.shape[:2]

    # Margem de mesa ao redor (variação de enquadramento)
    pad_x = rng.randint(120, 260)
    pad_y = rng.randint(110, 240)
    canvas_w = dw + 2 * pad_x
    canvas_h = dh + 2 * pad_y

    # Fundo de mesa escura texturizada (madeira/ardósia) para contraste realista com a folha
    base_cor = [rng.randint(35, 75), rng.randint(45, 85), rng.randint(55, 95)]
    canvas = np.full((canvas_h, canvas_w, 3), base_cor, dtype=np.uint8)
    ruido_mesa = rng.randint(6, 14)
    noise_bg = np.random.default_rng(rng.randint(1, 999999)).normal(0, ruido_mesa, canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise_bg, 0, 255).astype(np.uint8)

    # Pontos de origem e destino com distorção de perspectiva realista
    src_pts = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], dtype=np.float32)
    jitter_max = 105 if modo_dificil else 65
    angulo_deg = rng.uniform(-9.5, 9.5) if modo_dificil else rng.uniform(-6.5, 6.5)

    # Aplica rotação + deslocamento de perspectiva nos 4 cantos
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    rad = np.deg2rad(angulo_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    cantos_base = np.array(
        [[-dw / 2, -dh / 2], [dw / 2, -dh / 2], [dw / 2, dh / 2], [-dw / 2, dh / 2]],
        dtype=np.float32,
    )
    dst_pts = np.zeros((4, 2), dtype=np.float32)
    for i, (px, py) in enumerate(cantos_base):
        rx = px * cos_a - py * sin_a + cx + rng.uniform(-jitter_max, jitter_max)
        ry = px * sin_a + py * cos_a + cy + rng.uniform(-jitter_max * 0.75, jitter_max * 0.75)
        dst_pts[i] = [np.clip(rx, 25, canvas_w - 25), np.clip(ry, 25, canvas_h - 25)]

    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc, H_mat, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((dh, dw), 255, dtype=np.uint8), H_mat, (canvas_w, canvas_h))

    # Composita documento sobre a mesa
    mask_3 = (mask > 128)[:, :, None]
    foto = np.where(mask_3, warped_doc, canvas)

    # Variação de iluminação: gradiente de sombra direcional + exposição
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    xx /= max(canvas_w - 1, 1)
    yy /= max(canvas_h - 1, 1)

    direcao = rng.choice(["esq_dir", "dir_esq", "topo_base", "diagonal"])
    if direcao == "esq_dir":
        grad = xx
    elif direcao == "dir_esq":
        grad = 1.0 - xx
    elif direcao == "topo_base":
        grad = yy
    else:
        grad = 0.55 * xx + 0.45 * yy

    sombra_intensidade = rng.uniform(0.38, 0.55) if modo_dificil else rng.uniform(0.14, 0.34)
    fator_luz = rng.uniform(0.82, 1.08) * (1.0 - sombra_intensidade * grad)
    # Leve tom de cor (luz quente amarelada ou fria)
    tint = np.array([rng.uniform(0.94, 1.04), rng.uniform(0.96, 1.03), rng.uniform(0.95, 1.06)], dtype=np.float32)
    foto = np.clip(foto.astype(np.float32) * fator_luz[:, :, None] * tint[None, None, :], 0, 255).astype(np.uint8)

    # Ruído leve de câmera e eventual blur leve em casos difíceis
    sigma_ruido = rng.uniform(5.0, 9.5) if modo_dificil else rng.uniform(2.0, 5.5)
    sensor_noise = np.random.default_rng(rng.randint(1, 999999)).normal(0, sigma_ruido, foto.shape)
    foto = np.clip(foto.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)

    blur_aplicado = False
    if modo_dificil and rng.random() < 0.7:
        foto = cv2.GaussianBlur(foto, (3, 3), 0.9)
        blur_aplicado = True

    meta = {
        "angulo_graus": round(angulo_deg, 2),
        "sombra_intensidade": round(sombra_intensidade, 2),
        "direcao_luz": direcao,
        "blur_leve": blur_aplicado,
        "modo_dificil": modo_dificil,
        "resolucao": f"{canvas_w}x{canvas_h}",
    }
    return foto, meta


def gerar_conjunto(registros: List[Dict[str, Any]], pasta_saida: str, seed: int) -> None:
    os.makedirs(pasta_saida, exist_ok=True)
    rng = random.Random(seed)
    gabarito_lista: List[Dict[str, Any]] = []

    desenhadores = {
        "Enel Distribuição": _desenhar_fatura_enel,
        "Sabesp Saneamento": _desenhar_fatura_sabesp,
        "Claro Telecomunicações": _desenhar_fatura_claro,
    }

    for idx, reg in enumerate(registros):
        doc_pil = desenhadores[reg["empresa"]](reg)
        # 1 imagem por empresa em condição mais desafiadora (sombra forte + inclinação + blur)
        modo_dificil = (idx % 5 == 4)
        foto_bgr, meta_foto = _simular_foto_realista(doc_pil, rng, modo_dificil=modo_dificil)

        nome_img = f"{reg['id']}.jpg"
        caminho_img = os.path.join(pasta_saida, nome_img)
        cv2.imwrite(caminho_img, foto_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])

        item_gabarito = {
            "id": reg["id"],
            "arquivo": nome_img,
            "empresa": reg["empresa"],
            "endereco": {
                "logradouro": reg["logradouro"],
                "numero": reg["numero"],
                "complemento": reg["complemento"],
                "bairro": reg["bairro"],
                "cidade": reg["cidade"],
                "uf": reg["uf"],
                "cep": reg["cep"],
            },
            "condicoes_captura": meta_foto,
        }
        gabarito_lista.append(item_gabarito)

        with open(os.path.join(pasta_saida, f"{reg['id']}.json"), "w", encoding="utf-8") as fj:
            json.dump(item_gabarito, fj, indent=2, ensure_ascii=False)

    with open(os.path.join(pasta_saida, "gabarito.json"), "w", encoding="utf-8") as fg:
        json.dump(gabarito_lista, fg, indent=2, ensure_ascii=False)


def main() -> None:
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gerar_conjunto(REGISTROS_DEV, os.path.join(raiz, "dataset", "dev"), seed=202611)
    gerar_conjunto(REGISTROS_TEST, os.path.join(raiz, "dataset", "test"), seed=202699)
    print("Dataset gerado com sucesso: 15 imagens em dataset/dev e 15 imagens em dataset/test.")


if __name__ == "__main__":
    main()
