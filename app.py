import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError


# -----------------------------
# Config
# -----------------------------
REPORTS_DIR = "reports"
CALIB_DIR = "calibration"
CALIB_FILE = os.path.join(CALIB_DIR, "labels.jsonl")

CLICKABLE_SELECTOR = "button, [role='button'], a, input[type='button'], input[type='submit']"

OECD_CATEGORIES = [
    "forced_action",
    "interface_interference",
    "nagging",
    "obstruction",
    "sneaking",
    "social_proof",
    "urgency",
]

CATEGORY_LABEL = {
    "forced_action": "Ação forçada",
    "interface_interference": "Interferência de interface",
    "nagging": "Nagging",
    "obstruction": "Obstrução",
    "sneaking": "Sneaking",
    "social_proof": "Prova social",
    "urgency": "Urgência",
}

# PT/EN lexicon (MVP) — você vai ampliar isso com o tempo
RX = {
    "accept": re.compile(r"\b(aceitar|concordo|permitir|ok|continuar|prosseguir|confirmar|i\s+agree|accept|allow|continue|confirm)\b", re.I),
    "reject": re.compile(r"\b(recusar|rejeitar|não\s+aceito|nao\s+aceito|não\s+permitir|nao\s+permitir|fechar|voltar|decline|reject|no\s+thanks|close|back)\b", re.I),
    "subscribe": re.compile(r"\b(assinar|inscrever|ativar\s+plano|começar\s+teste|comecar\s+teste|comprar|finalizar|pagar|subscribe|start\s+trial|buy|checkout|pay|place\s+order)\b", re.I),
    "cancel": re.compile(r"\b(cancelar|encerrar|desativar|excluir\s+conta|deletar\s+conta|opt\s*out|unsubscribe|cancel\s+subscription|delete\s+account)\b", re.I),

    "urgency": re.compile(r"\b(só\s+hoje|so\s+hoje|termina\s+em|oferta\s+expira|últim[ao]s?\s+(minutos?|horas?)|ultim[ao]s?\s+(minutos?|horas?)|agora|já|ja|limited\s+time|ends?\s+in|deal\s+expires?|hurry|last\s+chance|restam\s+\d+)\b", re.I),
    "timer": re.compile(r"\b(\d{1,2}:\d{2}(:\d{2})?)\b"),

    "social_proof": re.compile(r"\b(\d+\s+(pessoas|usuários|usuarios|clientes)\s+(viram|compraram|estão\s+vendo|estao\s+vendo)|popular|em\s+alta|best\s+seller|trending|x\s+people\s+are\s+viewing|just\s+purchased)\b", re.I),
    "nagging": re.compile(r"\b(ativar\s+notifica|permitir\s+notifica|aceitar\s+push|permitir\s+localiza|allow\s+notifications?|enable\s+notifications?|turn\s+on\s+notifications?|allow\s+location)\b", re.I),

    "forced": re.compile(r"\b(crie\s+uma\s+conta|faça\s+login\s+para|faca\s+login\s+para|cadastre-?se\s+para|aceite\s+cookies\s+para\s+(continuar|acessar)|cookie\s+wall|sign\s+in\s+to\s+continue|create\s+an\s+account\s+to)\b", re.I),

    "price": re.compile(r"\b(R\$\s?\d+([.,]\d{2})?|\$\s?\d+([.,]\d{2})?|€\s?\d+([.,]\d{2})?)\b"),
}

# “Explore” — botões que costumam revelar preferências/configurações
RX_EXPLORE = re.compile(r"\b(manage|settings|preferences|preferências|preferencias|opções|opcoes|configura|saiba\s+mais|more|detalhes|details)\b", re.I)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TrailStep:
    action: str
    selector: Optional[str] = None
    text: Optional[str] = None
    screenshot: Optional[str] = None
    note: Optional[str] = None


@dataclass
class Evidence:
    category: str
    score: float
    reason: str
    selector: Optional[str] = None
    textSnippet: Optional[str] = None
    url: Optional[str] = None
    screenshotPath: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def bump(scores: Dict[str, float], cat: str, v: float) -> None:
    scores[cat] = max(scores.get(cat, 0.0), clamp01(v))

def slugify(url: str) -> str:
    # curto e determinístico
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return h

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def get_body_text(page, limit: int = 200000) -> str:
    try:
        t = page.locator("body").inner_text(timeout=2500)
        return normalize_ws(t)[:limit]
    except Exception:
        return ""

def take_screenshot(page, out_dir: str, name: str) -> str:
    ensure_dir(out_dir)
    p = os.path.join(out_dir, f"{name}.png")
    page.screenshot(path=p, full_page=True)
    return p

def safe_wait_dom(page, ms: int = 800) -> None:
    page.wait_for_timeout(ms)
    try:
        page.wait_for_load_state("domcontentloaded", timeout=2000)
    except PWTimeoutError:
        pass

def find_best_clickable_by_regex(page, rx: re.Pattern, max_scan: int = 250) -> Optional[Tuple[str, str]]:
    loc = page.locator(CLICKABLE_SELECTOR)
    count = min(loc.count(), max_scan)

    best_i = None
    best_score = -1.0
    best_text = ""

    for i in range(count):
        el = loc.nth(i)
        try:
            if not el.is_visible():
                continue

            text = ""
            try:
                text = normalize_ws(el.inner_text(timeout=350))
            except Exception:
                text = ""

            if not text:
                text = (el.get_attribute("value") or el.get_attribute("aria-label") or "").strip()
                text = normalize_ws(text)

            if not text or not rx.search(text):
                continue

            box = el.bounding_box()
            area = 0.2
            if box:
                area = min(1.0, (box["width"] * box["height"]) / 6000.0)

            score = 0.6 + 0.4 * area
            if score > best_score:
                best_score = score
                best_i = i
                best_text = text[:200]
        except Exception:
            continue

    if best_i is None:
        return None

    selector = f"{CLICKABLE_SELECTOR} >> nth={best_i}"
    return selector, best_text

def click_and_wait(page, selector: str) -> bool:
    try:
        page.locator(selector).click(timeout=1200)
        safe_wait_dom(page, 700)
        return True
    except Exception:
        return False


# -----------------------------
# Calibration (simple)
# -----------------------------
def load_calibration_stats() -> Dict[str, Dict[str, int]]:
    """
    Returns counts per category: {"tp": n, "fp": n}
    """
    ensure_dir(CALIB_DIR)
    stats = {c: {"tp": 0, "fp": 0} for c in OECD_CATEGORIES}
    if not os.path.exists(CALIB_FILE):
        return stats

    with open(CALIB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cat = obj.get("category")
                label = obj.get("label")  # "tp" or "fp"
                if cat in stats and label in ("tp", "fp"):
                    stats[cat][label] += 1
            except Exception:
                continue
    return stats

def calibration_multiplier(stats: Dict[str, Dict[str, int]], cat: str) -> float:
    """
    Very simple: multiplier in [0.7, 1.2] based on tp/(tp+fp)
    """
    tp = stats.get(cat, {}).get("tp", 0)
    fp = stats.get(cat, {}).get("fp", 0)
    total = tp + fp
    if total < 5:
        return 1.0  # pouco dado, não mexe
    precision = tp / total if total else 1.0
    # map precision [0..1] to [0.7..1.2] centered around 0.7->0.7, 0.5->0.95, 1.0->1.2
    return clamp01(0.7 + 0.5 * precision)  # clamp01 gives 0..1; we want 0.7..1.2
    # Oops: clamp01 squashes. We'll custom clamp instead.

def calibration_multiplier_fixed(stats: Dict[str, Dict[str, int]], cat: str) -> float:
    tp = stats.get(cat, {}).get("tp", 0)
    fp = stats.get(cat, {}).get("fp", 0)
    total = tp + fp
    if total < 5:
        return 1.0
    precision = tp / total if total else 1.0
    mult = 0.7 + 0.5 * precision  # 0.7..1.2
    return max(0.7, min(1.2, mult))

def save_label(evidence_id: str, url: str, category: str, label: str, note: str = "") -> None:
    ensure_dir(CALIB_DIR)
    rec = {
        "ts": now_iso(),
        "evidence_id": evidence_id,
        "url": url,
        "category": category,
        "label": label,  # "tp" | "fp"
        "note": note
    }
    with open(CALIB_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------
# Static signals -> evidences
# -----------------------------
def scan_static_signals(text: str) -> Dict[str, bool]:
    return {
        "hasUrgency": bool(RX["urgency"].search(text) or RX["timer"].search(text)),
        "hasSocialProof": bool(RX["social_proof"].search(text)),
        "hasNagging": bool(RX["nagging"].search(text)),
        "hasForced": bool(RX["forced"].search(text)),
    }


# -----------------------------
# Friction layer: attempt action with trail
# -----------------------------
def attempt_action_path(page, target_rx: re.Pattern, explore_rx: Optional[re.Pattern], max_depth: int, out_dir: str, name: str) -> Tuple[Optional[int], List[TrailStep]]:
    """
    Returns (clicks_to_success, trail).
    Success here = target option becomes available OR a click on target is executed.
    Steps are recorded with screenshots.
    """
    trail: List[TrailStep] = []

    # If target already visible
    target = find_best_clickable_by_regex(page, target_rx)
    if target:
        sel, txt = target
        # Try clicking once to prove it is actionable, but treat as 0 friction to reach option
        shot = take_screenshot(page, out_dir, f"{name}_depth0")
        trail.append(TrailStep(action="found_target", selector=sel, text=txt, screenshot=shot))
        return 0, trail

    for depth in range(1, max_depth + 1):
        if explore_rx is None:
            break
        explore = find_best_clickable_by_regex(page, explore_rx)
        if not explore:
            shot = take_screenshot(page, out_dir, f"{name}_depth{depth}_no_explore")
            trail.append(TrailStep(action="no_explore_found", screenshot=shot, note="Não achei botão/link para revelar opções."))
            break

        sel, txt = explore
        ok = click_and_wait(page, sel)
        shot = take_screenshot(page, out_dir, f"{name}_depth{depth}")
        trail.append(TrailStep(action="click_explore", selector=sel, text=txt, screenshot=shot, note=f"click ok={ok}"))

        target = find_best_clickable_by_regex(page, target_rx)
        if target:
            tsel, ttxt = target
            shot2 = take_screenshot(page, out_dir, f"{name}_depth{depth}_found_target")
            trail.append(TrailStep(action="found_target", selector=tsel, text=ttxt, screenshot=shot2))
            return depth, trail

    return None, trail


# -----------------------------
# Report export
# -----------------------------
def render_report_html(data: Dict[str, Any]) -> str:
    url = data["url"]
    started = data["startedAt"]
    finished = data["finishedAt"]
    scores = data["summary"]["categoryScores"]
    friction = data["summary"]["friction"]

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = ""
    for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        rows += f"<tr><td>{esc(CATEGORY_LABEL.get(cat, cat))}</td><td>{score:.2f}</td></tr>"

    fr = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in friction.items())

    ev_html = ""
    for e in sorted(data.get("evidences", []), key=lambda x: x.get("score", 0), reverse=True):
        ev_html += f"""
        <div style="border:1px solid #ddd;padding:12px;border-radius:10px;margin:10px 0;">
          <div style="font-weight:700">{esc(CATEGORY_LABEL.get(e.get("category"), e.get("category")))} — score {float(e.get("score",0)):.2f}</div>
          <div>{esc(e.get("reason",""))}</div>
          <div style="margin-top:6px;color:#555;font-size:12px;">Selector: {esc(e.get("selector","") or "")}</div>
          <div style="margin-top:6px;"><pre style="white-space:pre-wrap;">{esc(e.get("textSnippet","") or "")}</pre></div>
          {"<div style='margin-top:6px;'><img src='"+esc(e.get("screenshotPath"))+"' style='max-width:100%;border-radius:8px;'/></div>" if e.get("screenshotPath") else ""}
        </div>
        """

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Fair Design Report</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; margin: 28px; }}
        h1 {{ margin: 0 0 8px 0; }}
        .meta {{ color:#444; margin-bottom: 18px; }}
        table {{ border-collapse: collapse; width: 520px; }}
        td, th {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ text-align:left; background:#f5f5f5; }}
      </style>
    </head>
    <body>
      <h1>Fair Design — Relatório de Dark Patterns </h1>
      <div class="meta">
        <div><b>URL:</b> {esc(url)}</div>
        <div><b>Início:</b> {esc(started)} &nbsp;&nbsp; <b>Fim:</b> {esc(finished)}</div>
      </div>

      <h2>Resumo</h2>
      <table>
        <tr><th>Categoria</th><th>Score</th></tr>
        {rows}
      </table>

      <h3>Fricção (proxies + trilhas)</h3>
      <ul>{fr}</ul>

      <h2>Evidências</h2>
      {ev_html}
    </body>
    </html>
    """
    return html

def save_case_file(case_dir: str, data: Dict[str, Any], html: str) -> Tuple[str, str]:
    ensure_dir(case_dir)
    json_path = os.path.join(case_dir, "result.json")
    html_path = os.path.join(case_dir, "report.html")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return json_path, html_path

def export_pdf_via_playwright(html_path: str, pdf_path: str) -> None:
    # Use playwright to render HTML to PDF
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{os.path.abspath(html_path)}", wait_until="load")
        page.pdf(path=pdf_path, format="A4", print_background=True, margin={"top":"14mm","bottom":"14mm","left":"12mm","right":"12mm"})
        browser.close()


# -----------------------------
# Main crawl
# -----------------------------
def crawl(url: str, max_depth: int, headed: bool, use_calibration: bool) -> Dict[str, Any]:
    started_at = now_iso()

    case_id = f"{now_stamp()}_{slugify(url)}"
    case_dir = os.path.join(REPORTS_DIR, case_id)
    shots_dir = os.path.join(case_dir, "shots")
    ensure_dir(shots_dir)

    evidences: List[Evidence] = []
    category_scores = {c: 0.0 for c in OECD_CATEGORIES}

    friction: Dict[str, Any] = {
        "acceptClicks": None,
        "rejectClicks": None,
        "subscribeClicks": None,
        "cancelClicks": None,
        "acceptTrail": [],
        "rejectTrail": [],
        "subscribeTrail": [],
        "cancelTrail": [],
    }

    # calibration stats
    stats = load_calibration_stats() if use_calibration else {c: {"tp": 0, "fp": 0} for c in OECD_CATEGORIES}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            safe_wait_dom(page, 900)

            # Snapshot base
            shot0 = take_screenshot(page, shots_dir, "step0")
            evidences.append(Evidence(
                category="interface_interference",
                score=0.15,
                reason="Snapshot inicial coletado (base de auditoria).",
                url=url,
                screenshotPath=shot0
            ))
            bump(category_scores, "interface_interference", 0.15)

            text0 = get_body_text(page)
            static = scan_static_signals(text0)

            if static["hasUrgency"]:
                evidences.append(Evidence("urgency", 0.55, "Sinal textual de urgência/contador na página.", url=url))
                bump(category_scores, "urgency", 0.55)

            if static["hasSocialProof"]:
                evidences.append(Evidence("social_proof", 0.55, "Sinal textual de prova social na página.", url=url))
                bump(category_scores, "social_proof", 0.55)

            if static["hasNagging"]:
                evidences.append(Evidence("nagging", 0.50, "Sinal textual de pedido para habilitar notificações/localização.", url=url))
                bump(category_scores, "nagging", 0.50)

            if static["hasForced"]:
                evidences.append(Evidence("forced_action", 0.60, "Sinal textual de acesso condicionado (login/cadastro/consentimento).", url=url))
                bump(category_scores, "forced_action", 0.60)

            # --- friction: accept vs reject (consent-ish)
            accept_clicks, accept_trail = attempt_action_path(page, RX["accept"], RX_EXPLORE, max_depth=0, out_dir=shots_dir, name="accept")
            reject_clicks, reject_trail = attempt_action_path(page, RX["reject"], RX_EXPLORE, max_depth=max_depth, out_dir=shots_dir, name="reject")

            friction["acceptClicks"] = accept_clicks
            friction["rejectClicks"] = reject_clicks
            friction["acceptTrail"] = [asdict(s) for s in accept_trail]
            friction["rejectTrail"] = [asdict(s) for s in reject_trail]

            if accept_clicks is not None or reject_clicks is not None:
                a = accept_clicks if accept_clicks is not None else 0
                r = reject_clicks if reject_clicks is not None else (max_depth + 1)
                if r > a:
                    delta = min(1.0, (r - a) / 4.0)
                    score = 0.55 + 0.35 * delta
                    evidences.append(Evidence(
                        category="obstruction",
                        score=score,
                        reason=f"Assimetria de fricção: recusar parece exigir mais passos do que aceitar (proxy: {a} vs {r}).",
                        url=url,
                        meta={"acceptClicks": a, "rejectClicks": r}
                    ))
                    bump(category_scores, "obstruction", score)

                    evidences.append(Evidence(
                        category="interface_interference",
                        score=0.50,
                        reason="Possível interferência de interface: caminho ‘aceitar/continuar’ privilegiado frente a ‘recusar/fechar’.",
                        url=url
                    ))
                    bump(category_scores, "interface_interference", 0.50)

            # --- subscribe vs cancel
            subscribe_clicks, subscribe_trail = attempt_action_path(page, RX["subscribe"], RX["accept"], max_depth=0, out_dir=shots_dir, name="subscribe")
            cancel_clicks, cancel_trail = attempt_action_path(page, RX["cancel"], RX["accept"], max_depth=max_depth, out_dir=shots_dir, name="cancel")

            friction["subscribeClicks"] = subscribe_clicks
            friction["cancelClicks"] = cancel_clicks
            friction["subscribeTrail"] = [asdict(s) for s in subscribe_trail]
            friction["cancelTrail"] = [asdict(s) for s in cancel_trail]

            if subscribe_clicks is not None:
                s = subscribe_clicks
                c = cancel_clicks if cancel_clicks is not None else (max_depth + 1)
                if c > s:
                    delta = min(1.0, (c - s) / 4.0)
                    score = 0.55 + 0.35 * delta
                    evidences.append(Evidence(
                        category="obstruction",
                        score=score,
                        reason=f"Assimetria de fricção: assinar/comprar é mais acessível do que cancelar/opt-out (proxy: {s} vs {c}).",
                        url=url,
                        meta={"subscribeClicks": s, "cancelClicks": c}
                    ))
                    bump(category_scores, "obstruction", score)

            # --- sneaking proxy: price appears after one advance
            had_price0 = bool(RX["price"].search(text0))
            advance = find_best_clickable_by_regex(page, RX["accept"])
            if advance:
                sel, txt = advance
                ok = click_and_wait(page, sel)
                shot1 = take_screenshot(page, shots_dir, "step1")
                text1 = get_body_text(page)
                has_price1 = bool(RX["price"].search(text1))
                if ok and (not had_price0) and has_price1:
                    evidences.append(Evidence(
                        category="sneaking",
                        score=0.50,
                        reason="Possível ‘drip pricing’: sinal de preço/custo aparece após avanço no fluxo (proxy).",
                        url=url,
                        screenshotPath=shot1,
                        meta={"advanceClicked": txt}
                    ))
                    bump(category_scores, "sneaking", 0.50)

        finally:
            context.close()
            browser.close()

    # Apply calibration multipliers (optional)
    if use_calibration:
        for e in evidences:
            mult = calibration_multiplier_fixed(stats, e.category)
            e.score = clamp01(e.score * mult)

    # Aggregate category scores
    for e in evidences:
        bump(category_scores, e.category, e.score)

    finished_at = now_iso()

    data = {
        "caseId": case_id,
        "caseDir": case_dir,
        "url": url,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "evidences": [asdict(e) for e in evidences],
        "summary": {
            "categoryScores": category_scores,
            "friction": {
                "acceptClicks": friction["acceptClicks"],
                "rejectClicks": friction["rejectClicks"],
                "subscribeClicks": friction["subscribeClicks"],
                "cancelClicks": friction["cancelClicks"],
            }
        },
        "trails": {
            "accept": friction["acceptTrail"],
            "reject": friction["rejectTrail"],
            "subscribe": friction["subscribeTrail"],
            "cancel": friction["cancelTrail"],
        }
    }

    # Save report artifacts
    html = render_report_html(data)
    json_path, html_path = save_case_file(case_dir, data, html)
    data["artifacts"] = {
        "resultJson": json_path,
        "reportHtml": html_path,
        "shotsDir": shots_dir
    }

    return data


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fair Design Scanner", layout="wide")
st.title("Fair Design — Dark Patterns Scanner")

with st.sidebar:
    st.header("Scan")
    url = st.text_input("URL", placeholder="https://...")
    max_depth = st.slider("Profundidade de exploração (cliques)", 0, 8, 4)
    headed = st.checkbox("Modo visível (headed)", value=False, help="Abre o Chromium visível; ajuda contra alguns bloqueios.")
    use_cal = st.checkbox("Usar calibração (reduz falsos positivos)", value=True)
    run = st.button("Escanear")

st.caption("Dica: para evitar ambiente errado, rode com `python -m streamlit run app.py` dentro do `.venv`.")

stats = load_calibration_stats()
with st.expander("Calibração (histórico)", expanded=False):
    rows = []
    for cat in OECD_CATEGORIES:
        tp = stats[cat]["tp"]
        fp = stats[cat]["fp"]
        total = tp + fp
        prec = (tp / total) if total else None
        rows.append({
            "Categoria": CATEGORY_LABEL.get(cat, cat),
            "TP": tp,
            "FP": fp,
            "Precisão": (round(prec, 2) if prec is not None else "-"),
            "Multiplicador": round(calibration_multiplier_fixed(stats, cat), 2)
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

if run:
    if not url:
        st.error("Digite uma URL válida.")
        st.stop()

    with st.spinner("Escaneando com Playwright..."):
        try:
            data = crawl(url=url, max_depth=int(max_depth), headed=bool(headed), use_calibration=bool(use_cal))
        except Exception as e:
            st.error(f"Erro ao escanear: {e}")
            st.stop()

    st.success(f"Scan concluído. Case ID: {data['caseId']}")
    st.write("Arquivos salvos em:", data["caseDir"])

    # Summary
    st.subheader("Resumo")
    scores = data["summary"]["categoryScores"]
    df = pd.DataFrame(
        [{"Categoria": CATEGORY_LABEL.get(k, k), "Score": float(v)} for k, v in scores.items()]
    ).sort_values("Score", ascending=False)

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.dataframe(df, use_container_width=True)
    with col2:
        st.markdown("**Fricção (métrica comparativa)**")
        st.write(data["summary"]["friction"])

    # Export buttons
    st.subheader("Exportar relatório")
    a = data["artifacts"]
    colA, colB, colC = st.columns(3)
    with colA:
        with open(a["reportHtml"], "rb") as f:
            st.download_button("Baixar HTML", f, file_name="report.html", mime="text/html")
    with colB:
        with open(a["resultJson"], "rb") as f:
            st.download_button("Baixar JSON", f, file_name="result.json", mime="application/json")
    with colC:
        pdf_path = os.path.join(data["caseDir"], "report.pdf")
        if st.button("Gerar PDF (Playwright)"):
            try:
                export_pdf_via_playwright(a["reportHtml"], pdf_path)
                st.success("PDF gerado.")
            except Exception as e:
                st.error(f"Falhou ao gerar PDF: {e}")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("Baixar PDF", f, file_name="report.pdf", mime="application/pdf")

    # Trails
    st.subheader("Trilhas de interação (para auditoria)")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        with st.expander("Consent: aceitar (trail)", expanded=False):
            st.json(data["trails"]["accept"])
        with st.expander("Subscribe (trail)", expanded=False):
            st.json(data["trails"]["subscribe"])
    with tcol2:
        with st.expander("Consent: recusar (trail)", expanded=False):
            st.json(data["trails"]["reject"])
        with st.expander("Cancel/opt-out (trail)", expanded=False):
            st.json(data["trails"]["cancel"])

    # Evidences + labeling
    st.subheader("Evidências (com calibração)")
    evidences = data.get("evidences", [])
    if not evidences:
        st.info("Nenhuma evidência registrada.")
    else:
        df2 = pd.DataFrame([
            {
                "Categoria": CATEGORY_LABEL.get(e.get("category"), e.get("category")),
                "Score": e.get("score"),
                "Razão": e.get("reason"),
                "Screenshot": e.get("screenshotPath"),
            }
            for e in evidences
        ]).sort_values("Score", ascending=False)
        st.dataframe(df2, use_container_width=True)

        for idx, e in enumerate(sorted(evidences, key=lambda x: x.get("score", 0), reverse=True)[:30], start=1):
            cat = e.get("category")
            cat_label = CATEGORY_LABEL.get(cat, cat)
            score = float(e.get("score", 0))
            evidence_id = f"{data['caseId']}::{idx}::{cat}::{hashlib.md5((e.get('reason','')+str(e.get('screenshotPath',''))).encode('utf-8')).hexdigest()[:8]}"

            with st.expander(f"{idx}. {cat_label} — score {score:.2f}"):
                st.write(e.get("reason"))
                if e.get("textSnippet"):
                    st.code(e["textSnippet"])
                if e.get("meta"):
                    st.json(e["meta"])
                if e.get("screenshotPath") and os.path.exists(e["screenshotPath"]):
                    st.image(e["screenshotPath"], use_container_width=True)

                # Labeling UI
                lcol1, lcol2, lcol3 = st.columns([1,1,2])
                with lcol1:
                    if st.button("✅ Válido", key=f"tp_{evidence_id}"):
                        save_label(evidence_id, data["url"], cat, "tp")
                        st.success("Salvo como válido (TP).")
                with lcol2:
                    if st.button("❌ Falso positivo", key=f"fp_{evidence_id}"):
                        save_label(evidence_id, data["url"], cat, "fp")
                        st.success("Salvo como falso positivo (FP).")
                with lcol3:
                    note = st.text_input("Nota (opcional)", key=f"note_{evidence_id}")
                    if st.button("Salvar nota", key=f"savenote_{evidence_id}"):
                        # salva como tp/fp? aqui só salva nota neutra com label 'tp' não; melhor: escreve linha com label 'tp'/'fp' somente quando escolher
                        save_label(evidence_id, data["url"], cat, "tp", note=note)  # se quiser neutro, criamos terceiro tipo; por ora deixo simples
                        st.info("Nota salva (como TP por padrão). Ajusto se você quiser neutro.")

    st.subheader("JSON bruto")
    st.code(json.dumps(data, indent=2, ensure_ascii=False))

else:
    st.info("Digite uma URL e clique em **Escanear**.")
