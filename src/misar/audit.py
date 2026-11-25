# -*- coding: utf-8 -*-
"""
Misar Enterprise ROI & Security Audit Engine (محرك التدقيق المالي والأمني لمِسار).

Ingests datasets (JSONL, CSV, TSV, or TXT), executes adaptive routing and prompt guard
evaluations, computes token reduction, quadratic attention FLOPs savings, financial ROI,
PII redactions, and adversarial threat mitigation, producing executive HTML/JSON audit reports.
"""

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .classifier import classify_dialect
from .guard import sanitize_prompt
from .router import MisarRouter
from .tokenizer import load_default_tokenizer


@dataclass
class AuditResult:
    """Comprehensive executive audit telemetry for enterprise dataset evaluation."""
    total_records: int
    raw_token_count: int
    processed_token_count: int
    tokens_saved: int
    token_reduction_pct: float
    raw_attention_flops: float
    processed_attention_flops: float
    attention_flops_savings_pct: float
    cost_per_1k_tokens: float
    currency: str
    raw_cost: float
    processed_cost: float
    money_saved: float
    masked_pii_count: int
    adversarial_risks_count: int
    detected_risks: Dict[str, int] = field(default_factory=dict)
    dialect_distribution: Dict[str, int] = field(default_factory=dict)
    route_distribution: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    records_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_texts(
    texts: List[str],
    cost_per_1k_tokens: float = 0.005,
    currency: str = "SAR",
    model_key: str = "arabertv02",
    router: Optional[MisarRouter] = None,
) -> AuditResult:
    """
    Perform a complete financial and security audit on an in-memory list of Arabic texts.
    """
    t0 = time.perf_counter()
    if router is None:
        tok = load_default_tokenizer(model_key, allow_download=False)
        router = MisarRouter(tokenizer=tok, model_key=model_key, auto_heal=True)

    raw_tok_sum = 0
    proc_tok_sum = 0
    raw_flops_sum = 0.0
    proc_flops_sum = 0.0

    masked_pii = 0
    adv_risks_count = 0
    detected_risks_map: Dict[str, int] = {}
    dialect_map: Dict[str, int] = {}
    route_map: Dict[str, int] = {"FAST_MSA": 0, "DIALECT_PRESERVE": 0, "NOISY_SOCIAL": 0}

    for text in texts:
        if not text:
            continue

        # 1. Security & PII inspection
        guard = sanitize_prompt(text, mask_pii=True, detect_injection=True)
        if guard.masked_pii_count:
            masked_pii += guard.masked_pii_count
        if guard.is_suspicious:
            adv_risks_count += 1
        for r in guard.detected_risks:
            detected_risks_map[r] = detected_risks_map.get(r, 0) + 1

        # 2. Dialect classification
        diag = classify_dialect(text)
        dialect_map[diag.primary_dialect] = dialect_map.get(diag.primary_dialect, 0) + 1

        # 3. Routing & token measurement
        res = router.route(guard.sanitized_text)
        route_map[res.route_name] = route_map.get(res.route_name, 0) + 1

        raw_len = max(1, res.raw_token_count)
        proc_len = max(1, res.processed_token_count)

        raw_tok_sum += raw_len
        proc_tok_sum += proc_len
        raw_flops_sum += float(raw_len * raw_len)
        proc_flops_sum += float(proc_len * proc_len)

    elapsed = max(1e-4, time.perf_counter() - t0)
    total_records = len(texts)
    tokens_saved = max(0, raw_tok_sum - proc_tok_sum)
    red_pct = ((raw_tok_sum - proc_tok_sum) / raw_tok_sum * 100.0) if raw_tok_sum else 0.0

    attn_saved_pct = ((raw_flops_sum - proc_flops_sum) / raw_flops_sum * 100.0) if raw_flops_sum else 0.0

    raw_cost = (raw_tok_sum / 1000.0) * cost_per_1k_tokens
    proc_cost = (proc_tok_sum / 1000.0) * cost_per_1k_tokens
    money_saved = raw_cost - proc_cost

    return AuditResult(
        total_records=total_records,
        raw_token_count=raw_tok_sum,
        processed_token_count=proc_tok_sum,
        tokens_saved=tokens_saved,
        token_reduction_pct=round(red_pct, 2),
        raw_attention_flops=round(raw_flops_sum, 2),
        processed_attention_flops=round(proc_flops_sum, 2),
        attention_flops_savings_pct=round(attn_saved_pct, 2),
        cost_per_1k_tokens=cost_per_1k_tokens,
        currency=currency.upper(),
        raw_cost=round(raw_cost, 4),
        processed_cost=round(proc_cost, 4),
        money_saved=round(money_saved, 4),
        masked_pii_count=masked_pii,
        adversarial_risks_count=adv_risks_count,
        detected_risks=detected_risks_map,
        dialect_distribution=dialect_map,
        route_distribution=route_map,
        elapsed_seconds=round(elapsed, 3),
        records_per_second=round(total_records / elapsed, 1),
    )


def audit_dataset_file(
    input_file: Union[str, Path],
    text_key: str = "text",
    cost_per_1k_tokens: float = 0.005,
    currency: str = "SAR",
    model_key: str = "arabertv02",
    max_records: Optional[int] = None,
) -> AuditResult:
    """
    Ingest a dataset file (JSONL, CSV, TSV, or TXT) and execute an audit.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_file}")

    texts: List[str] = []
    ext = path.suffix.lower()

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and text_key in obj:
                        texts.append(str(obj[text_key]))
                    elif isinstance(obj, str):
                        texts.append(obj)
                except Exception:
                    continue

    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                if max_records and i >= max_records:
                    break
                if text_key in row and row[text_key]:
                    texts.append(str(row[text_key]))

    else:
        # Plain text line by line
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                val = line.strip()
                if val:
                    texts.append(val)

    return audit_texts(
        texts=texts,
        cost_per_1k_tokens=cost_per_1k_tokens,
        currency=currency,
        model_key=model_key,
    )


def generate_html_report(result: AuditResult, title: str = "تقرير تدقيق مِسار التنفيذي") -> str:
    """
    Generate an executive, responsive, and standalone HTML dashboard report.
    """
    res_dict = result.to_dict()

    dialect_rows = "".join(
        f"<tr><td><strong>{k}</strong></td><td class='ltr'>{v}</td><td class='ltr'>{(v / max(1, result.total_records) * 100):.1f}%</td></tr>"
        for k, v in result.dialect_distribution.items()
    )

    risk_tags = "".join(
        f"<span class='risk-badge ltr'>{k}: {v}</span>"
        for k, v in result.detected_risks.items()
    ) or "<span style='color:#16a34a;'>لم يتم رصد أي مخاطر أمنية</span>"

    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <title>{title} — Misar v3.1.0</title>
  <style>
    :root {{
      --navy: #0f2942;
      --navy-light: #1e3a5f;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --green: #15803d;
      --green-bg: #f0fdf4;
      --amber: #b45309;
      --red: #b91c1c;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.6; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .header {{ background: var(--navy); color: #fff; padding: 24px; border-radius: 12px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; border-bottom: 4px solid #2563eb; }}
    .header h1 {{ font-size: 1.6rem; }}
    .header p {{ color: #94a3b8; font-size: 0.9rem; margin-top: 4px; }}
    .ltr {{ direction: ltr; unicode-bidi: isolate; font-family: Consolas, monospace; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .kpi-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .kpi-label {{ font-size: 0.82rem; color: var(--text-muted); font-weight: 600; display: block; }}
    .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: var(--navy); margin-top: 4px; }}
    .kpi-green {{ color: var(--green); }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .card-title {{ font-size: 1.15rem; font-weight: 700; color: var(--navy); margin-bottom: 14px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 10px 12px; border: 1px solid var(--border); text-align: right; }}
    th {{ background: #f1f5f9; color: var(--navy); font-weight: 600; }}
    .risk-badge {{ display: inline-block; background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin: 3px; }}
    .footer {{ text-align: center; font-size: 0.8rem; color: var(--text-muted); padding: 12px 0; }}
  </style>
</head>
<body>
  <div class="container">
    <header class="header">
      <div>
        <h1>{title}</h1>
        <p>تحليل العائد الاستثماري (ROI) وحجب المخاطر والبيانات الحساسة بواسطة مِسار</p>
      </div>
      <div style="text-align: left;">
        <span class="ltr" style="background:#1e3a5f; padding:6px 12px; border-radius:6px; font-weight:600;">Misar v3.1.0 Audit</span>
      </div>
    </header>

    <div class="kpi-grid">
      <div class="kpi-card">
        <span class="kpi-label">الوفر المالي المحقق في الفواتير</span>
        <div class="kpi-value kpi-green ltr">{result.money_saved:.4f} {result.currency}</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">نسبة خفض الرموز (Tokens Reduction)</span>
        <div class="kpi-value kpi-green ltr">-{result.token_reduction_pct:.1f}%</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">توفير كلفة الانتباه التربيعية O(L²)</span>
        <div class="kpi-value kpi-green ltr">-{result.attention_flops_savings_pct:.1f}%</div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">البيانات الحساسة المحجوبة (PII)</span>
        <div class="kpi-value ltr" style="color:var(--amber);">{result.masked_pii_count}</div>
      </div>
    </div>

    <div class="card">
      <h2 class="card-title">1. القياسات الرقمية المقارنة والرموز (Token Metrics)</h2>
      <table>
        <thead>
          <tr>
            <th>المؤشر الفني</th>
            <th>المدخلات الخام (Raw)</th>
            <th>معالجة مِسار (Misar)</th>
            <th>الفارق / التوفير</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>إجمالي النصوص المفحوصة</td>
            <td class="ltr">{result.total_records}</td>
            <td class="ltr">{result.total_records}</td>
            <td class="ltr">—</td>
          </tr>
          <tr>
            <td>عدد الرموز الكلي (Tokens)</td>
            <td class="ltr">{result.raw_token_count}</td>
            <td class="ltr">{result.processed_token_count}</td>
            <td class="ltr" style="color:var(--green); font-weight:700;">-{result.tokens_saved} (-{result.token_reduction_pct}%)</td>
          </tr>
          <tr>
            <td>تكلفة استهلاك النماذج التقديرية</td>
            <td class="ltr">{result.raw_cost:.4f} {result.currency}</td>
            <td class="ltr">{result.processed_cost:.4f} {result.currency}</td>
            <td class="ltr" style="color:var(--green); font-weight:700;">وفر {result.money_saved:.4f} {result.currency}</td>
          </tr>
          <tr>
            <td>سرعة الفحص والتدقيق</td>
            <td colspan="3" class="ltr">{result.records_per_second:.1f} نصوص/ثانية (استغرق {result.elapsed_seconds:.3f} ثانية)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2 class="card-title">2. تقرير السلامة الأمنية وحجب البيانات (Security & PII Protection)</h2>
      <p style="margin-bottom:12px; font-size:0.9rem; color:var(--text-muted);">
        حجب الهويات الوطنية السعودية، أرقام الآيبان البنكية، أرقام الهواتف، ورصد محاولات كسر الحماية وتطويل الحروف.
      </p>
      <div>{risk_tags}</div>
    </div>

    <div class="card">
      <h2 class="card-title">3. توزيع اللهجات المكتشفة في مجموعة البيانات (Dialect Distribution)</h2>
      <table>
        <thead>
          <tr>
            <th>اللهجة / الأسرة الإقليمية</th>
            <th>عدد العينات</th>
            <th>النسبة المئوية</th>
          </tr>
        </thead>
        <tbody>
          {dialect_rows}
        </tbody>
      </table>
    </div>

    <footer class="footer ltr">
      Generated by Misar (مِسار) v3.1.0 · Autonomous Arabic NLP Preprocessing Gateway
    </footer>
  </div>
</body>
</html>"""
