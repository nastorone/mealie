# ruff: noqa
import csv
import json
from collections import defaultdict
from pathlib import Path

data_dir = Path("spikes/social-import/output")
csv_path = data_dir / "risultati_spike.csv"
methods_path = data_dir / "confronto_metodi_recupero.csv"

with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with open(methods_path, encoding="utf-8") as f:
    method_rows = list(csv.DictReader(f))

print(f"Total rows in results: {len(rows)}")
print(f"Total rows in methods: {len(method_rows)}")

# 1. Methods comparison
print("\n=== METODI DI RECUPERO ===")
by_p_method = defaultdict(
    lambda: {
        "total": 0,
        "oembed_ok": 0,
        "oembed_time": 0.0,
        "meta_ok": 0,
        "meta_time": 0.0,
        "ytdlp_ok": 0,
        "ytdlp_time": 0.0,
    }
)
for m in method_rows:
    p = m["platform"]
    by_p_method[p]["total"] += 1
    if m["oembed_success"] == "True":
        by_p_method[p]["oembed_ok"] += 1
    by_p_method[p]["oembed_time"] += float(m["oembed_time_s"])
    if m["meta_success"] == "True":
        by_p_method[p]["meta_ok"] += 1
    by_p_method[p]["meta_time"] += float(m["meta_time_s"])

# Check full retrieval success from results rows (model A)
rows_model_a = [r for r in rows if r["modello"] == "gemini-3.5-flash-lite"]
for r in rows_model_a:
    p = r["piattaforma"]
    if r["esito_recupero"] in ("SUCCESS", "PARTIAL"):
        by_p_method[p]["ytdlp_ok"] += 1
    by_p_method[p]["ytdlp_time"] += float(r["tempo_recupero_s"])

for p, d in by_p_method.items():
    tot = d["total"]
    print(f"\nPiattaforma: {p} (N={tot})")
    print(
        f"  oEmbed: {d['oembed_ok']}/{tot} ({d['oembed_ok'] / tot * 100:.1f}%), Tempo medio: {d['oembed_time'] / tot:.2f}s"
    )
    print(
        f"  Scraping HTML: {d['meta_ok']}/{tot} ({d['meta_ok'] / tot * 100:.1f}%), Tempo medio: {d['meta_time'] / tot:.2f}s"
    )
    print(
        f"  yt-dlp + fallback: {d['ytdlp_ok']}/{tot} ({d['ytdlp_ok'] / tot * 100:.1f}%), Tempo medio: {d['ytdlp_time'] / tot:.2f}s"
    )

# 2. Model comparison
print("\n=== CONFRONTO MODELLI ===")
for model in ["gemini-3.5-flash-lite", "gemini-3.5-flash"]:
    m_rows = [r for r in rows if r["modello"] == model]
    tot = len(m_rows)
    successes = [r for r in m_rows if int(r["ricette_estratte"]) > 0]
    costs = [float(r["costo_stimato_usd"]) for r in successes if float(r["costo_stimato_usd"]) > 0]
    ai_times = [float(r["tempo_ai_s"]) for r in successes]
    tot_times = [float(r["tempo_totale_s"]) for r in successes]
    ingrs = [int(r["ingredienti_estratti"]) for r in successes]
    with_qty = [int(r["ingredienti_con_quantita"]) for r in successes]
    warnings = [int(r["segnalazioni_da_verificare"]) for r in successes]

    print(f"\nModello: {model}")
    print(
        f"  Totale testati: {tot}, Ricette estratte con successo: {len(successes)}/{tot} ({len(successes) / tot * 100:.1f}%)"
    )
    if costs:
        print(f"  Costo medio: ${sum(costs) / len(costs):.6f}")
        print(f"  Costo min: ${min(costs):.6f}, Costo max: ${max(costs):.6f}")
    if ai_times:
        print(
            f"  Tempo AI medio: {sum(ai_times) / len(ai_times):.2f}s (Min: {min(ai_times):.2f}s, Max: {max(ai_times):.2f}s)"
        )
        print(f"  Tempo totale medio: {sum(tot_times) / len(tot_times):.2f}s")
    if ingrs:
        print(f"  Ingredienti medi: {sum(ingrs) / len(ingrs):.1f}")
        print(f"  Ingredienti con dose medi: {sum(with_qty) / len(with_qty):.1f}")
        print(f"  Segnalazioni 'da verificare' medie: {sum(warnings) / len(warnings):.1f}")

# 3. Ground truth: Where are quantities
print("\n=== CONFRONTO CON LA VERITÀ (DOVE SONO LE QUANTITÀ) ===")
qty_loc_counts = defaultdict(int)
need_frames_count = 0
for r in rows_model_a:
    loc = r["dove_quantita"]
    qty_loc_counts[loc] += 1
    if any(k in loc.lower() for k in ["nel video", "scritte sul video", "video"]):
        if "descrizione" not in loc.lower():
            need_frames_count += 1

print("Distribuzione posizioni quantità (su 25 reel):")
for loc, cnt in qty_loc_counts.items():
    print(f"  - {loc}: {cnt} ({cnt / 25 * 100:.1f}%)")

print(
    f"\nReel che necessitano lettura dei fotogrammi (quantità SOLO nel video): {need_frames_count}/25 ({need_frames_count / 25 * 100:.1f}%)"
)

# 4. Check edge cases: row 4, 8, 9, 11
print("\n=== CASI LIMITE ===")
for r_id in [4, 8, 9, 11]:
    r_a = next(r for r in rows_model_a if int(r["reel_id"]) == r_id)
    json_f = data_dir / "json" / r_a["file_json"]
    with open(json_f, encoding="utf-8") as jf:
        j_data = json.load(jf)
    ai_info = j_data["ai_extraction"]
    print(f"Reel #{r_id} ({r_a['piattaforma']}) - {r_a['dove_quantita']}:")
    print(f"  Ricette trovate: {ai_info['recipes_count']}")
    for rc in ai_info.get("recipes", []):
        print(
            f"    * '{rc['titolo']}' (Ingr: {len(rc.get('ingredienti', []))}, Passaggi: {len(rc.get('passaggi', []))}, Da verificare: {len(rc.get('da_verificare', []))})"
        )
        if rc.get("da_verificare"):
            for dv in rc["da_verificare"][:2]:
                print(f"      - [da verificare]: {dv}")
