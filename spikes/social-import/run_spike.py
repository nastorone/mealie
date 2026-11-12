# ruff: noqa
# /// script
# dependencies = [
#   "google-genai",
#   "yt-dlp",
#   "requests",
#   "beautifulsoup4",
#   "python-dotenv",
#   "pydantic"
# ]
# ///

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure local dir is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ai_extractor import AIExtractor
from content_retriever import ContentRetriever


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(description="Spike import ricette da social (Instagram, TikTok, YouTube)")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di reel da processare")
    parser.add_argument("--offset", type=int, default=0, help="Offset di partenza nel dataset")
    parser.add_argument("--skip-model-b", action="store_true", help="Salta il modello B per prove rapide")
    args = parser.parse_args()

    # 1. Load env
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        # fallback to home
        load_dotenv(Path.home() / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERRORE: GEMINI_API_KEY non trovata nel file .env!")
        sys.exit(1)

    print("=== SPIKE IMPORT SOCIAL (Instagram, TikTok, YouTube) ===")
    print("Inizializzazione ambiente e verifiche...")

    # Load dataset
    dataset_path = SCRIPT_DIR / "reels_dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        reels = json.load(f)

    if args.offset:
        reels = reels[args.offset :]
    if args.limit:
        reels = reels[: args.limit]

    output_dir = SCRIPT_DIR / "output"
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "risultati_spike.csv"
    metodi_csv_path = output_dir / "confronto_metodi_recupero.csv"

    retriever = ContentRetriever()
    ai_extractor = AIExtractor()

    results_rows = []
    methods_benchmark = []

    models_to_test = ["gemini-3.5-flash-lite"]
    if not args.skip_model_b:
        models_to_test.append("gemini-3.5-flash")

    print(f"Totale reel da esaminare: {len(reels)}")
    print(f"Modelli AI da confrontare: {', '.join(models_to_test)}\n")

    for idx, item in enumerate(reels, 1):
        r_id = item["id"]
        url = item["url"]
        platform = item["platform"]
        quantities_location = item["quantities_location"]
        notes = item.get("notes", "")

        print(f"[{idx}/{len(reels)}] Reel #{r_id} ({platform}): {url}")

        # A. Benchmark Method 1: Official oEmbed
        oembed_res = retriever.test_official_oembed(url, platform)

        # B. Benchmark Method 2: HTML Metadata Scraping
        meta_res = retriever.test_metadata_scraping(url, platform)

        # Record retrieval comparison for the report
        methods_benchmark.append(
            {
                "reel_id": r_id,
                "platform": platform,
                "url": url,
                "oembed_success": oembed_res["success"],
                "oembed_time_s": oembed_res["time_s"],
                "oembed_error": oembed_res["error"] or "",
                "meta_success": meta_res["success"],
                "meta_time_s": meta_res["time_s"],
                "meta_has_desc": bool(meta_res.get("description")),
                "meta_error": meta_res["error"] or "",
            }
        )

        # C. Approach 3: Full retrieval (Caption, Audio, Cover, Carousel)
        content = retriever.retrieve_full_content(r_id, url, platform)
        retrieval_status = content["retrieval_status"]
        has_caption = bool(content["caption"])
        has_audio = bool(content["audio_path"] and os.path.exists(content["audio_path"]))
        has_cover = bool(content["thumbnail_url"])
        audio_size_kb = round(os.path.getsize(content["audio_path"]) / 1024, 1) if has_audio else 0

        print(
            f"  -> Recupero: {retrieval_status} in {content['time_s']}s (Didascalia: {has_caption}, Audio: {has_audio} [{audio_size_kb} KB], Copertina: {has_cover})"
        )
        if content.get("images"):
            print(f"  -> Immagini carosello trovate: {len(content['images'])}")

        # D. Run AI Extraction for each model
        for model_name in models_to_test:
            model_tag = "model_a" if "flash-lite" in model_name else "model_b"
            print(f"  -> Chiamata AI con {model_name}...", end="", flush=True)

            ai_res = ai_extractor.extract_recipe(
                model_name=model_name, caption=content["caption"], audio_path=content["audio_path"]
            )

            total_time = round(content["time_s"] + ai_res["time_s"], 3)

            # Count ingredients
            total_ingredients = 0
            ingredients_with_qty = 0
            recipe_titles = []
            warnings = []

            for recipe in ai_res.get("recipes", []):
                recipe_titles.append(recipe.get("titolo", "Senza titolo"))
                warnings.extend(recipe.get("da_verificare", []))
                for ing in recipe.get("ingredienti", []):
                    total_ingredients += 1
                    if ing.get("quantita"):
                        ingredients_with_qty += 1

            print(
                f" completata in {ai_res['time_s']}s (Ricette: {len(recipe_titles)}, Ingr: {total_ingredients}, Con dose: {ingredients_with_qty}, Costo: ${ai_res['estimated_cost_usd']:.5f})"
            )

            # Save individual JSON
            platform_slug = slugify(platform)
            json_filename = f"{r_id:02d}_{platform_slug}_{model_tag}.json"
            json_filepath = json_dir / json_filename

            full_saved_data = {
                "reel_id": r_id,
                "url": url,
                "platform": platform,
                "quantities_location_declared": quantities_location,
                "notes_declared": notes,
                "retrieval": {
                    "status": retrieval_status,
                    "method": content["method_used"],
                    "time_s": content["time_s"],
                    "has_caption": has_caption,
                    "has_audio": has_audio,
                    "has_cover": has_cover,
                    "caption_char_len": len(content["caption"]) if content["caption"] else 0,
                    "thumbnail_url": content["thumbnail_url"],
                    "carousel_images_count": len(content.get("images", [])),
                },
                "ai_extraction": {
                    "model": model_name,
                    "model_tag": model_tag,
                    "success": ai_res["success"],
                    "latency_s": ai_res["time_s"],
                    "total_time_s": total_time,
                    "input_tokens": ai_res["input_tokens"],
                    "output_tokens": ai_res["output_tokens"],
                    "total_tokens": ai_res["total_tokens"],
                    "estimated_cost_usd": ai_res["estimated_cost_usd"],
                    "recipes_count": len(recipe_titles),
                    "recipes": ai_res.get("recipes", []),
                    "warnings_count": len(warnings),
                },
            }

            with open(json_filepath, "w", encoding="utf-8") as jf:
                json.dump(full_saved_data, jf, indent=2, ensure_ascii=False)

            # Record row for main CSV
            results_rows.append(
                {
                    "reel_id": r_id,
                    "piattaforma": platform,
                    "url": url,
                    "dove_quantita": quantities_location,
                    "metodo_recupero": content["method_used"],
                    "modello": model_name,
                    "esito_recupero": retrieval_status,
                    "didascalia_presente": "SI" if has_caption else "NO",
                    "audio_presente": "SI" if has_audio else "NO",
                    "copertina_presente": "SI" if has_cover else "NO",
                    "ricette_estratte": len(recipe_titles),
                    "ingredienti_estratti": total_ingredients,
                    "ingredienti_con_quantita": ingredients_with_qty,
                    "segnalazioni_da_verificare": len(warnings),
                    "token_input": ai_res["input_tokens"],
                    "token_output": ai_res["output_tokens"],
                    "costo_stimato_usd": f"{ai_res['estimated_cost_usd']:.6f}",
                    "tempo_recupero_s": content["time_s"],
                    "tempo_ai_s": ai_res["time_s"],
                    "tempo_totale_s": total_time,
                    "file_json": json_filename,
                    "note": f"{notes} | {content['notes']}".strip(" |"),
                }
            )

            # Small pause between calls
            time.sleep(1.0)

        # Clean local audio
        if content["audio_path"] and os.path.exists(content["audio_path"]):
            try:
                os.remove(content["audio_path"])
            except Exception:
                pass

        # Save Main CSV incrementally
        fieldnames = [
            "reel_id",
            "piattaforma",
            "url",
            "dove_quantita",
            "metodo_recupero",
            "modello",
            "esito_recupero",
            "didascalia_presente",
            "audio_presente",
            "copertina_presente",
            "ricette_estratte",
            "ingredienti_estratti",
            "ingredienti_con_quantita",
            "segnalazioni_da_verificare",
            "token_input",
            "token_output",
            "costo_stimato_usd",
            "tempo_recupero_s",
            "tempo_ai_s",
            "tempo_totale_s",
            "file_json",
            "note",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)

        # Save Methods Comparison CSV incrementally
        with open(metodi_csv_path, "w", newline="", encoding="utf-8") as f:
            m_fieldnames = [
                "reel_id",
                "platform",
                "url",
                "oembed_success",
                "oembed_time_s",
                "oembed_error",
                "meta_success",
                "meta_time_s",
                "meta_has_desc",
                "meta_error",
            ]
            writer = csv.DictWriter(f, fieldnames=m_fieldnames)
            writer.writeheader()
            writer.writerows(methods_benchmark)

        print()
        # Polite pause between reels
        time.sleep(1.5)

    print("=== SPIKE COMPLETATO CON SUCCESSO! ===")
    print(f"File CSV salvato in: {csv_path}")
    print(f"File confronto metodi salvato in: {metodi_csv_path}")
    print(f"File JSON salvati in: {json_dir}")
    print(f"File JSON salvati in: {json_dir}")


if __name__ == "__main__":
    main()
