# ruff: noqa
import json
import os
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    quantita: str | None = Field(
        default=None, description="Quantità numerica o dose (es. 200, 1/2, q.b.). Lasciare null se non specificata."
    )
    unita: str | None = Field(
        default=None, description="Unità di misura (es. g, ml, cucchiai, pizzico). Lasciare null se non specificata."
    )
    alimento: str = Field(description="Nome dell'ingrediente o alimento (es. pasta, olio evo, parmigiano)")
    nota: str | None = Field(
        default=None, description="Dettagli aggiuntivi (es. a temperatura ambiente, tritato finemente)"
    )


class Recipe(BaseModel):
    titolo: str = Field(description="Titolo chiaro e descrittivo della ricetta")
    porzioni: str | None = Field(default=None, description="Numero di porzioni o persone, se specificato")
    tempo: str | None = Field(default=None, description="Tempo di preparazione o cottura, se indicato")
    ingredienti: list[Ingredient] = Field(default_factory=list, description="Elenco strutturato degli ingredienti")
    passaggi: list[str] = Field(
        default_factory=list, description="Passaggi ordinati della preparazione, ripuliti da convenevoli social"
    )
    tag: list[str] = Field(
        default_factory=list, description="Tag e categorie suggeriti (es. primo piatto, vegetariano, pasta)"
    )
    da_verificare: list[str] = Field(
        default_factory=list,
        description="Dubbi, quantità mancanti, passaggi omessi, ambiguità o parti da verificare con attenzione",
    )


class ExtractionOutput(BaseModel):
    ricette: list[Recipe] = Field(description="Lista di una o più ricette estratte dal post")


# Pricing as of Nov 2026 (USD per token)
PRICING = {
    "gemini-3.5-flash-lite": {
        "input_per_million": 0.30,
        "output_per_million": 2.50,
        "input_cost_per_token": 0.30 / 1_000_000,
        "output_cost_per_token": 2.50 / 1_000_000,
        "name": "Gemini 3.5 Flash-Lite (Modello A)",
    },
    "gemini-3.5-flash": {
        "input_per_million": 1.50,
        "output_per_million": 9.00,
        "input_cost_per_token": 1.50 / 1_000_000,
        "output_cost_per_token": 9.00 / 1_000_000,
        "name": "Gemini 3.5 Flash (Modello B)",
    },
}

SYSTEM_INSTRUCTION = """Sei un assistente culinario specializzato nell'estrazione precisa di ricette da reel e post social (Instagram, TikTok, YouTube).
Dovrai estrarre i dati sia dal testo della didascalia che dall'audio (se disponibile).

Regole tassative:
1. Pulisci completamente il testo: rimuovi convenevoli, saluti ("ciao a tutti!"), inviti all'azione social ("lascia un like", "salva il reel", "segui il mio profilo"), pubblicità o sponsor.
2. Gestione ricette multiple: Se il post contiene più ricette distinte (es. un menu completo con primo e secondo, o un carosello con 3 marinate diverse), restituisci ciascuna ricetta come elemento separato nell'array 'ricette'.
3. FEDELTÀ ASSOLUTA: NON inventare ingredienti, quantità o passaggi.
   - Se per un ingrediente non c'è la dose, lascia 'quantita' e 'unita' a null e aggiungi una nota in 'da_verificare' (es. "Olio: quantità non specificata").
   - Se nel video o didascalia mancano i passaggi o le quantità, NON inventarli: segnala la mancanza in 'da_verificare'.
4. Campo 'da_verificare': Usalo per qualsiasi incertezza o informazione incompleta (es. "Mancano tempi di cottura", "Dosi per persone non indicate", "Quantità mostrate solo a vista nel video").
"""


class AIExtractor:
    def __init__(self):
        self.client = genai.Client()

    def extract_recipe(
        self,
        model_name: str,
        caption: str | None,
        audio_path: str | None = None,
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        uploaded_files = []
        contents = []

        result_data = {
            "model": model_name,
            "success": False,
            "recipes": [],
            "raw_json": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "time_s": 0.0,
            "error": None,
        }

        try:
            # 1. Upload audio if present
            if audio_path and os.path.exists(audio_path):
                try:
                    up_file = self.client.files.upload(file=audio_path)
                    uploaded_files.append(up_file)
                    contents.append(up_file)
                except Exception as e:
                    print(f" (Warning: fallito upload audio: {e})", end="", flush=True)

            # 2. Add text prompt
            user_prompt = f"Estrai la ricetta da questo post social.\n\nDidascalia del post:\n{caption or 'Nessuna didascalia disponibile.'}"
            contents.append(user_prompt)

            max_retries = 3
            backoff = 2.0
            resp = None

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ExtractionOutput,
                temperature=0.2,
            )

            for attempt in range(max_retries):
                try:
                    resp = self.client.models.generate_content(model=model_name, contents=contents, config=config)
                    break
                except Exception as e:
                    err_str = str(e)
                    if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries - 1:
                        print(f" (503 retry {attempt + 1}/{max_retries} in {backoff}s...)", end="", flush=True)
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        result_data["time_s"] = round(time.time() - t0, 3)
                        result_data["error"] = err_str
                        print(f"Errore chiamata AI ({model_name}): {e}")
                        break

            if resp is not None:
                result_data["time_s"] = round(time.time() - t0, 3)

                # Usage metadata
                if resp.usage_metadata:
                    in_tok = resp.usage_metadata.prompt_token_count or 0
                    out_tok = resp.usage_metadata.candidates_token_count or 0
                    tot_tok = resp.usage_metadata.total_token_count or (in_tok + out_tok)
                    result_data["input_tokens"] = in_tok
                    result_data["output_tokens"] = out_tok
                    result_data["total_tokens"] = tot_tok

                    pricing_info = PRICING.get(model_name, PRICING["gemini-3.5-flash-lite"])
                    cost = (in_tok * pricing_info["input_cost_per_token"]) + (
                        out_tok * pricing_info["output_cost_per_token"]
                    )
                    result_data["estimated_cost_usd"] = round(cost, 6)

                try:
                    parsed = json.loads(resp.text)
                    result_data["raw_json"] = parsed
                    result_data["recipes"] = parsed.get("ricette", [])
                    result_data["success"] = True
                except Exception as parse_err:
                    result_data["error"] = f"JSON parse error: {parse_err}"

        finally:
            # Clean up uploaded files in Gemini File API
            for f in uploaded_files:
                try:
                    self.client.files.delete(name=f.name)
                except Exception:
                    pass

        return result_data
