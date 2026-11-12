# ruff: noqa
import json
import sys
import unittest
from pathlib import Path

spike_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(spike_dir))

from ai_extractor import PRICING, ExtractionOutput, Ingredient, Recipe


class TestSocialSpike(unittest.TestCase):
    def test_pydantic_recipe_schema(self):
        ing = Ingredient(quantita="200", unita="g", alimento="farina", nota="setacciata")
        rec = Recipe(
            titolo="Torta semplice",
            porzioni="4",
            tempo="30 min",
            ingredienti=[ing],
            passaggi=["Mescola tutto", "Inforna a 180°C"],
            tag=["dolce", "forno"],
            da_verificare=["Controllare tipo di farina"],
        )
        output = ExtractionOutput(ricette=[rec])

        self.assertEqual(len(output.ricette), 1)
        self.assertEqual(output.ricette[0].titolo, "Torta semplice")
        self.assertEqual(output.ricette[0].ingredienti[0].alimento, "farina")
        self.assertEqual(output.ricette[0].ingredienti[0].quantita, "200")

    def test_pydantic_recipe_null_quantities(self):
        ing = Ingredient(quantita=None, unita=None, alimento="sale")
        rec = Recipe(titolo="Pasta", ingredienti=[ing], da_verificare=["Sale: quantità non indicata"])
        self.assertIsNone(rec.ingredienti[0].quantita)
        self.assertIsNone(rec.ingredienti[0].unita)
        self.assertEqual(len(rec.da_verificare), 1)

    def test_pricing_calculation(self):
        self.assertIn("gemini-3.5-flash-lite", PRICING)
        self.assertIn("gemini-3.5-flash", PRICING)
        info = PRICING["gemini-3.5-flash-lite"]
        self.assertEqual(info["input_per_million"], 0.30)
        self.assertEqual(info["output_per_million"], 2.50)

    def test_dataset_json_integrity(self):
        dataset_file = spike_dir / "reels_dataset.json"
        self.assertTrue(dataset_file.exists(), "reels_dataset.json deve esistere")
        with open(dataset_file, encoding="utf-8") as f:
            reels = json.load(f)
        self.assertEqual(len(reels), 25, "Il dataset deve contenere esattamente 25 reel")
        for r in reels:
            self.assertIn("id", r)
            self.assertIn("url", r)
            self.assertIn("platform", r)
            self.assertIn("quantities_location", r)

    def test_output_csv_and_json_exist(self):
        csv_file = spike_dir / "output" / "risultati_spike.csv"
        self.assertTrue(csv_file.exists(), "risultati_spike.csv deve esistere")
        json_dir = spike_dir / "output" / "json"
        self.assertTrue(json_dir.exists(), "directory output/json deve esistere")
        json_files = list(json_dir.glob("*.json"))
        self.assertEqual(len(json_files), 50, f"Attesi 50 file JSON (25 per 2 modelli), trovati {len(json_files)}")


if __name__ == "__main__":
    unittest.main()
