import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/convert_diffsynth_cache.py"
spec = importlib.util.spec_from_file_location("convert_diffsynth_cache", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class CacheConversionTest(unittest.TestCase):
    def make_cache(self, root):
        source = root / "source"
        (source / "0").mkdir(parents=True)
        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
            writer.writeheader()
            writer.writerows([
                {"video": "a.mp4", "prompt": "Two boxing bears."},
                {"video": "b.mp4", "prompt": "Two boxing cats."},
            ])
        latent = torch.arange(1 * 16 * 2 * 2 * 2).reshape(1, 16, 2, 2, 2).to(torch.bfloat16)
        context = torch.ones(1, 8, 4096, dtype=torch.bfloat16)
        for index in range(2):
            torch.save((
                {"input_latents": latent + index, "height": 16, "width": 16, "num_frames": 5},
                {"context": context + index}, {},
            ), source / "0" / f"{index:06d}.pth")
        manifest = {
            "cache_version": 1, "complete": True, "sample_count": 2,
            "files": ["0/000001.pth", "0/000000.pth"],
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "height": 16, "width": 16, "num_frames": 5,
        }
        (source / "cache_manifest.json").write_text(json.dumps(manifest))
        return source, metadata, latent, context

    def test_preserves_reference_tensors_and_maps_prompt_by_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, latent, context = self.make_cache(root)
            original_hash = hashlib.sha256((source / "0/000000.pth").read_bytes()).hexdigest()
            output = root / "converted"
            result = converter.convert_cache(source, metadata, output, repeat=3)
            self.assertEqual(result["training_records"], 6)
            records = [json.loads(line) for line in (output / "train_data.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(records[0], records[2])
            cache = torch.load(output / records[0]["training_cache"], weights_only=True)
            self.assertEqual(cache["conditioning"]["prompt"], "Two boxing bears.")
            torch.testing.assert_close(cache["inputs"]["latents"], latent, rtol=0, atol=0)
            torch.testing.assert_close(cache["conditioning"]["positive"]["prompt_embed"], context, rtol=0, atol=0)
            self.assertEqual(hashlib.sha256((source / "0/000000.pth").read_bytes()).hexdigest(), original_hash)
            with self.assertRaises(FileExistsError):
                converter.convert_cache(source, metadata, output)

    def test_rejects_metadata_from_another_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, _, _ = self.make_cache(root)
            metadata.write_text(metadata.read_text().replace("bears", "wolves"))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                converter.convert_cache(source, metadata, root / "converted")
            self.assertFalse((root / "converted").exists())


if __name__ == "__main__":
    unittest.main()
