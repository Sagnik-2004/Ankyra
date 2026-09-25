"""Fast integration smoke tests. No external model downloads or API calls."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import main1
import main2
from kg_retrieve.kg_retriever import KnowledgeGraphRetriever
from kg_retrieve.qa_generator import KnowledgeGraphQA, INSUFFICIENT, run_qa


SAMPLE_GRAPH = {
    "source_chunks": ["chunk_0001.txt"],
    "entities": [
        {"id": "e0", "text": "Ada", "type": "person", "confidence": .94,
         "mentions": 1, "source_chunks": ["chunk_0001.txt"],
         "attributes": {"occupation": "researcher"}},
        {"id": "e1", "text": "Expo", "type": "event", "confidence": .93,
         "mentions": 1, "source_chunks": ["chunk_0001.txt"], "attributes": {}},
    ],
    "relationships": [
        {"head_id": "e0", "head": "Ada", "relation": "participated_in",
         "tail_id": "e1", "tail": "Expo", "confidence": .9,
         "source_chunks": ["chunk_0001.txt"]},
    ],
}


def stub_ontology(argv: list[str]) -> int:
    out = Path(argv[argv.index("--output") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entity_types": {"person": "Person", "event": "Event"},
        "relation_types": {"participated_in": "Attendance explicitly stated"},
        "attribute_schemas": {"person": {"occupation": ""}, "event": {}},
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    Path(argv[argv.index("--candidates-output") + 1]).write_text("{}", encoding="utf-8")
    Path(argv[argv.index("--coverage-report") + 1]).write_text("{}", encoding="utf-8")
    cache = Path(argv[argv.index("--cache-dir") + 1])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "test_checkpoint.json").write_text("{}", encoding="utf-8")
    return 0


def stub_kg(argv: list[str]) -> None:
    out = Path(argv[argv.index("--output") + 1])
    out.write_text(json.dumps(SAMPLE_GRAPH), encoding="utf-8")
    Path(argv[argv.index("--graphml") + 1]).write_text("<graphml/>", encoding="utf-8")
    cache = Path(argv[argv.index("--cache-dir") + 1])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "test_checkpoint.json").write_text("{}", encoding="utf-8")


def args_for(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        pdf=name, chunk_size=1000, chunk_overlap=200, device="cpu",
        review=False, no_refine_sparse=False, refresh=False, preview_only=False,
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.pdfs = self.root / "PDFs"
        self.data = self.root / "Data"
        self.pdfs.mkdir()
        self.data.mkdir()
        self.patches = [
            mock.patch.object(main1, "PDFS_DIR", self.pdfs),
            mock.patch.object(main1, "DATA_DIR", self.data),
            mock.patch.object(main2, "DATA_DIR", self.data),
            mock.patch("kg_construct.chunker.extract_text_from_pdf", return_value="Ada attended Expo."),
            mock.patch("kg_construct.chunker.chunk_text", return_value=["Ada attended Expo."]),
            mock.patch.object(main1, "generate_ontology", side_effect=stub_ontology),
            mock.patch.object(main1, "generate_knowledge_graph", side_effect=stub_kg),
            mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-placeholder"}),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_two_pdfs_create_separate_artifacts(self):
        (self.pdfs / "Alpha.pdf").write_bytes(b"pdf bytes A")
        (self.pdfs / "Beta.pdf").write_bytes(b"pdf bytes B")
        with redirect_stdout(io.StringIO()):
            alpha = main1.run_pipeline(args_for("Alpha.pdf"))
            beta = main1.run_pipeline(args_for("Beta"))
        self.assertNotEqual(alpha, beta)
        for folder in [alpha, beta]:
            for relative in (
                "source.pdf", "chunks/chunk_0001.txt",
                "ontology_hybrid_cache/test_checkpoint.json",
                "ontology.json", "ontology_candidates.json", "ontology_coverage.json",
                "kg_chunks/test_checkpoint.json", "knowledge_graph.json",
                "knowledge_graph.graphml", "pipeline_manifest.json",
            ):
                self.assertTrue((folder / relative).is_file(), str(folder / relative))
            status = json.loads((folder / "pipeline_manifest.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual((status["entities"], status["relationships"]), (2, 1))
        self.assertNotEqual((alpha / "source.pdf").read_bytes(),
                            (beta / "source.pdf").read_bytes())
        self.assertEqual(main2.select_graph("Alpha.pdf"), alpha / "knowledge_graph.json")
        self.assertEqual(main2.select_graph("Beta"), beta / "knowledge_graph.json")

    def test_rebuild_uses_archived_pdf_after_original_removed(self):
        original = self.pdfs / "Alpha.pdf"
        original.write_bytes(b"pdf bytes A")
        with redirect_stdout(io.StringIO()):
            first = main1.run_pipeline(args_for("Alpha.pdf"))
        original.unlink()
        with redirect_stdout(io.StringIO()):
            second = main1.run_pipeline(args_for("Alpha.pdf"))
        self.assertEqual(first, second)
        manifest = json.loads((second / "pipeline_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["original_filename"], "Alpha.pdf")

    def test_preview_only_cannot_be_selected_for_qa(self):
        (self.pdfs / "Alpha.pdf").write_bytes(b"pdf bytes A")
        args = args_for("Alpha.pdf")
        args.preview_only = True
        with redirect_stdout(io.StringIO()):
            folder = main1.run_pipeline(args)
        self.assertEqual(
            json.loads((folder / "pipeline_manifest.json").read_text())["status"],
            "preview_only",
        )
        self.assertFalse((folder / "knowledge_graph.json").exists())
        with self.assertRaises(FileNotFoundError):
            main2.select_graph("Alpha.pdf")

    def test_bad_pdf_name_does_not_escape_data(self):
        with self.assertRaises(ValueError):
            main2.select_graph("../another.pdf")
        with self.assertRaises(FileNotFoundError):
            main1.resolve_pdf("missing.pdf")

    def test_failed_rebuild_does_not_expose_stale_graph(self):
        (self.pdfs / "Alpha.pdf").write_bytes(b"pdf bytes A")
        with redirect_stdout(io.StringIO()):
            folder = main1.run_pipeline(args_for("Alpha"))
        with mock.patch.object(main1, "generate_ontology", side_effect=RuntimeError("test failure")):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "test failure"):
                main1.run_pipeline(args_for("Alpha"))
        self.assertEqual(json.loads((folder / "pipeline_manifest.json").read_text())["status"],
                         "failed")
        with self.assertRaises(RuntimeError):
            main2.select_graph("Alpha")


class QATests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.graph_path = Path(self.dir.name) / "knowledge_graph.json"
        self.graph_path.write_text(json.dumps(SAMPLE_GRAPH), encoding="utf-8")
        self.calls = []
        calls = self.calls

        def fake_create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="Ada attended Expo."))])

        self.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)))

    def test_incoming_relation_and_attributes(self):
        with redirect_stdout(io.StringIO()):
            retriever = KnowledgeGraphRetriever(str(self.graph_path))
            result = retriever.retrieve("Who attended Expo?")
        self.assertEqual(result["retrieval_mode"], "relation_traversal")
        self.assertTrue(any(path["end_entity"] == "Ada" for path in result["results"]))
        first_path = next(path for path in result["results"] if path["end_entity"] == "Ada")
        self.assertEqual(first_path["end_attributes"]["occupation"], "researcher")
        self.assertEqual(first_path["steps"][0]["stored_triple"]["source_chunks"],
                         ["chunk_0001.txt"])

    def test_generation_receives_only_retrieved_context(self):
        with redirect_stdout(io.StringIO()):
            qa = KnowledgeGraphQA(self.graph_path, client=self.client)
            answer, evidence = qa.generate_answer("Who attended Expo?")
        self.assertEqual(answer, "Ada attended Expo.")
        self.assertTrue(evidence["results"])
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Ada", self.calls[0]["messages"][1]["content"])
        self.assertIn("Expo", self.calls[0]["messages"][1]["content"])

    def test_raw_prints_before_api_call(self):
        stdout = io.StringIO()
        with mock.patch("builtins.input", side_effect=["raw on", "Who attended Expo?", "exit"]), \
             redirect_stdout(stdout):
            run_qa(self.graph_path, client=self.client)
        content = stdout.getvalue()
        self.assertIn("RAW KNOWLEDGE-GRAPH RETRIEVAL", content)
        self.assertLess(content.index("RAW KNOWLEDGE-GRAPH RETRIEVAL"),
                        content.index("[NVIDIA] Sending"))
        self.assertIn("Ada attended Expo.", content)

    def test_attribute_only_graph_is_answerable(self):
        graph = dict(SAMPLE_GRAPH)
        graph["relationships"] = []
        self.graph_path.write_text(json.dumps(graph), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            qa = KnowledgeGraphQA(self.graph_path, client=self.client)
            answer, evidence = qa.generate_answer("What is Ada's occupation?")
        self.assertEqual(evidence["retrieval_mode"], "neighborhood_fallback")
        self.assertEqual(evidence["results"][0]["attributes"]["occupation"], "researcher")
        self.assertEqual(answer, "Ada attended Expo.")  # Deterministic fake client.
        self.assertEqual(len(self.calls), 1)

    def test_missing_evidence_skips_api(self):
        with redirect_stdout(io.StringIO()):
            qa = KnowledgeGraphQA(self.graph_path, client=self.client)
            answer, evidence = qa.generate_answer("QZXW unmatched query?")
        self.assertEqual(answer, INSUFFICIENT)
        self.assertEqual(len(self.calls), 0)


if __name__ == "__main__":
    unittest.main()
