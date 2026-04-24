import shutil
import unittest
import uuid
import zipfile
from pathlib import Path

from src.ingest import ingest_file_paths, load_documents_from_path

WORKSPACE_TEMP_DIR = Path(__file__).resolve().parent / ".tmp_test_data"


def make_case_dir() -> Path:
    WORKSPACE_TEMP_DIR.mkdir(exist_ok=True)
    case_dir = WORKSPACE_TEMP_DIR / f"case_{uuid.uuid4().hex}"
    case_dir.mkdir()
    return case_dir


def write_docx(path: Path, text: str) -> None:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>
  </w:body>
</w:document>
"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)


class IngestTests(unittest.TestCase):
    def test_load_documents_preserves_source_metadata(self):
        case_dir = make_case_dir()
        try:
            text_path = case_dir / "notes.txt"
            text_path.write_text("alpha beta gamma", encoding="utf-8")

            documents = load_documents_from_path(text_path)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["source"], "notes.txt")
            self.assertEqual(documents[0].metadata["file_type"], "txt")
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_ingest_multiple_formats(self):
        case_dir = make_case_dir()
        try:
            txt_path = case_dir / "notes.txt"
            csv_path = case_dir / "table.csv"
            docx_path = case_dir / "brief.docx"

            txt_path.write_text("LangChain loaders work on text files.", encoding="utf-8")
            csv_path.write_text("topic,summary\nrag,Retrieval augmented generation\n", encoding="utf-8")
            write_docx(docx_path, "Word document support is enabled.")

            result = ingest_file_paths([txt_path, csv_path, docx_path])

            self.assertEqual(len(result.files), 3)
            self.assertTrue(all(item.succeeded for item in result.files))
            self.assertGreaterEqual(len(result.chunks), 3)
            self.assertEqual({chunk.metadata["source"] for chunk in result.chunks}, {"notes.txt", "table.csv", "brief.docx"})
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
