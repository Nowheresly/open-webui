import fitz  # PyMuPDF
import logging
import os
import tempfile
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List, Iterator

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class DoclingLoader(BaseLoader):

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.converter = self._initialize_converter()

    def _initialize_converter(self):
        from docling.datamodel.settings import settings
        settings.perf.elements_batch_size = 3
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.do_code_enrichment = True
        # ocr_options = RapidOcrOptions(force_full_page_ocr=True,use_cuda=True,language='de')
        ocr_options = EasyOcrOptions(force_full_page_ocr=True, use_gpu=True)

        pipeline_options.ocr_options = ocr_options

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    def _split_pdf_temporarily(self) -> List[dict]:

        doc = fitz.open(self.file_path)
        temp_files = []

        for page_num in range(len(doc)):
            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_doc.save(temp_path)
            new_doc.close()

            temp_files.append({"page": page_num + 1, "path": temp_path})

        return temp_files

    def load(self) -> List[Document]:

        temp_pdfs = self._split_pdf_temporarily()
        documents = []

        for pdf_info in temp_pdfs:
            pdf_path = pdf_info["path"]
            page_num = pdf_info["page"]
            page_label = pdf_info.get("page_label", str(page_num))

            doc = self.converter.convert(pdf_path).document
            md = doc.export_to_markdown()

            documents.append(
                Document(
                    page_content=md,
                    metadata={"page": page_num, "page_label": page_label},
                )
            )

            os.remove(pdf_path)

        return documents

    def lazy_load(self) -> Iterator[Document]:

        temp_pdfs = self._split_pdf_temporarily()

        for pdf_info in temp_pdfs:
            pdf_path = pdf_info["path"]
            page_num = pdf_info["page"]
            page_label = pdf_info.get("page_label", str(page_num))

            doc = self.converter.convert(pdf_path).document
            md = doc.export_to_markdown()

            yield Document(
                page_content=md,
                metadata={"page": page_num, "page_label": page_label},
            )

            os.remove(pdf_path)
