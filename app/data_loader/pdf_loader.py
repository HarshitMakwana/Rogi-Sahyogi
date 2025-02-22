import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Union
from enum import Enum

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.ocr_mac_model import OcrMacOptions
from docling.models.tesseract_ocr_model import TesseractOcrOptions
from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions

class OcrEngine(Enum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    TESSERACT_CLI = "tesseract_cli"
    OCR_MAC = "ocr_mac"
    NONE = "none"

class PdfParser:
    def __init__(
        self,
        ocr_engine: OcrEngine = OcrEngine.EASYOCR,
        languages: List[str] = ["en"],
        use_gpu: bool = True,
        num_threads: int = 4,
        do_table_structure: bool = True,
        cell_matching: bool = True,
        output_dir: str = "output"
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipeline_options = self._configure_pipeline(
            ocr_engine,
            languages,
            use_gpu,
            num_threads,
            do_table_structure,
            cell_matching
        )
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

    def _configure_pipeline(
        self,
        ocr_engine: OcrEngine,
        languages: List[str],
        use_gpu: bool,
        num_threads: int,
        do_table_structure: bool,
        cell_matching: bool
    ) -> PdfPipelineOptions:
        pipeline_options = PdfPipelineOptions()
        
        # Configure OCR settings
        pipeline_options.do_ocr = ocr_engine != OcrEngine.NONE
        if pipeline_options.do_ocr:
            if ocr_engine == OcrEngine.TESSERACT:
                pipeline_options.ocr_options = TesseractOcrOptions()
            elif ocr_engine == OcrEngine.TESSERACT_CLI:
                pipeline_options.ocr_options = TesseractCliOcrOptions()
            elif ocr_engine == OcrEngine.OCR_MAC:
                pipeline_options.ocr_options = OcrMacOptions()
            else:  # EasyOCR
                pipeline_options.ocr_options.lang = languages
                pipeline_options.ocr_options.use_gpu = use_gpu
        
        # Configure table structure settings
        pipeline_options.do_table_structure = do_table_structure
        if do_table_structure:
            pipeline_options.table_structure_options.do_cell_matching = cell_matching
        
        # Configure acceleration settings
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=AcceleratorDevice.AUTO
        )
        
        return pipeline_options

    def parse_pdf(
        self,
        input_path: Union[str, Path],
        export_formats: List[str] = ["json", "txt", "md", "doctags"]
    ) -> Dict:
        """
        Parse PDF and export in specified formats.
        
        Args:
            input_path: Path to input PDF file
            export_formats: List of export formats (json, txt, md, doctags)
            
        Returns:
            Dictionary containing parsing metadata and export paths
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Convert document
        start_time = time.time()
        conv_result = self.doc_converter.convert(input_path)
        processing_time = time.time() - start_time
        
        self.logger.info(f"Document converted in {processing_time:.2f} seconds.")

        # Export results
        doc_filename = conv_result.input.file.stem
        export_paths = {}
        
        export_methods = {
            "json": (conv_result.document.export_to_dict, "json"),
            "txt": (conv_result.document.export_to_text, "txt"),
            "md": (conv_result.document.export_to_markdown, "md"),
            "doctags": (conv_result.document.export_to_document_tokens, "doctags")
        }
        
        for format_name in export_formats:
            if format_name in export_methods:
                export_func, extension = export_methods[format_name]
                output_path = self.output_dir / f"{doc_filename}.{extension}"
                
                with output_path.open("w", encoding="utf-8") as fp:
                    content = export_func()
                    if isinstance(content, dict):
                        json.dump(content, fp, ensure_ascii=False, indent=2)
                    else:
                        fp.write(content)
                        
                export_paths[format_name] = str(output_path)

        return {
            "processing_time": processing_time,
            "export_paths": export_paths,
            "document_name": doc_filename,
            "ocr_enabled": self.pipeline_options.do_ocr,
            "table_structure_enabled": self.pipeline_options.do_table_structure,
            "content": content,
        }