import base64
import io

from PIL import Image
import pandas as pd
import pdfplumber
import torch
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import fitz  # PyMuPDF

from transformers import AutoImageProcessor, TableTransformerForObjectDetection


TEST_PDF_PATH = "./tests/test_data/testPaper3.pdf"


class PDFLoader(BaseModel):
    """
    Load and extract information from PDF files, specifically research papers.

    It aims to extract structured information including text, metadata, and tables from PDF documents,
    returns a json list for .

    This extractor is primarily designed to extract information from PDF files, namely research papers.
    """

    chunk_size: Optional[int] = 1000
    pdf_path: str = Field(...,
                          description="Path to the PDF file to be processed.")

    def extract_text_pymupdf(self) -> List[Dict]:
        """
        Extracts text from each page of the PDF file using PyMuPDF (fitz).

        Returns:
            List[Dict]: Text extracted from each page along with page numbers.
        """

        doc = fitz.open(TEST_PDF_PATH)
        data = {}

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            page_data = []

            for block in blocks:
                if "lines" in block:  # Ensure it's a text block
                    paragraph_text = ""
                    bbox = block["bbox"]
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Aggregate text
                            paragraph_text += span["text"] + " "
                    paragraph_text = paragraph_text.strip()
                    page_data.append(
                        {"text": paragraph_text, "bbox": list(bbox)})

            data[f"Page_{page_num + 1}"] = page_data

        doc.close()
        print(data)

        return data

    def extract_metadata(self) -> Dict:
        """
        Extracts metadata from the PDF file.

        Returns:
            Dict: Extracted metadata.
        """
        doc = fitz.open(self.pdf_path)
        metadata = doc.metadata
        doc.close()

        return metadata

    def extract_tables(self) -> List[Dict]:
        """
        Extracts tables from the PDF using Camelot.

        Returns:
            List[Dict]: Extracted tables as lists of dictionaries, with page numbers.
        """
        tables_extract = {}
        all_tables = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    # convert tables into Pandas DataFrames for easier manipulation assuming first row is the header
                    df = pd.DataFrame(table[1:], columns=table[0])
                    all_tables.append(df)

        doc = fitz.open(self.pdf_path)
        for i in range(len(doc)):
            for img in doc.get_page_images(i):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                img_buffer = io.BytesIO()

                if pix.n < 5:  # GRAY or RGB
                    img_buffer.write(pix.tobytes("png"))
                else:  # CMYK: convert to RGB first
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    img_buffer.write(pix1.tobytes("png"))
                    pix1 = None

                pix = None

                img_buffer.seek(0)

                base64_image = base64.b64encode(
                    img_buffer.read()).decode("utf-8")
                base64_image_uri = f"data:image/jpeg;base64,{base64_image}"

                if "base64," in base64_image_uri:
                    header, base64_data = base64_image_uri.split("base64,")
                else:
                    base64_data = base64_image_uri

                image_data = base64.b64decode(base64_data)

                image_stream = io.BytesIO(image_data)
                image = Image.open(image_stream).convert("RGB")

                image_processor = AutoImageProcessor.from_pretrained(
                    "microsoft/table-transformer-detection")
                model = TableTransformerForObjectDetection.from_pretrained(
                    "microsoft/table-transformer-detection")

                inputs = image_processor(images=image, return_tensors="pt")
                outputs = model(**inputs)

                # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
                target_sizes = torch.tensor([image.size[::-1]])
                results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
                    0
                ]

                tableBBox = {}

                for label, box in zip(results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    tableBBox[model.config.id2label[label.item()]] = box

                tables_extract["tables"] = all_tables
                tables_extract["bounding_boxes"] = tableBBox

        return tables_extract

    def format_output(self, text_data: List[Dict], metadata: Dict, tables: List[Dict]) -> List[Dict]:
        """
        Formats the extracted information into a structured JSON-like format.

        Args:
            text_data (List[Dict]): Text extracted from each page.
            metadata (Dict): Extracted metadata.
            tables (List[Dict]): Extracted tables data.

        Returns:
            List[Dict]: Structured data combining text, metadata, and tables.
        """
        structured_data = {
            "text_data": text_data,
            "metadata": metadata,
            "tables": tables
        }
        return structured_data

    def load(self):
        """Extract text, metadata, and tables from the PDF and format the output."""
        text_data = self.extract_text_pymupdf()
        metadata = self.extract_metadata()
        tables = self.extract_tables()
        return self.format_output(text_data, metadata, tables)
