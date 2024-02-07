import pypdf
from pdfminder.pdfparser import PDFParser
from pdfminder.pdfdocument import PDFDocument

from typing import Optional
from io import StringIO

# for extracting tables
# import tabula
# import camelot


from typing import Dict, List, Type, Optional, Union
from pydantic import BaseModel, Json

TEST_PDF_PATH = "../tests/test_data/testPaper1.pdf"


class PDFExtractor(BaseModel):
    """
    To extract information from PDF files, specifically research papers.

    It aims to extract structured information including text, metadata, and tables from PDF documents,
    returns a json list for .

    This extractor is primarily designed to extract information from PDF files, namely research papers.

    Assuming the structure of the PDF research paper is
    Title, 
    Abstract, 
    Introduction, 
    Methodology, 
    Results, 
    Discussion, 
    Conclusion, 
    References, 
    and Appendices

    """

    chunk_size: Optional[int] = 1000

    def extract_text(self, pdf_path: str) -> str:
        """
        Extracts plain text from the given PDF file using pdfminer.

        Args:
        pdf_path (str): Path to the PDF file.

        Returns:
        str: Extracted plain text.
        """

        with open(pdf_path, "rb") as fp:
            pdf = pypdf.PdfReader(fp)
            num_pages = len(pdf.pages)
            objs = {}

            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                page_label = pdf.page_labels[page]

                metadata = {"page_label": page_label, "file_name": fp.name}

                objs[page] = {"text": page_text, "metadata": metadata}

            return objs

    def extract_pdf_metadata(self, pdf_file: str) -> Dict:
        """
        Extracts metadata and structured text (e.g., sections, titles) from the PDF.

        Args:
        pdf_path (str): Path to the PDF file.

        Returns:
        Dict: Extracted metadata and structured text information.
        """
        with open(pdf_file, 'rb') as file:
            parser = PDFParser(file)
            doc = PDFDocument(parser)
            # stored in the 'info' attribute of the PDFDocument object.
            metadata = doc.info

            if metadata:
                # 'metadata' is a list of dictionaries. Usually only one dictionary in the list.
                metadata_dict = {key.decode('utf8'): value.decode(
                    'utf8') for key, value in metadata[0].items()}
                return metadata_dict
            else:
                return "No metadata found."

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        Extracts tables from the PDF using Tabula and Camelot, prioritizing extraction quality.

        Args:
        pdf_path (str): Path to the PDF file.

        Returns:
        List[Dict]: Extracted tables as lists of dictionaries.
        """
        # TODO for table extraction logic either using Tabula and Camelot
        return [{"table": "Extracted table data"}]

    def format_json(self, text: str, metadata: Dict, tables: List[Dict]) -> Json:
        """
        Formats the extracted information into a predefined JSON structure.

        Args:
        text (str): Extracted plain text.
        metadata (Dict): Extracted metadata and structured text information.
        tables (List[Dict]): Extracted tables as lists of dictionaries.

        Returns:
        Json: Formatted JSON structure.
        """
        # TODO format the extracted information into a predefined JSON structure

        return {"Some": "JSON"}


extractor = PDFExtractor()
extractor.extract_text(TEST_PDF_PATH)