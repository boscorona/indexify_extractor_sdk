import pdfminer
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

# TODO test alternate approach for extrating text with .highlevel module
# from pdfminer.high_level import extract_text

from typing import Optional
from io import StringIO

# for extracting tables
import tabula
import camelot


from typing import Dict, List, Type, Optional, Union
from pydantic import BaseModel, Json


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
        # TODO experiment with different papers for complete text extraction.
        # alternate for extracting
        # with open('report.pdf', 'rb') as f:
        #     text = extract_text(f)
        # return text

        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(pdf_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()
        return text

    def extract_metadata(self, pdf_path: str) -> Dict:
        """
        Extracts metadata and structured text (e.g., sections, titles) from the PDF.

        Args:
        pdf_path (str): Path to the PDF file.

        Returns:
        Dict: Extracted metadata and structured text information.
        """

        return {"metadata": "Extracted metadata"}

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
