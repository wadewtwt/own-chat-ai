import os
import pdfplumber
import pandas as pd
from docx import Document
from typing import List, Dict

# 文档处理器类，用于提取不同格式文档的文本内容（暂未完善）
class DocumentProcessor:
    def __init__(self):
        self.text_extractors = {
            ".txt": self._extract_text_from_txt,
            ".pdf": self._extract_text_from_pdf,
            ".docx": self._extract_text_from_docx,
            ".xlsx": self._extract_text_from_xlsx,
            ".csv": self._extract_text_from_csv,
        }
    
    def process_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.text_extractors:
            return self.text_extractors[ext](file_path)
        return f"不支持的文件格式: {ext}"
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _extract_text_from_xlsx(self, file_path: str) -> str:
        dfs = pd.ExcelFile(file_path).parse_all_sheets()
        text = ""
        for sheet_name, df in dfs.items():
            text += f"工作表: {sheet_name}\n"
            text += df.to_csv(sep='\t', na_rep='nan') + "\n\n"
        return text
    
    def _extract_text_from_csv(self, file_path: str) -> str:
        df = pd.read_csv(file_path)
        return df.to_csv(sep='\t', na_rep='nan')