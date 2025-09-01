# main.py Annotated (Lines 1–200)

This is a literal, line-by-line explanation of `main.py` for lines 1–200: what each line generally means and why it is used in this codebase.

---

1. `import os` — Standard library for filesystem paths and environment variables. Needed throughout for reading files, building paths, and env access.
2. `import time` — Standard library for measuring durations or timestamps. Used in timing operations/logging.
3. `import asyncio` — Python async framework. Required because FastAPI handlers and some I/O are async.
4. `from typing import List, Optional, Dict, Any` — Type hints for readability and validation. We use them in Pydantic models and helper signatures.
5. `from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form` — FastAPI primitives for app creation, file upload handling, exceptions, and request/form parsing.
6. `from fastapi.middleware.cors import CORSMiddleware` — Middleware to permit cross-origin requests; important for local dev UI or different hosts.
7. `from fastapi.responses import JSONResponse, FileResponse` — Response helpers to return JSON or files from endpoints.
8. `from pydantic import BaseModel` — Pydantic models used for request/response schemas with validation.
9. `import autogen` — AutoGen library used to configure conversational agents that transform retrieved facts into final answers.
10. `from dotenv import load_dotenv` — Loads .env so keys (e.g., GOOGLE_API_KEY) are available without setting system-wide.
11. `import uvicorn` — ASGI server to run FastAPI during development or simple deployments.
12. `import PyPDF2` — PDF text extraction when embedded text exists (faster than OCR), used during quick extraction.
13. `import re` — Regular expressions for tokenization and pattern matching in extraction and filtering.
14. `import traceback` — For detailed stack traces when logging exceptions.
15. (blank line) — Readability separator.
16. `# Handle Gemini client errors for retries` — Comment documenting the following try/except pattern purpose.
17. `try:` — Attempt import of Gemini client error class.
18. `from google.genai.errors import ClientError as GeminiClientError` — Use the official error type when package is installed.
19. `except Exception:` — Fallback if the package isn’t present in the environment.
20. `class GeminiClientError(Exception):` — Define a local stub error type for consistent error handling.
21. `pass` — Placeholder body of the stub error class.
22. (blank line) — Separator.
23. `# Updated LangChain imports` — Comment grouping the LangChain import block.
24. `from langchain_community.document_loaders import PyPDFLoader` — Loader for PDFs into LangChain Document objects for RAG.
25. `from langchain_community.vectorstores import Chroma` — Vector store backend used to persist/retrieve embeddings.
26. `from langchain_google_genai import GoogleGenerativeAIEmbeddings` — Embedding model wrapper using Google’s embeddings.
27. `from langchain.text_splitter import RecursiveCharacterTextSplitter` — Splits text into chunks optimized for embedding and retrieval.
28. `import requests` — HTTP client for fetching external web content.
29. `from bs4 import BeautifulSoup` — HTML parser to extract text from web pages.
30. `from urllib.parse import urlparse` — Utility to parse URLs and domains; used for source labeling and domain checks.
31. `import random` — Randomize URL lists to vary probing order for external info.
32. `from pdf2image import convert_from_path` — Renders PDF pages to images for OCR fallback when PDFs are scanned.
33. `import pytesseract` — OCR engine binding to extract text from rendered images.
34. `from PIL import Image` — Image manipulation used with OCR.
35. `import tempfile` — Temporary directories/files for OCR intermediate artifacts.
36. `import gc` — Manual garbage collection hints to free large objects during heavy processing.
37. (blank line) — Separator.
38. `try:` — Try importing Document class from modern LangChain API.
39. `from langchain.schema import Document` — Preferred import path.
40. `except Exception:` — Fallback if API differs by installed version.
41. `from langchain_core.documents import Document` — Alternative path for compatibility.
42. (blank line) — Separator.
43. `# Load environment variables` — Comment about reading .env.
44. `load_dotenv()` — Loads variables from .env to process environment.
45. `GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")` — Read API key with fallback name; required for embeddings/LLM.
46. (blank line) — Separator.
47. `# FastAPI app` — Comment for the app creation line.
48. `app = FastAPI(title="AI Sales Consultant API", version="1.0.0")` — Create the FastAPI application object.
49. (blank line) — Separator.
50. `# CORS middleware` — Comment.
51. `app.add_middleware(` — Begin adding CORS middleware.
52. `    CORSMiddleware,` — Specify the middleware class.
53. `    allow_origins=["*"],` — Allow any origin (simplifies dev; can be restricted later).
54. `    allow_credentials=True,` — Allow cookies/credentials.
55. `    allow_methods=["*"],` — Allow all HTTP verbs.
56. `    allow_headers=["*"],` — Allow all headers.
57. `)` — Finish middleware setup.
58. (blank line) — Separator.
59. `# Create directories` — Comment for filesystem setup.
60. `os.makedirs("static", exist_ok=True)` — Ensure static folder exists for UI assets.
61. `os.makedirs("data", exist_ok=True)` — Ensure data folder exists for uploaded PDFs.
62. (blank line) — Separator.
63. `# Global state (in production, use Redis or database)` — Comment: this dict is a simple runtime store.
64. `app_state = {` — Start global state dict.
65. `    "retriever": None,` — Placeholder for the active retriever from Chroma.
66. `    "product_name": "",` — Current product display name.
67. `    "product_slug": "",` — Slugified name used for collection names and URLs.
68. `    "product_brand": "",` — Detected/confirmed brand.
69. `    "product_model": "",` — Detected/confirmed model.
70. `    "product_category": "",` — Detected/confirmed category.
71. `    "user_location": "Chennai, Tamil Nadu, India",` — Default location to influence dealer/price pages.
72. `    "disable_external": False,` — Toggle to disable external enrichment.
73. `    "source_chunk_counts": None,` — Diagnostics about how many chunks were stored per source.
74. `    "product_url": "",` — Guessed official product URL.
75. `    "product_info": {},` — Additional inferred info (e.g., features) for UI.
76. `    "ocr_settings": {` — OCR configuration nested dict.
77. `        "enable_ocr": True,` — Whether to try OCR when no embedded text.
78. `        "ocr_pages": 8,` — OCR up to N pages for speed/cost control.
79. `        "poppler_path": "",` — Optional path to Poppler binaries.
80. `        "tesseract_cmd": ""` — Optional path to Tesseract executable.
81. `    }` — End ocr_settings.
82. `}` — End app_state dict.
83. (blank line) — Separator.
84. `# Enhanced Product Extractor Class` — Comment for the following class.
85. `class ProductExtractor:` — Class encapsulating brand/model/category inference.
86. `    def __init__(self):` — Constructor.
87. `        self.category_patterns = {` — Define category-specific brand lists, indicators, and regexes.
88. `            'automotive': {` — Automotive section.
89. `                'brands': [...]` — Known makes used for filename token matching.
90. `                'indicators': [...]` — Words signaling automotive context for scoring.
91. `                'pattern': re.compile(...)` — Regex to find "Brand Model" patterns in content.
92. `            },` — End automotive.
93. `            'pharmaceutical': {` — Pharma section.
94. `                'brands': [...]` — Known pharma brands.
95. `                'indicators': [...]` — Words indicating drug/approval context.
96. `                'pattern': re.compile(...)` — Regex for pharma brand + term patterns.
97. `            },` — End pharma.
98. `            'electronics': {` — Electronics section.
99. `                'brands': [...]` — Electronics brands.
100. `                'indicators': [...]` — Terms indicating consumer electronics.
101. `                'pattern': re.compile(...)` — Regex for electronics brand + model.
102. `            }` — End electronics.
103. `        }` — End category_patterns.
104. (blank line) — Separator.
105. `    def detect_category(self, filename: str, content: str) -> str:` — Public method; infers category from signals.
106. `        """Auto-detect category from filename and content"""` — Docstring.
107. `        text = f"{filename} {content[:1000]}".lower()` — Combine filename and first 1000 chars of content; lowercase for matching.
108. `        ` — Blank continuation line for readability.
109. `        scores = {}` — Initialize category score map.
110. `        for category, data in self.category_patterns.items():` — Iterate categories.
111. `            score = sum(1 for indicator in data['indicators'] if indicator in text)` — Count indicator hits.
112. `            scores[category] = score` — Save category score.
113. `        ` — Blank line.
114. `        return max(scores, key=scores.get) if scores else 'other'` — Pick category with max hits; fallback to 'other'.
115. (blank line) — Separator.
116. `    def extract_parameters(self, filename: str, content: str, detected_category: str = None) -> Dict:` — Orchestrates extraction from filename+content.
117. `        """Extract all parameters in one go"""` — Docstring.
118. `        category = detected_category or self.detect_category(filename, content)` — Use given category or detect it.
119. `        ` — Blank line.
120. `        result = {` — Initialize result structure with defaults.
121. `            'category': category,` — Selected category.
122. `            'brand': None,` — Brand placeholder.
123. `            'model': None,` — Model placeholder.
124. `            'product': None,` — Product display name.
125. `            'confidence': 0.0,` — Confidence score.
126. `            'extraction_source': []` — Track whether filename or content provided data.
127. `        }` — Close result.
128. `        ` — Blank.
129. `        # Extract from filename` — Comment.
130. `        filename_brand, filename_model = self._extract_from_filename(filename, category)` — Try filename heuristics first.
131. `        if filename_brand:` — If brand found in filename...
132. `            result['brand'] = filename_brand` — Save it.
133. `            result['extraction_source'].append('filename')` — Note source.
134. `        if filename_model:` — If model found in filename...
135. `            result['model'] = filename_model` — Save it.
136. `            if 'filename' not in result['extraction_source']:` — Ensure source noted.
137. `                result['extraction_source'].append('filename')` — Note filename as source.
138. `        ` — Blank.
139. `        # Extract from content (first 2000 chars for speed)` — Comment.
140. `        content_brand, content_model = self._extract_from_content(content[:2000], category)` — Try regexes on content.
141. `        if content_brand and not result['brand']:` — Prefer filename brand unless missing.
142. `            result['brand'] = content_brand` — Save content-derived brand.
143. `            result['extraction_source'].append('content')` — Note content as source.
144. `        if content_model and not result['model']:` — Similarly for model.
145. `            result['model'] = content_model` — Save content-derived model.
146. `            if 'content' not in result['extraction_source']:` — Ensure content source recorded.
147. `                result['extraction_source'].append('content')` — Note content.
148. `        ` — Blank.
149. `        # Build product name` — Comment.
150. `        if result['brand'] and result['model']:` — If both brand and model...
151. `            result['product'] = f"{result['brand']} {result['model']}"` — Combine as "Brand Model".
152. `            result['confidence'] = 0.8` — Higher confidence.
153. `        elif result['brand']:` — Only brand known...
154. `            result['product'] = result['brand']` — Use brand.
155. `            result['confidence'] = 0.5` — Medium confidence.
156. `        else:` — Neither brand nor model found.
157. `            # Fallback to cleaned filename` — Comment.
158. `            result['product'] = self._clean_filename(filename)` — Use cleaned filename as product.
159. `            result['confidence'] = 0.3` — Lower confidence.
160. `        ` — Blank.
161. `        return result` — Return extraction results.
162. (blank line) — Separator.
163. `    def _extract_from_filename(self, filename: str, category: str) -> tuple:` — Private helper for filename parsing.
164. `        """Extract brand and model from filename"""` — Docstring.
165. `        base = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')` — Normalize filename for tokenization.
166. `        tokens = [t for t in base.split() if len(t) > 2]  # Filter short tokens` — Tokenize, ignore very short words.
167. `        ` — Blank.
168. `        brand, model = None, None` — Initialize outputs.
169. `        ` — Blank.
170. `        if category in self.category_patterns:` — Ensure category has hints configured.
171. `            brands = self.category_patterns[category]['brands']` — Get brand list for the category.
172. `            for i, token in enumerate(tokens):` — Iterate tokens to find brand.
173. `                if any(brand_name.lower() in token.lower() for brand_name in brands):` — Case-insensitive brand check.
174. `                    brand = next(b for b in brands if b.lower() in token.lower())` — Find canonical brand string.
175. `                    # Next significant token is likely model` — Heuristic: model follows brand.
176. `                    if i + 1 < len(tokens):` — If there is a next token...
177. `                        model = tokens[i + 1].title()` — Title-case as a presentable model.
178. `                    break` — Stop at first brand occurrence.
179. `        ` — Blank.
180. `        return brand, model` — Return inferred values.
181. (blank line) — Separator.
182. `    def _extract_from_content(self, content: str, category: str) -> tuple:` — Private helper scanning content regex.
183. `        """Extract brand and model from content"""` — Docstring.
184. `        if category in self.category_patterns:` — Check category configuration.
185. `            pattern = self.category_patterns[category]['pattern']` — Regex for category.
186. `            match = pattern.search(content)` — Try to find brand+model.
187. `            if match:` — If matched...
188. `                return match.group(1).title(), match.group(2).title()` — Return title-cased brand and model.
189. `        ` — Blank.
190. `        return None, None` — No match found.
191. (blank line) — Separator.
192. `    def _clean_filename(self, filename: str) -> str:` — Fallback name derivation.
193. `        """Clean filename as fallback product name"""` — Docstring.
194. `        base = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')` — Normalize.
195. `        # Remove common words` — Comment.
196. `        words = base.split()` — Tokenize.
197. `        filtered = [w for w in words if w.lower() not in ['brochure', 'manual', 'guide', '2024', '2023', 'new', 'reinforced', 'safety']]` — Remove generic words.
198. `        return ' '.join(filtered[:3])  # Take first 3 significant words` — Compose short presentable name.
199. (blank line) — Separator.
200. `# Initialize extractor` — Start of initialization section.