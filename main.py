import os
import time
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import autogen
from dotenv import load_dotenv
import uvicorn
import PyPDF2
import re
import traceback
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_agent")

# Handle Gemini client errors for retries
try:
    from google.genai.errors import ClientError as GeminiClientError
except Exception:
    class GeminiClientError(Exception):
        pass

# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import random
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import tempfile
import gc

try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# FastAPI app
app = FastAPI(title="AI Sales Consultant API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Global state (in production, use Redis or database)
app_state = {
    "retriever": None,
    "product_name": "",
    "product_slug": "",
    "product_brand": "",
    "product_model": "",
    "product_category": "",
    "user_location": "Chennai, Tamil Nadu, India",
    "disable_external": False,
    "source_chunk_counts": None,
    "product_url": "",
    "product_info": {},
    "ocr_settings": {
        "enable_ocr": True,
        "ocr_pages": 8,
        "poppler_path": "",
        "tesseract_cmd": ""
    }
}

# Base Content Extractor Class
class BaseContentExtractor:
    def __init__(self, category_name, keywords, spec_patterns):
        self.category_name = category_name
        self.keywords = keywords
        self.spec_patterns = spec_patterns
    
    def extract_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract category-specific content from HTML"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator=" ", strip=True)
            
            # Extract structured data
            extracted_data = {
                'url': url,
                'category': self.category_name,
                'title': soup.title.get_text(strip=True) if soup.title else '',
                'specifications': {},
                'features': [],
                'relevant_content': [],
                'price_info': [],
                'reviews': []
            }
            
            # Extract specifications using patterns
            for spec_type, pattern in self.spec_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    extracted_data['specifications'][spec_type] = matches[0].strip()
            
            # Extract sentences containing category keywords
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20 or len(sentence) > 300:
                    continue
                    
                # Check if sentence contains category keywords
                sentence_lower = sentence.lower()
                keyword_count = sum(1 for keyword in self.keywords if keyword in sentence_lower)
                
                if keyword_count >= 2:  # Sentence must contain at least 2 keywords
                    extracted_data['relevant_content'].append(sentence)
            
            # Extract feature lists
            feature_elements = soup.find_all(['li', 'div'], class_=re.compile(r'feature|spec|highlight|benefit', re.I))
            for element in feature_elements:
                feature_text = element.get_text(strip=True)
                if 10 <= len(feature_text) <= 150:
                    extracted_data['features'].append(feature_text)
            
            # Extract price information (enhanced for Indian pricing)
            price_elements = soup.find_all(text=re.compile(r'₹|price|cost|\$|INR|starting|from|rs\.?|rupees?|lakh|crore|ex-showroom', re.I))
            for price_text in price_elements[:8]:  # Increased limit for more price mentions
                price_clean = price_text.strip()
                if len(price_clean) > 5 and any(char.isdigit() for char in price_clean):
                    extracted_data['price_info'].append(price_clean)
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting {self.category_name} content from {url}: {str(e)}")
            return {'url': url, 'category': self.category_name, 'error': str(e)}
    
    def format_for_knowledge_base(self, extracted_data: Dict[str, Any]) -> str:
        """Format extracted data for knowledge base ingestion"""
        if 'error' in extracted_data:
            return f"Error processing {extracted_data['url']}: {extracted_data['error']}"
        
        formatted_content = []
        
        # Add title and category
        if extracted_data.get('title'):
            formatted_content.append(f"Title: {extracted_data['title']}")
        formatted_content.append(f"Category: {extracted_data.get('category', 'Unknown')}")
        
        # Add specifications
        if extracted_data.get('specifications'):
            formatted_content.append("\nSpecifications:")
            for spec_type, spec_value in extracted_data['specifications'].items():
                formatted_content.append(f"- {spec_type.replace('_', ' ').title()}: {spec_value}")
        
        # Add features
        if extracted_data.get('features'):
            formatted_content.append("\nKey Features:")
            for feature in extracted_data['features'][:12]:  # Limit to top 12 features
                formatted_content.append(f"- {feature}")
        
        # Add price information
        if extracted_data.get('price_info'):
            formatted_content.append("\nPricing Information:")
            for price in extracted_data['price_info'][:5]:
                formatted_content.append(f"- {price}")
        
        # Add relevant content
        if extracted_data.get('relevant_content'):
            formatted_content.append("\nDetailed Information:")
            for content in extracted_data['relevant_content'][:15]:  # Limit to top 15 sentences
                formatted_content.append(f"- {content}")
        
        formatted_content.append(f"\nSource: {extracted_data['url']}")
        
        return "\n".join(formatted_content)

# Automotive Content Extractor Class
class AutomotiveContentExtractor(BaseContentExtractor):
    def __init__(self):
        automotive_keywords = [
            'engine', 'fuel', 'mileage', 'horsepower', 'torque', 'transmission', 
            'displacement', 'cylinders', 'turbo', 'hybrid', 'electric', 'petrol', 
            'diesel', 'automatic', 'manual', 'cvt', 'safety', 'airbags', 'abs',
            'features', 'specifications', 'dimensions', 'wheelbase', 'ground clearance',
            'boot space', 'seating', 'infotainment', 'connectivity', 'warranty',
            'service', 'maintenance', 'price', 'variant', 'model', 'trim'
        ]
        
        automotive_patterns = {
            'engine': re.compile(r'(?:engine|motor)[\s:]*([^.\n]{10,100})', re.I),
            'fuel_type': re.compile(r'(?:fuel|petrol|diesel|electric|hybrid)[\s:]*([^.\n]{5,50})', re.I),
            'mileage': re.compile(r'(?:mileage|fuel economy|mpg|kmpl)[\s:]*([^.\n]{5,50})', re.I),
            'transmission': re.compile(r'(?:transmission|gearbox)[\s:]*([^.\n]{5,50})', re.I),
            'safety': re.compile(r'(?:safety|airbags|abs|esp)[\s:]*([^.\n]{10,100})', re.I),
            'price': re.compile(r'(?:price|cost|starts at|from|₹|rs\.?|rupees?|lakh|crore|inr)[\s:]*([^.\n]{5,80})', re.I)
        }
        
        super().__init__("automotive", automotive_keywords, automotive_patterns)

# Mobile Content Extractor Class
class MobileContentExtractor(BaseContentExtractor):
    def __init__(self):
        mobile_keywords = [
            'display', 'screen', 'camera', 'battery', 'processor', 'chipset', 'ram', 'storage',
            'android', 'ios', 'operating system', 'megapixel', 'mp', 'mah', 'charging',
            'wireless', '5g', '4g', 'wifi', 'bluetooth', 'fingerprint', 'face unlock',
            'dual sim', 'waterproof', 'gorilla glass', 'amoled', 'oled', 'lcd',
            'refresh rate', 'nits', 'ppi', 'zoom', 'portrait', 'night mode'
        ]
        
        mobile_patterns = {
            'display': re.compile(r'(?:display|screen)[\s:]*([^.\n]{10,80})', re.I),
            'camera': re.compile(r'(?:camera|mp|megapixel)[\s:]*([^.\n]{10,80})', re.I),
            'battery': re.compile(r'(?:battery|mah)[\s:]*([^.\n]{5,50})', re.I),
            'processor': re.compile(r'(?:processor|chipset|cpu)[\s:]*([^.\n]{10,80})', re.I),
            'ram': re.compile(r'(?:ram|memory)[\s:]*([^.\n]{5,30})', re.I),
            'storage': re.compile(r'(?:storage|gb|tb)[\s:]*([^.\n]{5,30})', re.I),
            'os': re.compile(r'(?:android|ios|operating system)[\s:]*([^.\n]{5,50})', re.I),
            'price': re.compile(r'(?:price|cost|₹|\$)[\s:]*([^.\n]{5,50})', re.I)
        }
        
        super().__init__("mobile", mobile_keywords, mobile_patterns)

# Laptop Content Extractor Class
class LaptopContentExtractor(BaseContentExtractor):
    def __init__(self):
        laptop_keywords = [
            'processor', 'cpu', 'intel', 'amd', 'ryzen', 'core', 'ram', 'memory', 'storage',
            'ssd', 'hdd', 'graphics', 'gpu', 'nvidia', 'radeon', 'display', 'screen',
            'resolution', 'keyboard', 'trackpad', 'battery', 'ports', 'usb', 'hdmi',
            'wifi', 'bluetooth', 'webcam', 'speakers', 'weight', 'thickness', 'gaming',
            'business', 'ultrabook', 'convertible', 'touchscreen', 'backlit'
        ]
        
        laptop_patterns = {
            'processor': re.compile(r'(?:processor|cpu|intel|amd|ryzen)[\s:]*([^.\n]{10,80})', re.I),
            'ram': re.compile(r'(?:ram|memory)[\s:]*([^.\n]{5,30})', re.I),
            'storage': re.compile(r'(?:storage|ssd|hdd)[\s:]*([^.\n]{5,50})', re.I),
            'graphics': re.compile(r'(?:graphics|gpu|nvidia|radeon)[\s:]*([^.\n]{10,80})', re.I),
            'display': re.compile(r'(?:display|screen|resolution)[\s:]*([^.\n]{10,80})', re.I),
            'battery': re.compile(r'(?:battery|hours)[\s:]*([^.\n]{5,50})', re.I),
            'weight': re.compile(r'(?:weight|kg|pounds)[\s:]*([^.\n]{5,30})', re.I),
            'price': re.compile(r'(?:price|cost|₹|\$)[\s:]*([^.\n]{5,50})', re.I)
        }
        
        super().__init__("laptop", laptop_keywords, laptop_patterns)

# Automotive Accessories Content Extractor Class
class AutomotiveAccessoriesContentExtractor(BaseContentExtractor):
    def __init__(self):
        accessories_keywords = [
            'car', 'vehicle', 'accessory', 'dashboard', 'seat cover', 'floor mat',
            'steering', 'wheel', 'mirror', 'charger', 'mount', 'holder', 'organizer',
            'sunshade', 'visor', 'led', 'light', 'bulb', 'horn', 'speaker', 'amplifier',
            'navigation', 'gps', 'camera', 'sensor', 'alarm', 'security', 'lock',
            'tire', 'wheel', 'rim', 'brake', 'oil', 'filter', 'battery', 'toolkit'
        ]
        
        accessories_patterns = {
            'compatibility': re.compile(r'(?:compatible|fits|suitable)[\s:]*([^.\n]{10,80})', re.I),
            'material': re.compile(r'(?:material|made of|fabric)[\s:]*([^.\n]{5,50})', re.I),
            'dimensions': re.compile(r'(?:size|dimensions|length|width)[\s:]*([^.\n]{5,50})', re.I),
            'installation': re.compile(r'(?:installation|setup|mounting)[\s:]*([^.\n]{10,80})', re.I),
            'warranty': re.compile(r'(?:warranty|guarantee)[\s:]*([^.\n]{5,50})', re.I),
            'price': re.compile(r'(?:price|cost|₹|\$)[\s:]*([^.\n]{5,50})', re.I)
        }
        
        super().__init__("automotive_accessories", accessories_keywords, accessories_patterns)

# Furniture Content Extractor Class
class FurnitureContentExtractor(BaseContentExtractor):
    def __init__(self):
        furniture_keywords = [
            'sofa', 'chair', 'table', 'bed', 'mattress', 'wardrobe', 'cabinet', 'shelf',
            'wood', 'wooden', 'metal', 'fabric', 'leather', 'cushion', 'foam', 'spring',
            'dimensions', 'size', 'height', 'width', 'depth', 'seating', 'capacity',
            'assembly', 'installation', 'delivery', 'warranty', 'care', 'maintenance',
            'style', 'design', 'color', 'finish', 'modern', 'classic', 'contemporary'
        ]
        
        furniture_patterns = {
            'material': re.compile(r'(?:material|wood|fabric|leather|metal)[\s:]*([^.\n]{5,50})', re.I),
            'dimensions': re.compile(r'(?:dimensions|size|height|width|depth)[\s:]*([^.\n]{5,50})', re.I),
            'seating_capacity': re.compile(r'(?:seating|capacity|persons)[\s:]*([^.\n]{5,30})', re.I),
            'assembly': re.compile(r'(?:assembly|installation)[\s:]*([^.\n]{10,80})', re.I),
            'warranty': re.compile(r'(?:warranty|guarantee)[\s:]*([^.\n]{5,50})', re.I),
            'care': re.compile(r'(?:care|maintenance|cleaning)[\s:]*([^.\n]{10,80})', re.I),
            'price': re.compile(r'(?:price|cost|₹|\$)[\s:]*([^.\n]{5,50})', re.I)
        }
        
        super().__init__("furniture", furniture_keywords, furniture_patterns)

# Home Appliances Content Extractor Class
class HomeAppliancesContentExtractor(BaseContentExtractor):
    def __init__(self):
        appliances_keywords = [
            'refrigerator', 'fridge', 'washing machine', 'microwave', 'oven', 'dishwasher',
            'air conditioner', 'ac', 'heater', 'fan', 'cooler', 'purifier', 'vacuum',
            'capacity', 'liters', 'kg', 'watts', 'power', 'energy', 'star rating',
            'inverter', 'compressor', 'temperature', 'timer', 'remote', 'digital',
            'stainless steel', 'plastic', 'glass', 'warranty', 'service', 'installation'
        ]
        
        appliances_patterns = {
            'capacity': re.compile(r'(?:capacity|liters|kg|cubic)[\s:]*([^.\n]{5,50})', re.I),
            'power': re.compile(r'(?:power|watts|consumption)[\s:]*([^.\n]{5,50})', re.I),
            'energy_rating': re.compile(r'(?:energy|star rating|efficiency)[\s:]*([^.\n]{5,50})', re.I),
            'features': re.compile(r'(?:features|functions|modes)[\s:]*([^.\n]{10,100})', re.I),
            'warranty': re.compile(r'(?:warranty|guarantee)[\s:]*([^.\n]{5,50})', re.I),
            'installation': re.compile(r'(?:installation|setup)[\s:]*([^.\n]{10,80})', re.I),
            'price': re.compile(r'(?:price|cost|₹|\$)[\s:]*([^.\n]{5,50})', re.I)
        }
        
        super().__init__("home_appliances", appliances_keywords, appliances_patterns)

# Fashion Clothes Content Extractor Class
class FashionClothesContentExtractor(BaseContentExtractor):
    def __init__(self):
        fashion_keywords = [
            'shirt', 'pant', 'dress', 'jeans', 'jacket', 'sweater', 'hoodie', 'tshirt',
            'cotton', 'polyester', 'silk', 'wool', 'denim', 'fabric', 'material',
            'size', 'small', 'medium', 'large', 'xl', 'xxl', 'fit', 'slim', 'regular',
            'color', 'pattern', 'design', 'style', 'casual', 'formal', 'party',
            'wash', 'care', 'dry clean', 'machine wash', 'brand', 'collection'
        ]
        
        fashion_patterns = {
            'material': re.compile(r'(?:material|fabric|cotton|polyester|silk|wool)[\s:]*([^.\n]{5,50})', re.I),
            'size': re.compile(r'(?:size|small|medium|large|xl|fit)[\s:]*([^.\n]{5,30})', re.I),
            'style': re.compile(r'(?:style|design|casual|formal|pattern)[\s:]*([^.\n]{5,50})', re.I),
            'care': re.compile(r'(?:care|wash|cleaning|dry clean)[\s:]*([^.\n]{10,80})', re.I),
            'color': re.compile(r'(?:color|colour|shade)[\s:]*([^.\n]{5,30})', re.I),
            'brand': re.compile(r'(?:brand|designer|label)[\s:]*([^.\n]{5,30})', re.I),
            'price': re.compile(r'(?:price|cost|₹|\$)[\s:]*([^.\n]{5,50})', re.I)
        }
        
        super().__init__("fashion_clothes", fashion_keywords, fashion_patterns)

# Sports Equipment Content Extractor Class
class SportsEquipmentContentExtractor(BaseContentExtractor):
    def __init__(self):
        sports_keywords = [
            'cricket', 'football', 'basketball', 'tennis', 'badminton', 'gym', 'fitness',
            'bat', 'ball', 'racket', 'shoes', 'jersey', 'helmet', 'pads', 'gloves',
            'weight', 'dumbbell', 'barbell', 'treadmill', 'cycle', 'yoga', 'mat',
            'size', 'weight', 'material', 'grip', 'cushion', 'support', 'breathable',
            'professional', 'beginner', 'intermediate', 'training', 'competition',
            'brand', 'quality', 'durability', 'performance', 'comfort'
        ]
        
        sports_patterns = {
            'sport_type': re.compile(r'(?:cricket|football|basketball|tennis|badminton|gym|fitness)[\s:]*([^.\n]{5,50})', re.I),
            'material': re.compile(r'(?:material|made of|fabric|leather|rubber)[\s:]*([^.\n]{5,50})', re.I),
            'size_weight': re.compile(r'(?:size|weight|dimensions)[\s:]*([^.\n]{5,50})', re.I),
            'level': re.compile(r'(?:professional|beginner|intermediate|training)[\s:]*([^.\n]{5,50})', re.I),
            'features': re.compile(r'(?:features|grip|cushion|support|breathable)[\s:]*([^.\n]{10,80})', re.I),
            'brand': re.compile(r'(?:brand|manufacturer)[\s:]*([^.\n]{5,30})', re.I),
            'price': re.compile(r'(?:price|cost|₹|\$)[\s:]*([^.\n]{5,50})', re.I)
        }
        
        super().__init__("sports_equipment", sports_keywords, sports_patterns)

# Enhanced Product Extractor Class
class ProductExtractor:
    def __init__(self):
        self.category_patterns = {
            'automotive': {
                'brands': ['Honda', 'Toyota', 'Hyundai', 'Maruti', 'Tata', 'Mahindra', 'Ford', 'BMW', 'Audi', 'Mercedes', 'Volkswagen'],
                'indicators': ['car', 'suv', 'sedan', 'hatchback', 'vehicle', 'auto', 'brochure', 'dealer', 'showroom'],
                'pattern': re.compile(r'(HONDA|TOYOTA|HYUNDAI|MARUTI|TATA|MAHINDRA|FORD|BMW|AUDI)\s+([A-Z][a-z]+)', re.I)
            },
            'mobile': {
                'brands': ['Apple', 'Samsung', 'Xiaomi', 'OnePlus', 'Realme', 'Oppo', 'Vivo', 'Google', 'Huawei'],
                'indicators': ['phone', 'mobile', 'smartphone', 'android', 'ios', 'camera', 'battery', 'display'],
                'pattern': re.compile(r'(Apple|Samsung|Xiaomi|OnePlus|Realme|Oppo|Vivo|Google)\s+([A-Z0-9][a-z0-9]*)', re.I)
            },
            'laptop': {
                'brands': ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 'Alienware', 'ThinkPad'],
                'indicators': ['laptop', 'notebook', 'ultrabook', 'gaming', 'processor', 'ram', 'ssd', 'graphics'],
                'pattern': re.compile(r'(Dell|HP|Lenovo|Asus|Acer|Apple|MSI)\s+([A-Z0-9][a-z0-9]*)', re.I)
            },
            'automotive_accessories': {
                'brands': ['Bosch', '3M', 'Philips', 'Osram', 'Pioneer', 'JBL', 'Kenwood'],
                'indicators': ['car accessory', 'vehicle accessory', 'dashboard', 'seat cover', 'floor mat', 'charger', 'mount'],
                'pattern': re.compile(r'(Bosch|3M|Philips|Osram|Pioneer|JBL|Kenwood)\s+([A-Z0-9][a-z0-9]*)', re.I)
            },
            'furniture': {
                'brands': ['IKEA', 'Godrej', 'Durian', 'Urban Ladder', 'Pepperfry', 'Nilkamal', 'Hometown'],
                'indicators': ['sofa', 'chair', 'table', 'bed', 'mattress', 'wardrobe', 'cabinet', 'furniture'],
                'pattern': re.compile(r'(IKEA|Godrej|Durian|Urban Ladder|Pepperfry|Nilkamal)\s+([A-Z][a-z]+)', re.I)
            },
            'home_appliances': {
                'brands': ['LG', 'Samsung', 'Whirlpool', 'Godrej', 'Haier', 'Bosch', 'IFB', 'Voltas', 'Blue Star'],
                'indicators': ['refrigerator', 'washing machine', 'microwave', 'air conditioner', 'dishwasher', 'appliance'],
                'pattern': re.compile(r'(LG|Samsung|Whirlpool|Godrej|Haier|Bosch|IFB|Voltas)\s+([A-Z0-9][a-z0-9]*)', re.I)
            },
            'fashion_clothes': {
                'brands': ['Nike', 'Adidas', 'Puma', 'Levi\'s', 'H&M', 'Zara', 'Uniqlo', 'Allen Solly', 'Peter England'],
                'indicators': ['shirt', 'pant', 'dress', 'jeans', 'jacket', 'clothing', 'apparel', 'fashion'],
                'pattern': re.compile(r'(Nike|Adidas|Puma|Levi\'s|H&M|Zara|Uniqlo|Allen Solly)\s+([A-Z][a-z]+)', re.I)
            },
            'sports_equipment': {
                'brands': ['Nike', 'Adidas', 'Puma', 'Reebok', 'Yonex', 'Wilson', 'Head', 'Babolat', 'Cosco'],
                'indicators': ['cricket', 'football', 'basketball', 'tennis', 'badminton', 'gym', 'fitness', 'sports'],
                'pattern': re.compile(r'(Nike|Adidas|Puma|Reebok|Yonex|Wilson|Head|Babolat|Cosco)\s+([A-Z][a-z]+)', re.I)
            }
        }
    
    def detect_category(self, filename: str, content: str) -> str:
        """Auto-detect category from filename and content"""
        text = f"{filename} {content[:1000]}".lower()
        
        scores = {}
        for category, data in self.category_patterns.items():
            score = sum(1 for indicator in data['indicators'] if indicator in text)
            scores[category] = score
        
        return max(scores, key=scores.get) if scores else 'other'
    
    def extract_parameters(self, filename: str, content: str, detected_category: str = None) -> Dict:
        """Extract all parameters in one go"""
        category = detected_category or self.detect_category(filename, content)
        
        result = {
            'category': category,
            'brand': None,
            'model': None,
            'product': None,
            'confidence': 0.0,
            'extraction_source': []
        }
        
        # Extract from filename
        filename_brand, filename_model = self._extract_from_filename(filename, category)
        if filename_brand:
            result['brand'] = filename_brand
            result['extraction_source'].append('filename')
        if filename_model:
            result['model'] = filename_model
            if 'filename' not in result['extraction_source']:
                result['extraction_source'].append('filename')
        
        # Extract from content (first 2000 chars for speed)
        content_brand, content_model = self._extract_from_content(content[:2000], category)
        if content_brand and not result['brand']:
            result['brand'] = content_brand
            result['extraction_source'].append('content')
        if content_model and not result['model']:
            result['model'] = content_model
            if 'content' not in result['extraction_source']:
                result['extraction_source'].append('content')
        
        # Build product name
        if result['brand'] and result['model']:
            result['product'] = f"{result['brand']} {result['model']}"
            result['confidence'] = 0.8
        elif result['brand']:
            result['product'] = result['brand']
            result['confidence'] = 0.5
        else:
            # Fallback to cleaned filename
            result['product'] = self._clean_filename(filename)
            result['confidence'] = 0.3
        
        return result
    
    def _extract_from_filename(self, filename: str, category: str) -> tuple:
        """Extract brand and model from filename"""
        base = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        tokens = [t for t in base.split() if len(t) > 2]  # Filter short tokens
        
        brand, model = None, None
        
        if category in self.category_patterns:
            brands = self.category_patterns[category]['brands']
            for i, token in enumerate(tokens):
                if any(brand_name.lower() in token.lower() for brand_name in brands):
                    brand = next(b for b in brands if b.lower() in token.lower())
                    # Next significant token is likely model
                    if i + 1 < len(tokens):
                        model = tokens[i + 1].title()
                    break
        
        return brand, model
    
    def _extract_from_content(self, content: str, category: str) -> tuple:
        """Extract brand and model from content"""
        if category in self.category_patterns:
            pattern = self.category_patterns[category]['pattern']
            match = pattern.search(content)
            if match:
                return match.group(1).title(), match.group(2).title()
        
        return None, None
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename as fallback product name"""
        base = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        # Remove common words
        words = base.split()
        filtered = [w for w in words if w.lower() not in ['brochure', 'manual', 'guide', '2024', '2023', 'new', 'reinforced', 'safety']]
        return ' '.join(filtered[:3])  # Take first 3 significant words

# Initialize all extractors
extractor = ProductExtractor()
automotive_extractor = AutomotiveContentExtractor()
mobile_extractor = MobileContentExtractor()
laptop_extractor = LaptopContentExtractor()
automotive_accessories_extractor = AutomotiveAccessoriesContentExtractor()
furniture_extractor = FurnitureContentExtractor()
home_appliances_extractor = HomeAppliancesContentExtractor()
fashion_clothes_extractor = FashionClothesContentExtractor()
sports_equipment_extractor = SportsEquipmentContentExtractor()

# Category to extractor mapping
CATEGORY_EXTRACTORS = {
    'automotive': automotive_extractor,
    'mobile': mobile_extractor,
    'laptop': laptop_extractor,
    'automotive_accessories': automotive_accessories_extractor,
    'furniture': furniture_extractor,
    'home_appliances': home_appliances_extractor,
    'fashion_clothes': fashion_clothes_extractor,
    'sports_equipment': sports_equipment_extractor
}

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class SettingsUpdate(BaseModel):
    user_location: Optional[str] = None
    disable_external: Optional[bool] = None
    ocr_settings: Optional[Dict[str, Any]] = None

class URLInput(BaseModel):
    urls: List[str]

class ChatResponse(BaseModel):
    response: str
    used_external: bool = False

# --- Helper functions (keeping all existing ones) ---
def fetch_and_extract_category_content(urls: List[str], category: str, max_urls: int = 5) -> List[str]:
    """Pre-fetch URLs and extract category-specific content with Indian source prioritization"""
    extracted_contents = []
    
    # Get the appropriate extractor for the category
    extractor_instance = CATEGORY_EXTRACTORS.get(category)
    if not extractor_instance:
        logger.warning(f"No extractor found for category: {category}")
        return extracted_contents
    
    # Prioritize Indian sources
    indian_domains = ['.in', '.co.in', 'carwale', 'cardekho', 'zigwheels', '91mobiles', 'smartprix', 'flipkart', 'amazon.in', 'autocarindia', 'team-bhp', 'gadgets360']
    
    def is_indian_source(url):
        return any(domain in url.lower() for domain in indian_domains)
    
    # Sort URLs to prioritize Indian sources
    prioritized_urls = sorted(urls, key=lambda x: (not is_indian_source(x), urls.index(x)))
    
    for i, url in enumerate(prioritized_urls[:max_urls]):
        try:
            logger.info(f"Fetching {category} content from: {url} ({'Indian' if is_indian_source(url) else 'International'} source)")
            resp = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8"
            }, timeout=15)
            if not resp.ok:
                continue
                
            # Extract category-specific content
            extracted_data = extractor_instance.extract_content(resp.text, url)
            formatted_content = extractor_instance.format_for_knowledge_base(extracted_data)
            
            if formatted_content and len(formatted_content.strip()) > 100:  # Only add substantial content
                # Add source priority indicator for Indian sources
                if is_indian_source(url):
                    formatted_content = f"[INDIAN SOURCE - HIGH PRIORITY FOR PRICING & AVAILABILITY]\n{formatted_content}"
                
                extracted_contents.append(formatted_content)
                logger.info(f"Successfully extracted {category} content from: {url}")
            
        except Exception as e:
            logger.error(f"Error fetching {category} content from {url}: {str(e)}")
            continue
    
    indian_sources_count = sum(1 for url in prioritized_urls[:len(extracted_contents)] if is_indian_source(url))
    logger.info(f"Extracted {len(extracted_contents)} contents for {category} ({indian_sources_count} from Indian sources)")
    return extracted_contents

# Backward compatibility function
def fetch_and_extract_automotive_content(urls: List[str], max_urls: int = 5) -> List[str]:
    """Pre-fetch automotive URLs and extract relevant content (backward compatibility)"""
    return fetch_and_extract_category_content(urls, 'automotive', max_urls)

def build_candidate_urls_by_category(product: str, category: str, brand: str = None, location: str = None) -> list:
    """Enhanced URL building with better patterns and search fallbacks"""
    urls = []
    slug = product.lower().replace(' ', '-').replace('_', '-')
    search_query = product.replace(' ', '+')
    city = extract_city_from_location(location) if location else "Chennai"
    
    # Category-specific URL patterns
    if category == 'automotive':
        # Prioritize Indian automotive sources based on location
        is_indian_location = location and ('india' in location.lower() or 'chennai' in location.lower() or 'delhi' in location.lower() or 'mumbai' in location.lower() or 'bangalore' in location.lower())
        
        if is_indian_location or not location:
            # Indian automotive sources with multiple URL patterns
            if brand:
                brand_slug = brand.lower().replace(' ', '-')
                # CarWale patterns
                urls.extend([
                    f"https://www.carwale.com/{brand_slug}-{slug}",
                    f"https://www.carwale.com/{brand_slug}/{slug}",
                    f"https://www.carwale.com/search?q={search_query}",
                ])
                
                # CarDekho patterns  
                urls.extend([
                    f"https://www.cardekho.com/{brand_slug}-{slug}",
                    f"https://www.cardekho.com/{brand_slug}/{slug}",
                    f"https://www.cardekho.com/search?q={search_query}",
                ])
                
                # ZigWheels patterns
                urls.extend([
                    f"https://www.zigwheels.com/{brand_slug}-{slug}",
                    f"https://www.zigwheels.com/{brand_slug}/{slug}",
                    f"https://www.zigwheels.com/search?q={search_query}",
                ])
            else:
                # Without brand, use search-based URLs
                urls.extend([
                    f"https://www.carwale.com/search?q={search_query}",
                    f"https://www.cardekho.com/search?q={search_query}",
                    f"https://www.zigwheels.com/search?q={search_query}",
                ])
            
            # Additional Indian automotive sources
            urls.extend([
                f"https://www.autocarindia.com/search?q={search_query}",
                f"https://www.team-bhp.com/forum/search.php?query={search_query}",
                f"https://www.drivespark.com/search?q={search_query}",
                f"https://www.91wheels.com/search?q={search_query}",
                f"https://www.rushlane.com/search?q={search_query}",
                f"https://www.motorbeam.com/search?q={search_query}"
            ])
            
            # Indian brand official sites with better patterns
            if brand:
                brand_lower = brand.lower().replace(' ', '')
                urls.extend([
                    f"https://www.{brand_lower}.co.in",
                    f"https://www.{brand_lower}.in", 
                    f"https://www.{brand_lower}india.com",
                    f"https://www.{brand_lower}.com/in"
                ])
        else:
            # International automotive sources
            if brand:
                brand_lower = brand.lower().replace(' ', '')
                urls.extend([
                    f"https://www.{brand_lower}.com",
                    f"https://www.{brand_lower}.com/vehicles",
                    f"https://www.{brand_lower}.com/models"
                ])
            
            urls.extend([
                f"https://www.autotrader.com/cars-for-sale?searchRadius=0&makeCodeList={brand.upper() if brand else ''}&modelCodeList={slug.split('-')[-1].upper()}",
                f"https://www.cars.com/shopping/results/?q={search_query}",
                f"https://www.edmunds.com/{brand.lower()}/{slug.split('-')[-1]}/" if brand else f"https://www.edmunds.com/search/?q={search_query}"
            ])
        
        # Wikipedia as fallback (lower priority)
        urls.append(f"https://en.wikipedia.org/wiki/{product.replace(' ', '_')}")
    
    elif category == 'mobile':
        # Prioritize Indian mobile sources based on location
        is_indian_location = location and ('india' in location.lower() or 'chennai' in location.lower() or 'delhi' in location.lower() or 'mumbai' in location.lower() or 'bangalore' in location.lower())
        
        if is_indian_location or not location:
            # Indian mobile sources with search fallbacks
            urls.extend([
                f"https://www.91mobiles.com/{slug}-price-in-india",
                f"https://www.91mobiles.com/search?query={search_query}",
                f"https://www.smartprix.com/mobiles/{slug}",
                f"https://www.smartprix.com/mobiles/search/{search_query}",
                f"https://www.flipkart.com/search?q={search_query}+mobile",
                f"https://www.amazon.in/s?k={search_query}+mobile",
                f"https://www.gadgets360.com/{slug}",
                f"https://www.gadgets360.com/search?searchtext={search_query}",
                f"https://www.mysmartprice.com/mobile/{slug}",
                f"https://www.mysmartprice.com/mobile/search/{search_query}",
                f"https://www.gsmarena.com/search.php3?sName={search_query}"
            ])
            
            # Indian brand official sites
            if brand:
                brand_lower = brand.lower().replace(' ', '')
                urls.extend([
                    f"https://www.{brand_lower}.in",
                    f"https://www.{brand_lower}.co.in",
                    f"https://www.{brand_lower}india.com",
                    f"https://www.{brand_lower}.com/in"
                ])
        else:
            # International mobile sources
            urls.extend([
                f"https://www.gsmarena.com/search.php3?sName={search_query}",
                f"https://www.phonearena.com/search?term={search_query}",
                f"https://www.androidcentral.com/search?query={search_query}",
                f"https://www.techradar.com/search?searchTerm={search_query}"
            ])
            
            if brand:
                brand_lower = brand.lower().replace(' ', '')
                urls.extend([
                    f"https://www.{brand_lower}.com",
                    f"https://www.{brand_lower}.com/mobile",
                    f"https://www.{brand_lower}.com/smartphones"
                ])
        
        # Wikipedia as fallback
        urls.append(f"https://en.wikipedia.org/wiki/{product.replace(' ', '_')}")
    
    elif category == 'laptop':
        # Prioritize Indian laptop sources based on location
        is_indian_location = location and ('india' in location.lower() or 'chennai' in location.lower() or 'delhi' in location.lower() or 'mumbai' in location.lower() or 'bangalore' in location.lower())
        
        if is_indian_location or not location:
            # Indian laptop sources with search fallbacks
            urls.extend([
                f"https://www.flipkart.com/search?q={search_query}+laptop",
                f"https://www.amazon.in/s?k={search_query}+laptop",
                f"https://www.croma.com/search/?q={search_query}+laptop",
                f"https://www.reliancedigital.in/search?q={search_query}+laptop",
                f"https://www.smartprix.com/laptops/{slug}",
                f"https://www.smartprix.com/laptops/search/{search_query}",
                f"https://www.gadgets360.com/{slug}",
                f"https://www.gadgets360.com/search?searchtext={search_query}+laptop",
                f"https://www.digit.in/search?q={search_query}+laptop",
                f"https://www.91mobiles.com/laptops/{slug}",
                f"https://www.91mobiles.com/laptops/search?query={search_query}"
            ])
            
            # Indian brand official sites
            if brand:
                brand_lower = brand.lower().replace(' ', '')
                urls.extend([
                    f"https://www.{brand_lower}.in",
                    f"https://www.{brand_lower}.co.in",
                    f"https://www.{brand_lower}india.com",
                    f"https://www.{brand_lower}.com/in"
                ])
        else:
            # International laptop sources
            urls.extend([
                f"https://www.notebookcheck.net/search.html?q={search_query}",
                f"https://www.laptopmag.com/search?searchTerm={search_query}",
                f"https://www.techradar.com/search?searchTerm={search_query}+laptop",
                f"https://www.pcmag.com/search?query={search_query}+laptop"
            ])
            
            if brand:
                brand_lower = brand.lower().replace(' ', '')
                urls.extend([
                    f"https://www.{brand_lower}.com",
                    f"https://www.{brand_lower}.com/laptops",
                    f"https://www.{brand_lower}.com/products"
                ])
        
        # Wikipedia as fallback
        urls.append(f"https://en.wikipedia.org/wiki/{product}")
    
    elif category == 'automotive_accessories':
        if brand:
            brand_lower = brand.lower()
            urls.extend([
                f"https://www.{brand_lower}.com/{slug}",
                f"https://www.{brand_lower}.in/{slug}"
            ])
        
        # Automotive accessories sources
        urls.extend([
            f"https://www.amazon.in/s?k={product.replace(' ', '+')}+car+accessory",
            f"https://www.flipkart.com/search?q={product.replace(' ', '+')}+car",
            f"https://www.carwale.com/accessories/{slug}",
            f"https://en.wikipedia.org/wiki/{product}"
        ])
    
    elif category == 'furniture':
        if brand:
            brand_lower = brand.lower().replace(' ', '')
            urls.extend([
                f"https://www.{brand_lower}.com/{slug}",
                f"https://www.{brand_lower}.in/{slug}"
            ])
        
        # Furniture-specific sources
        urls.extend([
            f"https://www.ikea.com/in/en/search/products/?q={product.replace(' ', '+')}",
            f"https://www.urbanladder.com/search?q={product.replace(' ', '+')}",
            f"https://www.pepperfry.com/search.html?q={product.replace(' ', '+')}",
            f"https://www.amazon.in/s?k={product.replace(' ', '+')}+furniture",
            f"https://en.wikipedia.org/wiki/{product}"
        ])
    
    elif category == 'home_appliances':
        if brand:
            brand_lower = brand.lower()
            urls.extend([
                f"https://www.{brand_lower}.com/{slug}",
                f"https://www.{brand_lower}.in/{slug}",
                f"https://www.{brand_lower}.co.in/{slug}"
            ])
        
        # Home appliances sources
        urls.extend([
            f"https://www.amazon.in/s?k={product.replace(' ', '+')}",
            f"https://www.flipkart.com/search?q={product.replace(' ', '+')}",
            f"https://www.croma.com/search/?q={product.replace(' ', '+')}",
            f"https://www.reliancedigital.in/search?q={product.replace(' ', '+')}",
            f"https://en.wikipedia.org/wiki/{product}"
        ])
    
    elif category == 'fashion_clothes':
        if brand:
            brand_lower = brand.lower().replace(' ', '').replace("'", "")
            urls.extend([
                f"https://www.{brand_lower}.com/{slug}",
                f"https://www.{brand_lower}.in/{slug}"
            ])
        
        # Fashion sources
        urls.extend([
            f"https://www.myntra.com/{slug}",
            f"https://www.ajio.com/search/?text={product.replace(' ', '+')}",
            f"https://www.amazon.in/s?k={product.replace(' ', '+')}+clothing",
            f"https://www.flipkart.com/search?q={product.replace(' ', '+')}+clothes",
            f"https://en.wikipedia.org/wiki/{product}"
        ])
    
    elif category == 'sports_equipment':
        if brand:
            brand_lower = brand.lower()
            urls.extend([
                f"https://www.{brand_lower}.com/{slug}",
                f"https://www.{brand_lower}.in/{slug}"
            ])
        
        # Sports equipment sources
        urls.extend([
            f"https://www.decathlon.in/search?q={product.replace(' ', '+')}",
            f"https://www.amazon.in/s?k={product.replace(' ', '+')}+sports",
            f"https://www.flipkart.com/search?q={product.replace(' ', '+')}+sports",
            f"https://www.sportskeeda.com/{slug}",
            f"https://en.wikipedia.org/wiki/{product}"
        ])
    
    else:
        # Fallback to general sources
        urls.extend([
            f"https://en.wikipedia.org/wiki/{product}",
            f"https://www.amazon.in/s?k={product.replace(' ', '+')}",
            f"https://www.flipkart.com/search?q={product.replace(' ', '+')}"
        ])
    
    random.shuffle(urls)
    return urls

def infer_product_name_from_path(pdf_path: str) -> str:
    base = os.path.basename(pdf_path)
    name = os.path.splitext(base)[0]
    return name.replace('_', ' ').replace('-', ' ').strip()

def infer_product_name_from_content(documents: list) -> str:
    try:
        first_pages = []
        for d in documents:
            p = d.metadata.get("page") if isinstance(d.metadata, dict) else None
            if p in (0, 1, None) and len(first_pages) < 2:
                first_pages.append(d.page_content or "")
            if len(first_pages) >= 2:
                break
        blob = "\n".join(first_pages)[:8000]
        if not blob:
            return ""
        
        lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
        blacklist = {"brochure","safety","guide","manual","leaflet","insert","information","table of contents","index"}
        candidates = []
        for ln in lines:
            if 3 <= len(ln) <= 60 and any(c.isalpha() for c in ln):
                low = ln.lower()
                if any(b in low for b in blacklist):
                    continue
                caps = sum(1 for c in ln if c.isupper())
                score = caps / max(1, len(ln))
                if ln.istitle():
                    score += 0.2
                candidates.append((score, ln))
        
        seqs = re.findall(r"\b(?:[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+){0,3})\b", blob)
        for s in seqs:
            if 3 <= len(s) <= 40:
                candidates.append((0.15, s))
        if not candidates:
            return ""
        
        candidates.sort(key=lambda x: (abs(len(x[1].split())-2) <= 1, x[0]), reverse=True)
        best = candidates[0][1]
        # Strip common separators and quotes from ends
        best = best.strip('-â€":| "')
        return best
    except Exception:
        return ""

async def extract_pdf_text_fast(pdf_path: str) -> str:
    """Fast PDF text extraction - first 3 pages only"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Only read first 3 pages for speed
            for page_num in range(min(3, len(reader.pages))):
                text += reader.pages[page_num].extract_text()
                if len(text) > 2000:  # Stop if we have enough text
                    break
            return text
    except Exception:
        return ""

def extract_relevant_sentences(text: str) -> list:
    pattern = r"[^\.!?]{0,200}(?:FDA|Food and Drug Administration|EMA|approved|approval|indication)[^\.!?]{0,200}[\.!?]"
    return list({s.strip() for s in re.findall(pattern, text, flags=re.I)})

def extract_price_sentences(text: str) -> list:
    pattern = r"[^\.!?]{0,200}(?:price|pricing|msrp|mrp|cost|starts at|starting price|from|[$â‚¬Â£â‚¹]|INR|USD|EUR|GBP)[^\.!?]{0,200}[\.!?]"
    return list({s.strip() for s in re.findall(pattern, text, flags=re.I)})

def fetch_external_pricing_info(product: str, max_items: int = 4) -> str:
    findings = []
    category = app_state.get("product_category", "other")
    brand = app_state.get("product_brand", "")
    location = app_state.get("user_location", "")
    
    for url in build_candidate_urls_by_category(product, category, brand, location):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            if not resp.ok:
                continue
            html = resp.text
        except Exception:
            continue
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        except Exception:
            text = html
        for sent in extract_price_sentences(text):
            if len(findings) >= max_items:
                break
            domain = urlparse(url).netloc
            findings.append(f"{sent} [Source: {domain}]({url})")
        if len(findings) >= max_items:
            break
    return "\n".join(f"- {f}" for f in findings)

def needs_external_pricing(question: str, context: str) -> bool:
    q = (question or "").lower()
    ctx = (context or "")
    keywords = ["price", "pricing", "cost", "msrp", "mrp", "quote", "discount", "starting price", "starts at", "from price"]
    price_in_ctx = ("price" in ctx.lower()) or bool(re.search(r"[$â‚¬Â£â‚¹]\s?\d", ctx))
    return any(k in q for k in keywords) and not price_in_ctx

def _normalize_keywords(text: str) -> list:
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", text)
    stop = {"the","a","an","and","or","of","to","for","is","are","on","in","with","what","which","who","when","how","do","does","did","be","can","could","would","should","this","that","it","its","about"}
    return [w for w in words if w not in stop and len(w) > 2]

def extract_sentences_by_keywords(text: str, keywords: list, max_chars: int = 400) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keyset = set(keywords)
    picked = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) > max_chars:
            continue
        tokens = set(re.findall(r"[a-z0-9]+", s_clean.lower()))
        if tokens & keyset:
            picked.append(s_clean)
    return list(dict.fromkeys(picked))

def _official_brand_domains(brand: str) -> list:
    """Guess official domains generically without brand-specific overrides. Merged variants."""
    b = (brand or "").strip().lower()
    if not b:
        return []
    guesses = [
        # Common official domains with and without www
        f"www.{b}.com", f"{b}.com",
        f"www.{b}.co.in", f"{b}.co.in",
        f"www.{b}.in", f"{b}.in",
        f"www.{b}.co.uk", f"{b}.co.uk",
        # Heuristic alternates sometimes used by brands
        f"{b}cars.com", f"{b}motors.com", f"{b}auto.com", f"{b}india.com",
    ]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for g in guesses:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out

def is_official_domain(domain: str, brand: str) -> bool:
    host = (domain or "").lower()
    allowed = _official_brand_domains(brand)
    return any(host.endswith(d.replace("www.", "")) or host == d for d in allowed)

def guess_official_url(product: str) -> str:
    try:
        tokens = re.findall(r"[A-Za-z0-9]+", product or "")
        if not tokens:
            return ""
        brand = tokens[0].lower()
        slug_dash = "-".join(t.lower() for t in tokens)
        prod_low = (product or "").lower()

        domains = _official_brand_domains(brand)
        candidates = []
        for dom in domains:
            base = f"https://{dom}/"
            candidates.extend([
                base, base + slug_dash, base + f"models/{slug_dash}",
                base + f"{slug_dash}/", base + f"cars/{slug_dash}"
            ])

        first_loaded_official = ""
        for url in candidates:
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                if not resp.ok:
                    continue
                html = resp.text
                if not first_loaded_official:
                    first_loaded_official = url
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                except Exception:
                    text = html
                tl = text.lower()
                if (prod_low and prod_low in tl) or (slug_dash in tl) or (len(tokens) > 1 and tokens[1].lower() in tl):
                    return url
            except Exception:
                continue
        return first_loaded_official
    except Exception:
        return ""

def fetch_external_general_info(product: str, question: str, max_items: int = 8, official_url: str = None) -> str:
    keys = _normalize_keywords(question)
    if not keys:
        keys = _normalize_keywords(product)
    findings = []

    if official_url:
        try:
            resp = requests.get(official_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            if resp.ok:
                html = resp.text
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                except Exception:
                    text = html
                for sent in extract_sentences_by_keywords(text, keys):
                    if len(findings) >= max_items:
                        break
                    domain = urlparse(official_url).netloc
                    findings.append(f"{sent} [Source: {domain}]({official_url})")
        except Exception:
            pass

    if len(findings) < max_items:
        brand = app_state.get("product_brand", "")
        category = app_state.get("product_category", "other")
        location = app_state.get("user_location", "")
        
        for url in build_candidate_urls_by_category(product, category, brand, location):
            if official_url and url == official_url:
                continue
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                if not resp.ok:
                    continue
                html = resp.text
            except Exception:
                continue
            try:
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
            except Exception:
                text = html
            domain = urlparse(url).netloc
            for sent in extract_sentences_by_keywords(text, keys):
                if len(findings) >= max_items:
                    break
                tag = " (Official)" if brand and is_official_domain(domain, brand) else ""
                findings.append(f"{sent} [Source: {domain}{tag}]({url})")
            if len(findings) >= max_items:
                break

    return "\n".join(f"- {f}" for f in findings)

def needs_external_general(question: str, context: str) -> bool:
    keys = set(_normalize_keywords(question))
    if not keys:
        return False
    ctx_tokens = set(re.findall(r"[a-z0-9]+", (context or "").lower()))
    return len(keys & ctx_tokens) == 0

def extract_city_from_location(location: str) -> str:
    try:
        if not location:
            return "Chennai"
        parts = [p.strip() for p in str(location).split(',') if p.strip()]
        return parts[0] if parts else "Chennai"
    except Exception:
        return "Chennai"

def needs_dealer_search(question: str) -> bool:
    q = (question or "").lower()
    kws = ["dealer", "dealership", "showroom", "nearest", "test drive", "book a test drive", "sales center", "service center"]
    return any(k in q for k in kws)






def fetch_official_dealers_generic(brand: str, location: str) -> str:
    """Generic dealer locator scraper based on brand's official domains.
    Tries common dealer locator paths and extracts concise dealer-like entries.
    """
    try:
        brand = (brand or "").strip()
        if not brand:
            return ""
        city = extract_city_from_location(location)
        domains = _official_brand_domains(brand)
        if not domains:
            return ""
        paths = [
            "dealers", "dealer-locator", "find-a-dealer", "find-dealer",
            "locate-a-dealer", "dealerships", "retailers", "store-locator",
            "locate", "showrooms", "findus", "storelocator"
        ]
        candidates = []
        for dom in domains:
            base = f"https://{dom}/"
            for p in paths:
                candidates.append(base + p)
                candidates.append(base + f"{p}?city={requests.utils.quote(city)}")
                candidates.append(base + f"{p}?q={requests.utils.quote(city)}")
        findings = []
        first_ok = ""
        for url in candidates:
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                if not resp.ok:
                    continue
                if not first_ok:
                    first_ok = url
                html = resp.text
            except Exception:
                continue
            try:
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
            except Exception:
                text = html
                soup = None
            # Heuristic extraction
            try:
                candidate_tags = soup.find_all(
                    lambda tag: tag.name in ("div", "li", "p", "span") and tag.get_text(strip=True)
                    and any(k in tag.get_text() for k in ["Dealer", "Showroom", "Sales", "Service", "Address", "Phone"]) 
                ) if soup else []
            except Exception:
                candidate_tags = []
            for tag in candidate_tags[:12]:
                txt = " ".join(tag.get_text(separator=" ", strip=True).split())
                if city.lower() in txt.lower() or "Dealer" in txt or "Showroom" in txt:
                    if 10 <= len(txt) <= 180:
                        findings.append(txt)
            if findings:
                break
        fallback_url = first_ok or (f"https://{domains[0]}/dealers" if domains else "")
        if not fallback_url and not findings:
            return ""
        out = [f"Official {brand.title()} dealer locator for {city}: {fallback_url}"]
        for it in dict.fromkeys(findings[:8]):
            out.append(it)
        return "\n".join(f"- {line}" for line in out if line.strip())
    except Exception:
        return ""

def fetch_official_dealers(brand: str, location: str) -> str:
    """Brand-agnostic official dealer lookup using generic heuristics only."""
    return fetch_official_dealers_generic(brand, location)

def fetch_external_approval_info(product: str, max_items: int = 8) -> str:
    findings = []
    category = app_state.get("product_category", "other")
    brand = app_state.get("product_brand", "")
    location = app_state.get("user_location", "")
    
    for url in build_candidate_urls_by_category(product, category, brand, location):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            if not resp.ok:
                continue
            html = resp.text
        except Exception:
            continue
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        except Exception:
            text = html
        for sent in extract_relevant_sentences(text):
            if len(findings) >= max_items:
                break
            domain = urlparse(url).netloc
            findings.append(f"{sent} [Source: {domain}]({url})")
        if len(findings) >= max_items:
            break
    return "\n".join(f"- {f}" for f in findings)

def needs_external_search(question: str, context: str) -> bool:
    q = (question or "").lower()
    ctx = (context or "")
    keywords = ["approval", "approved", "who approved", "fda", "ema", "indication", "approval history"]
    return any(k in q for k in keywords) and ("FDA" not in ctx and "approval" not in ctx.lower())

def setup_rag_pipeline(pdf_path: str, product_slug: str):
    db_persist_directory = "db_chroma"
    app_state["product_name"] = infer_product_name_from_path(pdf_path)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"Document not found at {pdf_path}")

    embed_model = "models/embedding-001"
    collection_name = f"{product_slug}_{embed_model.replace('/', '_')}"
    embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        if not docs and app_state["ocr_settings"]["enable_ocr"]:
            # OCR fallback
            ocr_pages = app_state["ocr_settings"]["ocr_pages"]
            poppler_path = app_state["ocr_settings"]["poppler_path"] or None
            if app_state["ocr_settings"]["tesseract_cmd"]:
                pytesseract.pytesseract.tesseract_cmd = app_state["ocr_settings"]["tesseract_cmd"]
            
            ocr_texts = []
            for idx in range(1, ocr_pages + 1):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        images = convert_from_path(
                            pdf_path,
                            first_page=idx,
                            last_page=idx,
                            poppler_path=poppler_path,
                            output_folder=tmpdir,
                            fmt="png"
                        )
                        if images:
                            img = images[0]
                            txt = pytesseract.image_to_string(img)
                            if txt and txt.strip():
                                ocr_texts.append((idx, txt))
                            del img
                            del images
                            gc.collect()
                except Exception:
                    continue
            
            if ocr_texts:
                ocr_docs = [Document(page_content=t, metadata={"source": pdf_path, "page": p}) for p, t in ocr_texts]
                docs = text_splitter.split_documents(ocr_docs)
        
        if not docs:
            raise HTTPException(status_code=422, detail="No extractable text found in the PDF")
        
        # Try to refine product name
        try:
            new_name = infer_product_name_from_content(documents if 'documents' in locals() else [])
            if new_name:
                app_state["product_name"] = new_name
        except Exception:
            pass
        
        vector_db = Chroma.from_documents(
            docs,
            embedding_function,
            persist_directory=db_persist_directory,
            collection_name=collection_name,
        )
        
        app_state["source_chunk_counts"] = {
            "mode": "PDF",
            "items": [(pdf_path, len(docs))]
        }
        
        return vector_db.as_retriever(search_kwargs={"k": 3})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

async def setup_rag_pipeline_from_urls(urls: List[str], product_slug: str):
    db_persist_directory = "db_chroma"
    embed_model = "models/embedding-001"
    collection_name = f"{product_slug}_{embed_model.replace('/', '_')}_urls"
    embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_docs = []
    name_candidates = []
    url_counts = []

    # Check category and pre-fetch content if supported
    category = app_state.get("product_category", "other")
    if category in CATEGORY_EXTRACTORS:
        logger.info(f"{category.title()} category detected - pre-fetching and extracting content")
        category_contents = fetch_and_extract_category_content(urls, category, max_urls=8)
        
        # Process pre-extracted category content
        for i, content in enumerate(category_contents):
            base_doc = Document(page_content=content, metadata={"source": f"{category}_extracted_{i}", "type": f"{category}_specs"})
            split_docs = text_splitter.split_documents([base_doc])
            url_counts.append((f"{category}_content_{i}", len(split_docs)))
            all_docs.extend(split_docs)
        
        logger.info(f"Added {len(category_contents)} pre-extracted {category} content pieces to knowledge base")

    # Process remaining URLs with standard extraction
    for url in urls:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            if not resp.ok:
                continue
            html = resp.text
        except Exception:
            continue
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        except Exception:
            text = html
        
        if not text or len(text.strip()) < 20:
            continue
        
        base_doc = Document(page_content=text, metadata={"source": url})
        split_docs = text_splitter.split_documents([base_doc])
        url_counts.append((url, len(split_docs)))
        all_docs.extend(split_docs)
        
        try:
            title = soup.title.get_text(strip=True) if soup and soup.title else ""
            if title:
                name_candidates.append(title)
        except Exception:
            pass

    if not all_docs:
        raise HTTPException(status_code=422, detail="No usable content from provided URLs")

    try:
        best = max(name_candidates, key=len) if name_candidates else ""
        if best:
            app_state["product_name"] = best
    except Exception:
        pass

    try:
        vector_db = Chroma.from_documents(
            all_docs,
            embedding_function,
            persist_directory=db_persist_directory,
            collection_name=collection_name,
        )
        
        app_state["source_chunk_counts"] = {
            "mode": "URL",
            "items": url_counts,
        }
        
        return vector_db.as_retriever(search_kwargs={"k": 3})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URLs: {str(e)}")

async def setup_multiple_pdfs(pdf_paths: List[str], product_slug: str):
    db_persist_directory = "db_chroma"
    embed_model = "models/embedding-001"
    collection_name = f"{product_slug}_{embed_model.replace('/', '_')}_pdfs"
    embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_docs = []
    per_pdf_counts = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        
        if not docs and app_state["ocr_settings"]["enable_ocr"]:
            try:
                ocr_pages = app_state["ocr_settings"]["ocr_pages"]
                poppler_path = app_state["ocr_settings"]["poppler_path"] or None
                if app_state["ocr_settings"]["tesseract_cmd"]:
                    pytesseract.pytesseract.tesseract_cmd = app_state["ocr_settings"]["tesseract_cmd"]
                
                ocr_texts = []
                for idx in range(1, ocr_pages + 1):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            images = convert_from_path(
                                pdf_path,
                                first_page=idx,
                                last_page=idx,
                                poppler_path=poppler_path,
                                output_folder=tmpdir,
                                fmt="png"
                            )
                            if images:
                                img = images[0]
                                txt = pytesseract.image_to_string(img)
                                if txt and txt.strip():
                                    ocr_texts.append((idx, txt))
                                del img
                                del images
                                gc.collect()
                    except Exception:
                        continue
                
                if ocr_texts:
                    ocr_docs = [Document(page_content=t, metadata={"source": pdf_path, "page": p}) for p, t in ocr_texts]
                    docs = text_splitter.split_documents(ocr_docs)
            except Exception:
                pass
        
        per_pdf_counts.append((pdf_path, len(docs)))
        all_docs.extend(docs)

    if not all_docs:
        raise HTTPException(status_code=422, detail="No extractable text found in any PDF")

    vector_db = Chroma.from_documents(all_docs, embedding_function, persist_directory=db_persist_directory, collection_name=collection_name)
    
    app_state["source_chunk_counts"] = {
        "mode": "PDF",
        "items": per_pdf_counts,
    }
    
    return vector_db.as_retriever(search_kwargs={"k": 3})

# Setup AutoGen agents
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
config_list = [{"api_type": "google", "model": model_name, "api_key": GOOGLE_API_KEY}]
llm_config = {"config_list": config_list, "temperature": 0.0, "timeout": 45}

InsightExtractor = autogen.AssistantAgent(
    name="InsightExtractor",
    system_message=(
        "You are a specialized data analyst for automotive, pharmaceutical, and electronics products. Your task is to read the user's question and the provided factual information. "
        "From the factual information, extract the most critical points that directly answer the user's question. "
        "For automotive products, prioritize specifications like engine details, fuel efficiency, safety features, dimensions, and pricing. "
        "For pharmaceutical products, focus on indications, dosing, safety, and regulatory information. "
        "For electronics, emphasize technical specifications, features, and performance metrics. "
        "Present these points as a concise, structured list (e.g., using bullet points or a numbered list). "
        "Do not be conversational. Do not add any information not present in the text. Your output is for an internal team member."
    ),
    llm_config=llm_config,
)

SalesConversationalist = autogen.AssistantAgent(
    name="SalesConversationalist",
    system_message=(
        "You are an AI Salesman Agent, a friendly and persuasive guide. "
        "You will receive a structured list of key insights. Your task is to: "
        "1. Weave these insights into a natural, helpful, and convincing conversational response. "
        "2. If no insights are available, clearly say you searched the product brochure and external sources but could not find an answer. "
        "3. If insights came from external sources, mention that you found them via external trusted sources. "
        "4. Address the user's underlying needs and concerns, using the insights as your evidence. "
        "5. Maintain a positive and reassuring tone to build trust. "
    ),
    llm_config={"config_list": config_list, "temperature": 0.3, "timeout": 45},
)

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config=False,
)

# === NEW API ROUTES FOR PARAMETER EXTRACTION ===

@app.post("/api/upload-pdf-with-extraction")
async def upload_pdf_with_extraction(files: List[UploadFile] = File(...)):
    """Upload PDF and extract parameters immediately"""
    try:
        results = []
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files allowed")
            
            # Save file temporarily
            base_name = os.path.splitext(file.filename)[0]
            base_slug = re.sub(r"[^a-z0-9]+", "-", base_name.lower()).strip("-")
            pdf_path = os.path.join("data", f"{base_slug}.pdf")
            
            with open(pdf_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Extract text quickly (first few pages only for speed)
            text_content = await extract_pdf_text_fast(pdf_path)
            
            # Extract parameters
            parameters = extractor.extract_parameters(file.filename, text_content)
            
            results.append({
                'filename': file.filename,
                'pdf_path': pdf_path,
                'parameters': parameters
            })
        
        return {
            'status': 'extracted',
            'files': results,
            'message': 'Parameters extracted. Please verify before processing.'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/confirm-and-process")
async def confirm_and_process(
    pdf_path: str = Form(...),
    brand: str = Form(...), 
    model: str = Form(...),
    category: str = Form(...),
    product_name: str = Form(...)
):
    """Process PDF with confirmed parameters"""
    try:
        print(f"Received parameters: pdf_path={pdf_path}, brand={brand}, model={model}, category={category}, product_name={product_name}")
        
        # Store confirmed parameters
        app_state.update({
            'product_brand': brand,
            'product_model': model,
            'product_category': category,
            'product_name': product_name,
            'product_info': {
                'brand': brand,
                'model': model,
                'product': product_name,
                'category': category
            }
        })
        
        # Create product slug from confirmed parameters
        product_slug = f"{brand}_{model}".lower().replace(' ', '_')
        app_state['product_slug'] = product_slug
        
        # Now do the full RAG pipeline setup
        print(f"Setting up RAG pipeline for: {pdf_path}")
        retriever = setup_rag_pipeline(pdf_path, product_slug)
        
        # If supported category, enhance with pre-fetched web content
        if category in CATEGORY_EXTRACTORS:
            print(f"{category.title()} category detected - enhancing with web content")
            try:
                # Build candidate URLs for category content
                candidate_urls = build_candidate_urls_by_category(product_name, category, brand, app_state.get("user_location"))
                
                # Pre-fetch and extract category content
                category_contents = fetch_and_extract_category_content(candidate_urls, category, max_urls=5)
                
                if category_contents:
                    # Add category content to existing retriever
                    embed_model = "models/embedding-001"
                    collection_name = f"{product_slug}_{embed_model.replace('/', '_')}_{category}"
                    embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                    
                    category_docs = []
                    for i, content in enumerate(category_contents):
                        base_doc = Document(page_content=content, metadata={"source": f"{category}_web_{i}", "type": f"{category}_specs"})
                        split_docs = text_splitter.split_documents([base_doc])
                        category_docs.extend(split_docs)
                    
                    if category_docs:
                        # Create additional vector store for category content
                        category_db = Chroma.from_documents(
                            category_docs,
                            embedding_function,
                            persist_directory="db_chroma",
                            collection_name=collection_name,
                        )
                        
                        # Update source counts to include category content
                        current_counts = app_state.get("source_chunk_counts", {"mode": "PDF", "items": []})
                        current_counts["items"].append((f"{category}_web_content", len(category_docs)))
                        app_state["source_chunk_counts"] = current_counts
                        
                        print(f"Enhanced knowledge base with {len(category_contents)} {category} web sources")
                        
            except Exception as e:
                print(f"Warning: Could not enhance with {category} web content: {str(e)}")
        
        app_state["retriever"] = retriever
        
        # Try to get official URL
        if not app_state.get("product_url"):
            guessed = guess_official_url(product_name)
            if guessed:
                app_state["product_url"] = guessed
        
        return {
            'status': 'success',
            'message': f'Successfully processed {product_name}',
            'ready_for_chat': True,
            'product_info': app_state['product_info']
        }
    
    except Exception as e:
        print(f"Error in confirm_and_process: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/greeting")
async def get_chat_greeting():
    """Get personalized greeting with detected parameters"""
    product_info = app_state.get('product_info', {})
    
    if not product_info:
        return {
            'greeting': 'Hello! Please upload a document first to get started.',
            'product_info': {}
        }
    
    product_name = product_info.get('product', 'Unknown Product')
    category = product_info.get('category', 'product')
    
    greeting = f"""Hello! I've analyzed your uploaded document and detected it's about the **{product_name}** in the {category} category.

I'm ready to answer any questions about:
- Key features and specifications
- Pricing and availability  
- Safety information
- Where to purchase or find dealers
- Technical details

What would you like to know about the {product_name}?"""
    
    return {
        'greeting': greeting,
        'product_info': product_info
    }

# === EXISTING API ROUTES (updated) ===

@app.get("/api/status")
async def get_status():
    try:
        return {
            "product_name": app_state.get("product_name", ""),
            "product_info": app_state.get("product_info", {}),
            "has_retriever": app_state["retriever"] is not None,
            "source_counts": app_state["source_chunk_counts"],
            "user_location": app_state["user_location"],
            "disable_external": app_state["disable_external"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        saved_paths = []
        bases = []
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            base_name = os.path.splitext(file.filename)[0]
            bases.append(base_name)
            base_slug = re.sub(r"[^a-z0-9]+", "-", base_name.lower()).strip("-")
            pdf_path = os.path.join("data", f"{base_slug}.pdf")
            
            with open(pdf_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_paths.append(pdf_path)
        
        # Create combined slug
        combined = "-".join(re.sub(r"[^a-z0-9]+", "-", b.lower()).strip("-") for b in bases)
        product_slug = (combined[:60] or "product-multi")
        
        app_state["product_slug"] = product_slug
        
        # Process PDFs
        if len(saved_paths) == 1:
            retriever = setup_rag_pipeline(saved_paths[0], product_slug)
        else:
            retriever = await setup_multiple_pdfs(saved_paths, product_slug)
        
        app_state["retriever"] = retriever
        
        return {
            "status": "success",
            "message": f"Uploaded and processed {len(saved_paths)} PDF(s)",
            "product_name": app_state["product_name"],
            "source_counts": app_state["source_chunk_counts"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/setup-urls")
async def setup_urls(url_input: URLInput):
    try:
        urls = url_input.urls
        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided")
        
        parsed_first = urlparse(urls[0])
        slug_source = (parsed_first.netloc + parsed_first.path).lower()
        product_slug = re.sub(r"[^a-z0-9]+", "-", slug_source).strip("-") or "product-urls"
        
        app_state["product_slug"] = product_slug
        retriever = await setup_rag_pipeline_from_urls(urls, product_slug)
        app_state["retriever"] = retriever
        
        return {
            "status": "success",
            "message": f"Processed {len(urls)} URL(s)",
            "product_name": app_state["product_name"],
            "source_counts": app_state["source_chunk_counts"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        if not app_state["retriever"]:
            raise HTTPException(status_code=400, detail="No knowledge base loaded. Please upload a PDF or configure URLs first.")
        
        prompt = message.message
        
        # Step 1: Enhanced Retrieval with Boosted Content Priority
        try:
            # Get main PDF content
            retrieved_docs = app_state["retriever"].invoke(prompt)
            main_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Initialize context with main content
            context_parts = [f"=== MAIN PRODUCT DOCUMENTATION ===\n{main_context}"]
            
            # If supported category, prioritize boosted content
            category = app_state.get("product_category", "other")
            if category in CATEGORY_EXTRACTORS:
                product_slug = app_state.get('product_slug', '')
                embed_model = "models/embedding-001"
                embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)
                
                # Try to load category collections (prioritize manual/boosted content)
                category_collections = [
                    (f"{product_slug}_{embed_model.replace('/', '_')}_{category}_manual", "BOOSTED WEB CONTENT", 4),  # Higher k for boosted
                    (f"{product_slug}_{embed_model.replace('/', '_')}_{category}", "AUTO WEB CONTENT", 2)  # Lower k for auto
                ]
                
                boosted_content_found = False
                for collection_name, content_type, k_value in category_collections:
                    try:
                        category_db = Chroma(
                            persist_directory="db_chroma",
                            collection_name=collection_name,
                            embedding_function=embedding_function
                        )
                        category_retriever = category_db.as_retriever(search_kwargs={"k": k_value})
                        category_docs = category_retriever.invoke(prompt)
                        
                        if category_docs:
                            category_context = "\n\n".join([doc.page_content for doc in category_docs])
                            context_parts.insert(1, f"=== {content_type} ({category.upper()}) ===\n{category_context}")
                            
                            if "BOOSTED" in content_type:
                                boosted_content_found = True
                                logger.info(f"✅ Using boosted {category} content with {len(category_docs)} chunks")
                            
                    except Exception as category_e:
                        logger.debug(f"{category.title()} collection {collection_name} not available: {str(category_e)}")
                
                if not boosted_content_found:
                    logger.warning(f"⚠️ No boosted content found for {category}. Consider using the boost functionality.")
            
            # Combine all context with boosted content prioritized
            context = "\n\n".join(context_parts)
                    
        except Exception as e:
            context = f"Error retrieving data: {e}"
        
        # Step 2: Optional external enrichment (if enabled)
        external_notes = ""
        used_external = False
        
        if not app_state["disable_external"]:
            if not app_state.get("product_url"):
                inferred_name = app_state.get("product_name") or ""
                guessed = guess_official_url(inferred_name)
                if guessed:
                    app_state["product_url"] = guessed

            if needs_external_search(prompt, context):
                product_name = app_state.get("product_name") or ""
                ext = fetch_external_approval_info(product_name, max_items=8)
                if ext:
                    external_notes += f"\n\nExternal Trusted Sources (auto-fetched):\n{ext}"
                    used_external = True
            
            if needs_external_pricing(prompt, context):
                product_name = app_state.get("product_name") or ""
                price_ext = fetch_external_pricing_info(product_name, max_items=8)
                if price_ext:
                    external_notes += f"\n\nExternal Pricing (auto-fetched):\n{price_ext}"
                    used_external = True

            if needs_dealer_search(prompt):
                brand = app_state.get("product_brand", "")
                dealer_info = fetch_official_dealers(brand, app_state.get("user_location", "Chennai, Tamil Nadu, India"))
                if dealer_info:
                    external_notes += f"\n\nOfficial Dealer Info (auto-fetched):\n{dealer_info}"
                    used_external = True

            if needs_external_general(prompt, context):
                product_name = app_state.get("product_name") or ""
                gen_ext = fetch_external_general_info(product_name, prompt, max_items=8, official_url=app_state.get("product_url"))
                if gen_ext:
                    external_notes += f"\n\nExternal General Info (auto-fetched):\n{gen_ext}"
                    used_external = True
        
        # Step 3: Agent processing with boosted content awareness
        boosted_indicator = "✅ ENHANCED WITH WEB SOURCES" if "BOOSTED WEB CONTENT" in context else "📄 PDF ONLY"
        
        analysis_task = f"""
        User's Question: "{prompt}"
        
        Knowledge Base Status: {boosted_indicator}
        
        Available Information Sources:
        ---
        {context if context else "No relevant information was found in the document."}
        ---
        {external_notes}
        
        IMPORTANT INSTRUCTIONS:
        1. If BOOSTED WEB CONTENT is available, prioritize it for pricing, specifications, and current market information
        2. Use MAIN PRODUCT DOCUMENTATION for detailed technical specifications and official information
        3. Clearly indicate when information comes from enhanced web sources vs. original documentation
        4. For pricing questions, always prefer boosted web content over PDF documentation
        5. Include brief citations with links when available
        
        Please extract and synthesize the key points from all available sources above.
        """
        
        async def _safe_initiate(recipient, message_content):
            delays = [2, 4, 6]
            for i, d in enumerate([0] + delays):
                if d:
                    await asyncio.sleep(d)
                try:
                    user_proxy.initiate_chat(
                        recipient=recipient,
                        message=message_content,
                        max_turns=1,
                        clear_history=True
                    )
                    return
                except GeminiClientError as ge:
                    if "RESOURCE_EXHAUSTED" not in str(ge):
                        raise
                    if i == len(delays):
                        raise
        
        await _safe_initiate(InsightExtractor, analysis_task)
        extracted_msg = user_proxy.last_message(InsightExtractor)
        insights = extracted_msg["content"] if extracted_msg else "No insights extracted."
        
        communication_task = f"""
        Here are the key insights our analyst has prepared. Include the links in-line where provided.
        ---
        {insights}
        ---
        Please transform these points into a polished, helpful, and convincing response for the user.
        If the insights reference external sources, mention that you found them via external trusted sources.
        If insights are empty or non-informative, clearly say you searched the product brochure and reliable open sources but could not find specific pricing or details, and offer to connect with a sales representative.
        """
        
        await _safe_initiate(SalesConversationalist, communication_task)
        final_msg = user_proxy.last_message(SalesConversationalist)
        final_response = final_msg["content"] if final_msg else "I encountered an issue processing your request."
        
        if app_state["product_url"] and app_state["product_url"] not in final_response:
            final_response += f"\n\nOfficial product page: {app_state['product_url']}"
        
        return {
            "response": final_response,
            "used_external": used_external
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize_document():
    try:
        if not app_state["retriever"]:
            raise HTTPException(status_code=400, detail="No knowledge base loaded")
        
        sample_docs = app_state["retriever"].invoke("summary overview")[:8]
        context_summary = "\n\n".join([d.page_content for d in sample_docs])[:12000]
        
        analysis_task = f"""
        Please produce a concise 5-7 bullet summary of the uploaded document.
        Focus on: purpose, key findings, indications, dosing (if present), safety, and any trial results.
        Source text:
        ---
        {context_summary if context_summary else "No content available."}
        ---
        Only use the text above.
        """
        
        async def _safe_initiate(recipient, message_content):
            delays = [2, 4, 6]
            for i, d in enumerate([0] + delays):
                if d:
                    await asyncio.sleep(d)
                try:
                    user_proxy.initiate_chat(
                        recipient=recipient,
                        message=message_content,
                        max_turns=1,
                        clear_history=True
                    )
                    return
                except GeminiClientError as ge:
                    if "RESOURCE_EXHAUSTED" not in str(ge):
                        raise
                    if i == len(delays):
                        raise
        
        await _safe_initiate(InsightExtractor, analysis_task)
        insights_msg = user_proxy.last_message(InsightExtractor)
        insights = insights_msg["content"] if insights_msg else "No insights extracted."
        
        communication_task = f"""
        Turn the following bullet points into a friendly, persuasive overview for a prospective customer.
        End with a clear call-to-action offering to connect them with a sales representative for tailored options and pricing.
        ---
        {insights}
        ---
        """
        
        await _safe_initiate(SalesConversationalist, communication_task)
        final_msg = user_proxy.last_message(SalesConversationalist)
        summary = final_msg["content"] if final_msg else "Couldn't generate summary."
        
        return {"summary": summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/settings")
async def update_settings(settings: SettingsUpdate):
    try:
        if settings.user_location:
            app_state["user_location"] = settings.user_location
        if settings.disable_external is not None:
            app_state["disable_external"] = settings.disable_external
        if settings.ocr_settings:
            app_state["ocr_settings"].update(settings.ocr_settings)
        
        return {"status": "success", "message": "Settings updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhance-category")
async def enhance_with_category_content():
    """Manually trigger category-specific content enhancement for existing products"""
    try:
        category = app_state.get("product_category", "other")
        if category not in CATEGORY_EXTRACTORS:
            raise HTTPException(status_code=400, detail=f"Content enhancement is not available for {category} category")
        
        if not app_state.get("retriever"):
            raise HTTPException(status_code=400, detail="No knowledge base loaded")
        
        product_name = app_state.get("product_name", "")
        brand = app_state.get("product_brand", "")
        product_slug = app_state.get("product_slug", "")
        
        if not product_name or not product_slug:
            raise HTTPException(status_code=400, detail="Product information not available")
        
        # Build candidate URLs for category content
        candidate_urls = build_candidate_urls_by_category(product_name, category, brand, app_state.get("user_location"))
        
        # Pre-fetch and extract category content
        category_contents = fetch_and_extract_category_content(candidate_urls, category, max_urls=8)
        
        if not category_contents:
            return {"status": "no_content", "message": f"No additional {category} content found"}
        
        # Add category content to knowledge base
        embed_model = "models/embedding-001"
        collection_name = f"{product_slug}_{embed_model.replace('/', '_')}_{category}_manual"
        embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        
        category_docs = []
        for i, content in enumerate(category_contents):
            base_doc = Document(page_content=content, metadata={"source": f"{category}_manual_{i}", "type": f"{category}_specs"})
            split_docs = text_splitter.split_documents([base_doc])
            category_docs.extend(split_docs)
        
        if category_docs:
            # Create additional vector store for category content
            category_db = Chroma.from_documents(
                category_docs,
                embedding_function,
                persist_directory="db_chroma",
                collection_name=collection_name,
            )
            
            # Update source counts to include category content
            current_counts = app_state.get("source_chunk_counts", {"mode": "PDF", "items": []})
            current_counts["items"].append((f"{category}_manual_content", len(category_docs)))
            app_state["source_chunk_counts"] = current_counts
            
            return {
                "status": "success", 
                "message": f"Enhanced knowledge base with {len(category_contents)} {category} web sources",
                "category": category,
                "sources_added": len(category_contents),
                "chunks_added": len(category_docs)
            }
        else:
            return {"status": "no_content", "message": f"No processable {category} content found"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Category-specific boost APIs for better performance
@app.post("/api/boost-automotive")
async def boost_automotive():
    """Boost automotive product knowledge with specialized content"""
    return await enhance_specific_category("automotive")

@app.post("/api/boost-mobile")
async def boost_mobile():
    """Boost mobile product knowledge with specialized content"""
    return await enhance_specific_category("mobile")

@app.post("/api/boost-laptop")
async def boost_laptop():
    """Boost laptop product knowledge with specialized content"""
    return await enhance_specific_category("laptop")

@app.post("/api/boost-automotive-accessories")
async def boost_automotive_accessories():
    """Boost automotive accessories knowledge with specialized content"""
    return await enhance_specific_category("automotive_accessories")

@app.post("/api/boost-furniture")
async def boost_furniture():
    """Boost furniture product knowledge with specialized content"""
    return await enhance_specific_category("furniture")

@app.post("/api/boost-home-appliances")
async def boost_home_appliances():
    """Boost home appliances knowledge with specialized content"""
    return await enhance_specific_category("home_appliances")

@app.post("/api/boost-fashion-clothes")
async def boost_fashion_clothes():
    """Boost fashion/clothes knowledge with specialized content"""
    return await enhance_specific_category("fashion_clothes")

@app.post("/api/boost-sports-equipment")
async def boost_sports_equipment():
    """Boost sports equipment knowledge with specialized content"""
    return await enhance_specific_category("sports_equipment")

async def enhance_specific_category(target_category: str):
    """Enhanced category-specific content fetching with optimized performance"""
    try:
        current_category = app_state.get("product_category", "other")
        
        # Validate category match
        if current_category != target_category:
            raise HTTPException(
                status_code=400, 
                detail=f"Current product is {current_category}, but trying to boost {target_category}"
            )
        
        if target_category not in CATEGORY_EXTRACTORS:
            raise HTTPException(
                status_code=400, 
                detail=f"Boost feature not available for {target_category} category"
            )
        
        if not app_state.get("retriever"):
            raise HTTPException(status_code=400, detail="No knowledge base loaded")
        
        product_name = app_state.get("product_name", "")
        brand = app_state.get("product_brand", "")
        product_slug = app_state.get("product_slug", "")
        
        if not product_name or not product_slug:
            raise HTTPException(status_code=400, detail="Product information not available")
        
        # Category-specific URL building with higher limits for better performance
        candidate_urls = build_candidate_urls_by_category(
            product_name, target_category, brand, app_state.get("user_location")
        )
        
        # Enhanced content fetching with category-specific optimizations
        max_urls = get_category_max_urls(target_category)
        category_contents = fetch_and_extract_category_content(
            candidate_urls, target_category, max_urls=max_urls
        )
        
        if not category_contents:
            return {
                "status": "no_content", 
                "message": f"No additional {target_category.replace('_', ' ')} content found",
                "category": target_category
            }
        
        # Optimized embedding and storage
        embed_model = "models/embedding-001"
        collection_name = f"{product_slug}_{embed_model.replace('/', '_')}_{target_category}_boost"
        embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)
        
        # Category-specific chunk sizing for optimal performance
        chunk_size, chunk_overlap = get_category_chunk_settings(target_category)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        category_docs = []
        for i, content in enumerate(category_contents):
            base_doc = Document(
                page_content=content, 
                metadata={
                    "source": f"{target_category}_boost_{i}", 
                    "type": f"{target_category}_specs",
                    "boost_timestamp": str(int(time.time()))
                }
            )
            split_docs = text_splitter.split_documents([base_doc])
            category_docs.extend(split_docs)
        
        if category_docs:
            # Create optimized vector store
            category_db = Chroma.from_documents(
                category_docs,
                embedding_function,
                persist_directory="db_chroma",
                collection_name=collection_name,
            )
            
            # Update source counts
            current_counts = app_state.get("source_chunk_counts", {"mode": "PDF", "items": []})
            current_counts["items"].append((f"{target_category}_boost_content", len(category_docs)))
            app_state["source_chunk_counts"] = current_counts
            
            return {
                "status": "success", 
                "message": f"🚀 Boosted {target_category.replace('_', ' ')} knowledge with {len(category_contents)} specialized sources",
                "category": target_category,
                "category_display": target_category.replace('_', ' ').title(),
                "sources_added": len(category_contents),
                "chunks_added": len(category_docs),
                "performance_optimized": True
            }
        else:
            return {
                "status": "no_content", 
                "message": f"No processable {target_category.replace('_', ' ')} content found",
                "category": target_category
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_category_max_urls(category: str) -> int:
    """Get optimized URL limits for each category"""
    category_limits = {
        'automotive': 10,           # High - lots of specs needed
        'mobile': 8,               # High - detailed tech specs
        'laptop': 8,               # High - detailed tech specs  
        'automotive_accessories': 6, # Medium - simpler products
        'furniture': 6,            # Medium - design focused
        'home_appliances': 7,      # Medium-High - specs important
        'fashion_clothes': 5,      # Lower - style focused
        'sports_equipment': 6      # Medium - specs and usage
    }
    return category_limits.get(category, 5)

def get_category_chunk_settings(category: str) -> tuple:
    """Get optimized chunk settings for each category"""
    category_settings = {
        'automotive': (1800, 250),        # Larger chunks for detailed specs
        'mobile': (1600, 200),           # Good for tech specifications
        'laptop': (1600, 200),           # Good for tech specifications
        'automotive_accessories': (1200, 150), # Smaller for simpler products
        'furniture': (1400, 180),        # Medium for descriptions
        'home_appliances': (1500, 200),  # Good for features and specs
        'fashion_clothes': (1000, 120),  # Smaller for style descriptions
        'sports_equipment': (1300, 160)  # Medium for usage and specs
    }
    return category_settings.get(category, (1500, 200))

# Backward compatibility endpoint
@app.post("/api/enhance-automotive")
async def enhance_with_automotive_content():
    """Backward compatibility for automotive content enhancement"""
    return await boost_automotive()

@app.get("/api/product-info")
async def get_product_info():
    """Get current product information detected from uploaded content"""
    try:
        product_info = {
            "product_name": app_state.get("product_name", ""),
            "product_brand": app_state.get("product_brand", ""),
            "product_model": app_state.get("product_model", ""),
            "product_category": app_state.get("product_category", ""),
            "product_slug": app_state.get("product_slug", ""),
            "user_location": app_state.get("user_location", ""),
            "has_knowledge_base": app_state.get("retriever") is not None,
            "supported_categories": list(CATEGORY_EXTRACTORS.keys()),
            "source_counts": app_state.get("source_chunk_counts")
        }
        
        # Generate sample URLs that would be used for boost
        if product_info["product_name"] and product_info["product_category"]:
            sample_urls = build_candidate_urls_by_category(
                product_info["product_name"], 
                product_info["product_category"], 
                product_info["product_brand"], 
                product_info["user_location"]
            )[:10]  # Show first 10 URLs as examples
            product_info["sample_boost_urls"] = sample_urls
        
        return product_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ManualBoostRequest(BaseModel):
    product_name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    max_urls: Optional[int] = 8

@app.post("/api/manual-boost")
async def manual_boost_with_params(request: ManualBoostRequest):
    """Manually trigger content boost with custom parameters"""
    try:
        if not app_state.get("retriever"):
            raise HTTPException(status_code=400, detail="No knowledge base loaded. Please upload a document first.")
        
        # Use provided parameters or fall back to detected ones
        product_name = request.product_name or app_state.get("product_name", "")
        brand = request.brand or app_state.get("product_brand", "")
        category = request.category or app_state.get("product_category", "")
        max_urls = request.max_urls or 8
        
        if not product_name:
            raise HTTPException(status_code=400, detail="Product name is required")
        
        if not category or category not in CATEGORY_EXTRACTORS:
            available_categories = list(CATEGORY_EXTRACTORS.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Valid category is required. Available categories: {available_categories}"
            )
        
        # Update app state with provided parameters
        if request.product_name:
            app_state["product_name"] = product_name
        if request.brand:
            app_state["product_brand"] = brand
        if request.category:
            app_state["product_category"] = category
        
        # Generate product slug
        product_slug = f"{brand}_{product_name}".lower().replace(' ', '_') if brand else product_name.lower().replace(' ', '_')
        app_state["product_slug"] = product_slug
        
        # Build candidate URLs
        candidate_urls = build_candidate_urls_by_category(product_name, category, brand, app_state.get("user_location"))
        
        logger.info(f"Manual boost: Fetching content for {product_name} ({category}) from {len(candidate_urls)} URLs")
        
        # Pre-fetch and extract category content
        category_contents = fetch_and_extract_category_content(candidate_urls, category, max_urls=max_urls)
        
        if not category_contents:
            return {
                "status": "no_content", 
                "message": f"No additional {category} content found for {product_name}",
                "urls_tried": candidate_urls[:max_urls],
                "suggestions": [
                    "Try adjusting the product name or brand",
                    "Check if the category is correct",
                    "Verify internet connectivity",
                    "Some websites might be blocking automated requests"
                ]
            }
        
        # Add category content to knowledge base
        embed_model = "models/embedding-001"
        collection_name = f"{product_slug}_{embed_model.replace('/', '_')}_{category}_manual"
        embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model=embed_model)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        
        category_docs = []
        for i, content in enumerate(category_contents):
            base_doc = Document(page_content=content, metadata={"source": f"{category}_manual_{i}", "type": f"{category}_specs"})
            split_docs = text_splitter.split_documents([base_doc])
            category_docs.extend(split_docs)
        
        if category_docs:
            # Create additional vector store for category content
            category_db = Chroma.from_documents(
                category_docs,
                embedding_function,
                persist_directory="db_chroma",
                collection_name=collection_name,
            )
            
            # Update source counts to include category content
            current_counts = app_state.get("source_chunk_counts", {"mode": "PDF", "items": []})
            current_counts["items"].append((f"{category}_manual_boost", len(category_docs)))
            app_state["source_chunk_counts"] = current_counts
            
            return {
                "status": "success", 
                "message": f"Successfully enhanced knowledge base with {len(category_contents)} {category} web sources for {product_name}",
                "product_name": product_name,
                "brand": brand,
                "category": category,
                "sources_added": len(category_contents),
                "chunks_added": len(category_docs),
                "urls_used": candidate_urls[:len(category_contents)]
            }
        else:
            return {
                "status": "error",
                "message": "Failed to process the fetched content",
                "urls_tried": candidate_urls[:max_urls]
            }
            
    except Exception as e:
        logger.error(f"Manual boost error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === FRONTEND ROUTE (must be last) ===
@app.get("/")
async def serve_app():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        raise ValueError("Google Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in .env file")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)