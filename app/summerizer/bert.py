from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqGeneration,
    pipeline,
    BartForConditionalGeneration
)
import re
from typing import List, Dict, Union, Optional
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from collections import defaultdict

class MedicalReportSummarizer:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        medical_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_length: int = 150,
        min_length: int = 50,
        use_medical_preprocessing: bool = True
    ):
        """
        Initialize the medical report summarizer.
        
        Args:
            model_name: Base summarization model
            medical_model_name: Medical domain-specific model
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            use_medical_preprocessing: Whether to use medical-specific preprocessing
        """
        # Download necessary NLTK data
        nltk.download('punkt', quiet=True)
        
        # Initialize models and tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Initialize medical-specific pipeline
        self.medical_summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name
        )
        
        self.max_length = max_length
        self.min_length = min_length
        self.use_medical_preprocessing = use_medical_preprocessing
        
        # Medical terminology patterns
        self.medical_patterns = {
            'measurements': r'\d+\.?\d*\s*(mg|ml|g|kg|mm|cm|mcg)',
            'lab_values': r'\d+\.?\d*\s*(WBC|RBC|HGB|HCT|MCV|PLT)',
            'vital_signs': r'(BP|HR|RR|SpO2|Temp):?\s*\d+\.?\d*',
            'dates': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        }
        
        # Common medical section headers
        self.section_headers = {
            'history': ['past medical history', 'history of present illness', 'social history'],
            'examination': ['physical examination', 'clinical examination', 'findings'],
            'assessment': ['assessment', 'impression', 'diagnosis'],
            'plan': ['plan', 'treatment plan', 'recommendations']
        }

    def preprocess_medical_text(self, text: str) -> str:
        """
        Preprocess medical text with domain-specific rules.
        
        Args:
            text: Input medical report text
            
        Returns:
            Preprocessed text
        """
        # Normalize line breaks and spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Standardize section headers
        for section_type, headers in self.section_headers.items():
            for header in headers:
                pattern = re.compile(f'{header}:', re.IGNORECASE)
                text = pattern.sub(f'\n{section_type.upper()}:', text)
        
        # Preserve important medical patterns
        for pattern_name, pattern in self.medical_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(0)
                text = text.replace(value, f' {value} ')
        
        # Remove redundant whitespace
        text = ' '.join(text.split())
        
        return text

    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key sections from the medical report.
        
        Args:
            text: Input medical report text
            
        Returns:
            Dictionary containing text from each section
        """
        sections = defaultdict(str)
        current_section = 'other'
        
        lines = text.split('\n')
        for line in lines:
            # Check if line is a section header
            is_header = False
            for section_type, headers in self.section_headers.items():
                if any(header in line.lower() for header in headers):
                    current_section = section_type
                    is_header = True
                    break
            
            if not is_header:
                sections[current_section] += line + ' '
        
        return dict(sections)

    def summarize(
        self, 
        text: str,
        focus_areas: Optional[List[str]] = None,
        include_sections: bool = True
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Generate a comprehensive summary of the medical report.
        
        Args:
            text: Input medical report text
            focus_areas: Optional list of specific areas to focus on
            include_sections: Whether to include section-wise summaries
            
        Returns:
            Dictionary containing the summary and additional information
        """
        # Preprocess text
        if self.use_medical_preprocessing:
            processed_text = self.preprocess_medical_text(text)
        else:
            processed_text = text
        
        # Extract sections
        sections = self.extract_key_sections(processed_text)
        
        # Generate main summary
        main_summary = self.medical_summarizer(
            processed_text,
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=False
        )[0]['summary_text']
        
        result = {
            'main_summary': main_summary,
            'sections': {},
            'key_findings': self.extract_key_findings(processed_text)
        }
        
        # Generate section-wise summaries if requested
        if include_sections:
            for section, content in sections.items():
                if content.strip():
                    section_summary = self.medical_summarizer(
                        content,
                        max_length=self.max_length // 2,
                        min_length=self.min_length // 2,
                        do_sample=False
                    )[0]['summary_text']
                    result['sections'][section] = section_summary
        
        # Focus on specific areas if requested
        if focus_areas:
            result['focused_summary'] = self.generate_focused_summary(
                processed_text, focus_areas
            )
        
        return result

    def extract_key_findings(self, text: str) -> Dict[str, List[str]]:
        """
        Extract key medical findings from the text.
        
        Args:
            text: Preprocessed medical report text
            
        Returns:
            Dictionary containing categorized key findings
        """
        findings = {
            'measurements': [],
            'lab_values': [],
            'vital_signs': [],
            'dates': []
        }
        
        # Extract findings using medical patterns
        for category, pattern in self.medical_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            findings[category] = [match.group(0) for match in matches]
        
        return findings

    def generate_focused_summary(
        self,
        text: str,
        focus_areas: List[str]
    ) -> str:
        """
        Generate a summary focused on specific medical aspects.
        
        Args:
            text: Preprocessed medical report text
            focus_areas: List of areas to focus on
            
        Returns:
            Focused summary text
        """
        # Extract relevant sentences
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            if any(area.lower() in sentence.lower() for area in focus_areas):
                relevant_sentences.append(sentence)
        
        if not relevant_sentences:
            return "No specific information found for the requested focus areas."
        
        # Summarize relevant content
        focused_text = ' '.join(relevant_sentences)
        focused_summary = self.medical_summarizer(
            focused_text,
            max_length=self.max_length // 2,
            min_length=self.min_length // 2,
            do_sample=False
        )[0]['summary_text']
        
        return focused_summary

    def batch_summarize(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        """
        Summarize multiple medical reports in batches.
        
        Args:
            texts: List of medical report texts
            batch_size: Number of reports to process at once
            
        Returns:
            List of summaries for each report
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_summaries = [self.summarize(text) for text in batch]
            summaries.extend(batch_summaries)
        
        return summaries