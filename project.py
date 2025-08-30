import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import PyPDF2
import docx
import spacy
import re
import io
from datetime import datetime
import pandas as pd
from textstat import flesch_reading_ease
import json

class LegalDocumentAnalyzer:
    def __init__(self):
        """Initialize the Legal Document Analyzer with IBM Granite model"""
        self.model_name = "ibm-granite/granite-3.2-2b-instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        print("Loading IBM Granite 3.2 2B Instruct model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Load spaCy for NER
        self.nlp = spacy.load("en_core_web_sm")

        # Document type keywords
        self.doc_type_keywords = {
            "NDA": ["confidential", "non-disclosure", "proprietary", "trade secret", "confidentiality"],
            "Employment Contract": ["employment", "employee", "employer", "salary", "wage", "position", "job"],
            "Lease Agreement": ["lease", "rent", "tenant", "landlord", "property", "premises"],
            "Service Agreement": ["service", "services", "provider", "client", "deliverable", "scope of work"],
            "Purchase Agreement": ["purchase", "sale", "buyer", "seller", "goods", "merchandise"],
            "Partnership Agreement": ["partnership", "partner", "joint venture", "collaboration"]
        }

    def extract_text_from_file(self, file):
        """Extract text from uploaded file (PDF, DOCX, or TXT)"""
        if file is None:
            return ""

        # Determine the file path and name based on input type
        if isinstance(file, str):  # Input is a file path (from type="filepath")
            file_path = file
            file_extension = file_path.lower().split('.')[-1]
            file_name = file_path.split('/')[-1]
        elif hasattr(file, 'name'): # Input is a file-like object with a name
             file_path = file.name
             file_extension = file_path.lower().split('.')[-1]
             file_name = file_path.split('/')[-1]
        else: # Input might be bytes or an unexpected type
            return "Unsupported file input type."


        try:
            if file_extension == 'pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == 'docx':
                return self._extract_from_docx(file_path)
            elif file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                  return f.read()
            else:
                return "Unsupported file format. Please upload PDF, DOCX, or TXT files."
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def _extract_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_from_docx(self, file_path):
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def simplify_clause(self, clause_text):
        """Simplify complex legal clauses using IBM Granite model"""
        if not clause_text.strip():
            return "Please provide a legal clause to simplify."

        prompt = f"""<|user|>
You are a legal expert tasked with simplifying complex legal language. Please rewrite the following legal clause in simple, everyday language that a regular person can easily understand. Make it clear and concise while preserving the legal meaning.

Legal Clause: {clause_text}

Please provide a simplified version:
<|assistant|>
"""

        try:
            response = self.generator(
                prompt,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']

            simplified = response.split("<|assistant|>")[-1].strip()

            original_score = flesch_reading_ease(clause_text)
            simplified_score = flesch_reading_ease(simplified)

            result = f"**Simplified Version:**\n{simplified}\n\n"
            result += f"**Readability Improvement:**\n"
            result += f"Original Score: {original_score:.1f} → Simplified Score: {simplified_score:.1f}"

            return result

        except Exception as e:
            return f"Error in simplification: {str(e)}"

    def extract_named_entities(self, text):
        """Extract legal entities using spaCy and custom patterns"""
        if not text.strip():
            return "Please provide text for entity extraction."

        doc = self.nlp(text)
        entities = {
            "Persons/Organizations": [],
            "Dates": [],
            "Money": [],
            "Legal Terms": [],
            "Locations": [],
            "Other": []
        }

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                entities["Persons/Organizations"].append(f"{ent.text} ({ent.label_})")
            elif ent.label_ == "DATE":
                entities["Dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["Money"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["Locations"].append(f"{ent.text} ({ent.label_})")
            else:
                entities["Other"].append(f"{ent.text} ({ent.label_})")

        legal_terms = re.findall(r'\b(?:whereas|hereby|thereof|heretofore|agreement|contract|party|clause|section|article|terms|conditions|obligations|liability|indemnify|terminate|breach|default|force majeure)\b', text.lower())
        entities["Legal Terms"] = list(set(legal_terms))

        result = "**Named Entities Extracted:**\n\n"
        for category, items in entities.items():
            if items:
                result += f"**{category}:**\n"
                for item in set(items):
                    result += f"• {item}\n"
                result += "\n"

        return result if any(entities.values()) else "No significant entities found in the text."

    def extract_clauses(self, text):
        """Break down document into individual clauses"""
        if not text.strip():
            return "Please provide a document to analyze."

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        clauses = []
        current_clause = ""

        for sentence in sentences:
            if re.match(r'^\d+\.|\b(?:WHEREAS|NOW THEREFORE|Article|Section|Clause)\b', sentence, re.IGNORECASE):
                if current_clause:
                    clauses.append(current_clause.strip())
                current_clause = sentence
            else:
                current_clause += " " + sentence

            if len(current_clause) > 100 and sentence.endswith('.'):
                clauses.append(current_clause.strip())
                current_clause = ""

        if current_clause:
            clauses.append(current_clause.strip())

        result = f"**Document Breakdown - {len(clauses)} Clauses Identified:**\n\n"
        for i, clause in enumerate(clauses[:10], 1):
            result += f"**Clause {i}:**\n{clause}\n\n"

        if len(clauses) > 10:
            result += f"*... and {len(clauses) - 10} more clauses*"

        return result

    def classify_document(self, text):
        """Classify the document type using keyword matching and AI analysis"""
        if not text.strip():
            return "Please provide a document to classify."

        text_lower = text.lower()

        keyword_scores = {}
        for doc_type, keywords in self.doc_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                keyword_scores[doc_type] = score

        prompt = f"""<|user|>
Analyze the following legal document and classify it into one of these categories:
- NDA (Non-Disclosure Agreement)
- Employment Contract
- Lease Agreement
- Service Agreement
- Purchase Agreement
- Partnership Agreement
- Other

Document excerpt: {text[:1000]}...

What type of legal document is this? Provide a brief explanation.
<|assistant|>
"""

        try:
            ai_response = self.generator(
                prompt,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']

            ai_classification = ai_response.split("<|assistant|>")[-1].strip()

        except Exception as e:
            ai_classification = f"AI analysis failed: {str(e)}"

        result = "**Document Classification Results:**\n\n"

        if keyword_scores:
            result += "**Keyword-based Analysis:**\n"
            sorted_scores = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
            for doc_type, score in sorted_scores[:3]:
                result += f"• {doc_type}: {score} matching keywords\n"
            result += f"\n**Most Likely Type (Keywords):** {sorted_scores[0][0]}\n\n"

        result += f"**AI Analysis:**\n{ai_classification}"

        return result

    def analyze_full_document(self, file):
        """Perform comprehensive analysis of uploaded document"""
        if file is None:
            return "Please upload a document first."

        text = self.extract_text_from_file(file)
        if text.startswith("Error") or text.startswith("Unsupported"):
            return text

        word_count = len(text.split())
        char_count = len(text)
        readability = flesch_reading_ease(text)

        result = f"**📄 Document Analysis Report**\n\n"
        # Safely get file name
        file_name = file.name if hasattr(file, 'name') else "Uploaded File"
        result += f"**File:** {file_name}\n"
        result += f"**Size:** {char_count:,} characters, {word_count:,} words\n"
        result += f"**Readability Score:** {readability:.1f} (Flesch Reading Ease)\n"
        result += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        classification = self.classify_document(text)
        result += f"**Document Type:**\n{classification}\n\n"

        entities = self.extract_named_entities(text)
        result += f"**Key Entities:**\n{entities}\n\n"

        clauses = self.extract_clauses(text)
        result += f"**Document Structure:**\n{clauses}"

        return result

def create_legal_ai_interface():
    analyzer = LegalDocumentAnalyzer()

    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .feature-tab {
        border-radius: 8px;
        border: 1px solid #e1e5e9;
    }
    .output-text {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
    }
    """

    with gr.Blocks(css=custom_css, title="Legal Document AI Analyzer", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>⚖️ Legal Document AI Analyzer</h1>
            <p>Powered by IBM Granite 3.2 2B Instruct | Advanced Legal Text Processing</p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("📄 Document Analysis", elem_classes="feature-tab"):
                gr.Markdown("### Upload your legal document for comprehensive analysis")

                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload Legal Document",
                            file_types=[".pdf", ".docx", ".txt"],
                            type="filepath"   # ✅ fixed here
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyze Document",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        analysis_output = gr.Textbox(
                            label="Analysis Results",
                            lines=20,
                            elem_classes="output-text",
                            show_copy_button=True
                        )

                analyze_btn.click(
                    fn=analyzer.analyze_full_document,
                    inputs=[file_upload],
                    outputs=[analysis_output]
                )

            with gr.Tab("✨ Clause Simplification", elem_classes="feature-tab"):
                gr.Markdown("### Simplify complex legal language into plain English")

                with gr.Row():
                    with gr.Column():
                        clause_input = gr.Textbox(
                            label="Enter Legal Clause",
                            placeholder="Paste your complex legal clause here...",
                            lines=6
                        )
                        simplify_btn = gr.Button("🔄 Simplify Clause", variant="primary")

                    with gr.Column():
                        simplified_output = gr.Textbox(
                            label="Simplified Version",
                            lines=8,
                            elem_classes="output-text",
                            show_copy_button=True
                        )

                simplify_btn.click(
                    fn=analyzer.simplify_clause,
                    inputs=[clause_input],
                    outputs=[simplified_output]
                )

            with gr.Tab("🏷️ Entity Extraction", elem_classes="feature-tab"):
                gr.Markdown("### Extract key legal entities from your text")

                with gr.Row():
                    with gr.Column():
                        ner_input = gr.Textbox(
                            label="Enter Legal Text",
                            placeholder="Paste legal text here to extract entities...",
                            lines=8
                        )
                        extract_btn = gr.Button("🔍 Extract Entities", variant="primary")

                    with gr.Column():
                        ner_output = gr.Textbox(
                            label="Extracted Entities",
                            lines=10,
                            elem_classes="output-text",
                            show_copy_button=True
                        )

                extract_btn.click(
                    fn=analyzer.extract_named_entities,
                    inputs=[ner_input],
                    outputs=[ner_output]
                )

            with gr.Tab("📋 Document Classification", elem_classes="feature-tab"):
                gr.Markdown("### Automatically classify your legal documents")

                with gr.Row():
                    with gr.Column():
                        classify_input = gr.Textbox(
                            label="Enter Document Text",
                            placeholder="Paste document text here for classification...",
                            lines=10
                        )
                        classify_btn = gr.Button("🏷️ Classify Document", variant="primary")

                    with gr.Column():
                        classify_output = gr.Textbox(
                            label="Classification Results",
                            lines=12,
                            elem_classes="output-text",
                            show_copy_button=True
                        )

                classify_btn.click(
                    fn=analyzer.classify_document,
                    inputs=[classify_input],
                    outputs=[classify_output]
                )

            with gr.Tab("📑 Clause Breakdown", elem_classes="feature-tab"):
                gr.Markdown("### Break down documents into individual clauses")

                with gr.Row():
                    with gr.Column():
                        breakdown_input = gr.Textbox(
                            label="Enter Document Text",
                            placeholder="Paste document text here for clause extraction...",
                            lines=10
                        )
                        breakdown_btn = gr.Button("📋 Extract Clauses", variant="primary")

                    with gr.Column():
                        breakdown_output = gr.Textbox(
                            label="Clause Breakdown",
                            lines=15,
                            elem_classes="output-text",
                            show_copy_button=True
                        )

                breakdown_btn.click(
                    fn=analyzer.extract_clauses,
                    inputs=[breakdown_input],
                    outputs=[breakdown_output]
                )

        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <p><strong>Legal Document AI Analyzer v1.0</strong></p>
            <p>🤖 Powered by IBM Granite 3.2 2B Instruct | Built with Gradio</p>
            <p><em>⚠️ This tool is for assistance only. Always consult with qualified legal professionals for official legal advice.</em></p>
        </div>
        """)

    return demo

if __name__ == "__main__":
    print("🚀 Starting Legal Document AI Analyzer...")
    demo = create_legal_ai_interface()
    demo.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )