# Bangladesh Bank Rules and Regulations AI

An intelligent AI-powered consultation system for Bangladesh Bank regulations, rules, and guidelines. This system provides quick and accurate answers about banking regulations using Retrieval-Augmented Generation (RAG) with Google Gemini 2.5 Flash and Pinecone vector database.

## 🌟 Features

- **Fast & Accurate Consultation**: Get instant answers about Bangladesh Bank regulations
- **Bilingual Support**: Available in both English and Bangla
- **Google Gemini 2.5 Flash**: Lightning-fast responses with advanced AI
- **Document-Backed Answers**: All responses grounded in official Bangladesh Bank documents
- **RAG Architecture**: Combines vector search with Gemini AI for relevant, accurate responses
- **Simple & Lightweight**: Streamlined codebase with just two main files

## 📋 What You Can Ask About

This system covers various Bangladesh Bank regulations including:
- **Banking Company Act, 1991** (with amendments)
- **Bangladesh Bank Order, 1972**
- **Foreign Exchange Regulations Act, 1947**
- **Money Laundering Prevention Act**
- **Anti-Terrorism Act**
- **Basel III Guidelines** (RBCA)
- **Internal Control & Compliance Guidelines**
- **Corporate Social Responsibility (CSR) Guidelines**
- **Digital Banking Guidelines**
- And more...

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Pinecone API key
- Bangladesh Bank regulation documents (PDF format)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/NayemHasanLoLMan/Bangladesh-Bank-Rules-and-Regulation.git
cd Bangladesh-Bank-Rules-and-Regulation
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install google-generativeai pinecone-client pypdf2 python-dotenv
```

3. **Set up environment variables**

Create a `.env` file in the project root (or edit `prompt_leftout.txt` with your keys):
```env
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### Setup in 3 Steps

#### Step 1: Prepare Documents
Place all Bangladesh Bank regulation documents in the `documents/` folder:
- Supported format: PDF
- Official circulars, guidelines, and acts
- Ensure documents are readable and properly formatted

The repository includes a sample `issue_pdf` folder for reference documents.

#### Step 2: Upload to Pinecone
Run the upload script to process and index documents:

```bash
python upload.py
```

This will:
- Read all PDFs from the `documents/` folder
- Extract text from documents
- Split content into chunks with metadata
- Generate embeddings using Gemini
- Upload to Pinecone vector database

#### Step 3: Start Chatting
Run the chat interface:

```bash
python chat.py
```

## 📂 Project Structure

```
Bangladesh-Bank-Rules-and-Regulation/
│
├── 📄 chat.py                 # Main chat consultation system
├── 📄 upload.py               # Document upload to Pinecone
├── 📄 requirements.txt        # Python dependencies
├── 📄 prompt_leftout.txt      # Prompt templates and notes
├── 📄 .gitignore              # Git ignore rules
├── 📄 README.md               # This file
├── 📁 documents/              # Bangladesh Bank documents folder
└── 📁 issue_pdf/              # Sample/reference documents
```

## 💻 Usage Examples

### Basic Query (English)

```python
# Run chat.py and ask questions
python chat.py

# Example interaction:
You: What is the minimum capital requirement for banks?

AI: According to Bangladesh Bank regulations, the minimum 
paid-up capital requirement for a new bank is BDT 4,000 crore 
(40 billion taka) as per the recent guidelines...
```

### Bangla Query

```python
You: ব্যাংকের জন্য ন্যূনতম মূলধন প্রয়োজনীয়তা কী?

AI: বাংলাদেশ ব্যাংকের নির্দেশনা অনুযায়ী, নতুন ব্যাংকের জন্য 
ন্যূনতম পরিশোধিত মূলধন ৪,০০০ করোড় টাকা...
```

### Complex Regulatory Question

```python
You: What are the Basel III CRAR requirements for banks in Bangladesh?

AI: According to Basel III RBCA Guidelines issued by Bangladesh Bank, 
banks must maintain:
- Minimum Capital to Risk-Weighted Assets Ratio (CRAR): 10%
- Capital Conservation Buffer (CCB): 2.5%
- Total CRAR including CCB: 12.5%
```

### Foreign Exchange Regulations

```python
You: What are the regulations for foreign exchange transactions?

AI: Foreign exchange transactions in Bangladesh are governed by 
the Foreign Exchange Regulation Act, 1947 and subsequent Bangladesh Bank 
circulars. Key requirements include...
```

## 🏗️ System Architecture

```
User Query
    ↓
Language Detection (English/Bangla)
    ↓
Query Embedding (Gemini embedding-001)
    ↓
Vector Search (Pinecone) → Retrieve Top 5 Relevant Documents
    ↓
Context Building with Metadata
    ↓
Gemini 2.5 Flash Response Generation
    ↓
Formatted Answer with Source Citations
```

### Key Components

1. **upload.py**: Document processing and Pinecone indexing
   - PDF text extraction
   - Chunking with overlap
   - Embedding generation
   - Pinecone upload with metadata

2. **chat.py**: Main consultation interface
   - User query processing
   - Vector search
   - Context-aware response generation
   - Bilingual support

3. **documents/**: Official Bangladesh Bank regulations
   - PDFs of acts, circulars, guidelines
   - Source of truth for all responses

## ⚙️ Configuration

### Gemini Settings

```python
# Model configuration
MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

# Generation parameters
temperature = 0.3        # Lower for more factual responses
max_output_tokens = 2000  # Sufficient for detailed answers
top_k = 5                 # Number of documents to retrieve
top_p = 0.95             # Nucleus sampling parameter
```

### Pinecone Settings

```python
# Index configuration
INDEX_NAME = "bangladesh-bank-regulations"
DIMENSION = 3072          # Gemini embedding dimension
METRIC = "cosine"        # Similarity metric
CLOUD = "aws"
REGION = "us-east-1"
```

## 📊 Performance

- **Average Response Time**: 8-15 seconds
  - Embedding Generation: 1-2 seconds
  - Vector Search: 1-3 seconds
  - Gemini 2.5 Flash Generation: 5-10 seconds
  
- **Accuracy**: High (grounded in official documents)
- **Cost**: Very low (Gemini Flash is cost-effective)
- **Supported Languages**: English and Bangla

## 🔍 Key Features

### 1. Google Gemini 2.5 Flash
- **Lightning Fast**: Optimized for speed without compromising quality
- **Cost-Effective**: Significantly lower API costs
- **High Quality**: Excellent response quality and accuracy
- **Bangla Native Support**: Superior understanding of Bangla queries
- **Large Context Window**: Can process extensive regulatory text

### 2. Simple Two-File Architecture
- **upload.py**: One-time document processing and upload
- **chat.py**: Interactive consultation interface
- Easy to understand, modify, and maintain
- Minimal dependencies
- Quick deployment

### 3. Accurate Source Citations
Responses include references to specific regulations:
```
"According to Banking Company Act 1991, Section 11..."
"As per Bangladesh Bank BRPD Circular No. 15 (2016)..."
"Following Basel III RBCA Guidelines dated..."
```

### 4. Context-Aware Responses
- Understands follow-up questions
- Maintains conversation context
- Provides relevant examples
- Explains complex regulations clearly

## 📝 Example Use Cases

### For Bank Employees
- Quick lookup of regulatory requirements
- Compliance verification before actions
- Understanding new circulars and amendments
- Regulatory guidance for daily operations
- Policy clarification

### For Compliance Officers
- AML/CFT requirement verification
- Risk management guideline reference
- Audit preparation assistance
- Regulatory update tracking

### For Auditors
- Checking compliance requirements
- Understanding audit guidelines
- Regulatory framework verification
- Documentation requirements

### For Students & Researchers
- Learning about banking regulations
- Understanding regulatory framework
- Research on banking laws and policies
- Academic reference

### For Legal Professionals
- Quick reference to banking laws
- Understanding regulatory requirements
- Legal compliance verification
- Case preparation support

## 🛠️ Installation Details

### Required Python Packages

```txt
google-generativeai>=0.3.0
pinecone-client>=2.2.4
pypdf2>=3.0.1
python-dotenv>=1.0.0
```

### Installation Command

```bash
# From requirements.txt
pip install -r requirements.txt

# Or install individually
pip install google-generativeai pinecone-client pypdf2 python-dotenv
```

## 🚨 Error Handling

The system handles common errors gracefully:

```python
# API Key Errors
- Missing or invalid Google API key
- Missing or invalid Pinecone API key

# Document Processing Errors
- Corrupted PDF files
- Empty documents
- Unsupported file formats

# Query Errors
- Empty queries
- Network timeouts
- API rate limits
- No relevant documents found
```

## 🔒 Security Best Practices

1. **Environment Variables**
   - Store API keys in `.env` file
   - Never commit `.env` to version control
   - Use `.gitignore` to exclude sensitive files

2. **API Key Management**
   - Rotate keys regularly
   - Use separate keys for development and production
   - Monitor API usage and set limits

3. **Data Security**
   - Documents processed locally
   - Secure transmission to Pinecone
   - No data stored in chat logs by default

## 💡 Tips for Best Results

1. **Be Specific**: 
   - ✅ "What is the minimum CRAR under Basel III?"
   - ❌ "Tell me about capital"

2. **Use Keywords**: 
   - Include terms like "circular", "guideline", "act", "regulation"
   - Mention specific years or document numbers if known

3. **Provide Context**: 
   - "For conventional banks..." vs "For Islamic banks..."
   - Specify the type of institution when relevant

4. **Try Both Languages**: 
   - System works well in both English and Bangla
   - Mix languages if needed (code-switching supported)

5. **Follow-up Questions**: 
   - Ask clarifying questions
   - Request specific sections or details
   - Ask for examples or scenarios

## 🎯 Sample Questions You Can Ask

### Capital & Liquidity
- "What is the minimum CRAR for banks in Bangladesh?"
- "What are the Basel III liquidity requirements?"
- "How much capital does a new bank need?"
- "What is the Capital Conservation Buffer requirement?"

### Regulations & Compliance
- "What are the AML/CFT requirements for banks?"
- "What is the Foreign Exchange Regulation Act about?"
- "What are the CSR guidelines for banks?"
- "What are the KYC requirements?"

### Operations & Risk Management
- "What are the internal control guidelines?"
- "What are the requirements for digital banking?"
- "How should banks classify and provision NPLs?"
- "What are the credit risk management guidelines?"

### Governance & Management
- "What are the board composition requirements?"
- "What are the fit and proper criteria for directors?"
- "What are the corporate governance guidelines?"
- "What are the requirements for bank CEO?"

### Specific Circulars
- "What does BRPD Circular No. 15 say about...?"
- "Latest guidelines on agent banking?"
- "Mobile financial services regulations?"

## 📈 Roadmap & Future Enhancements

### Completed ✅
- [x] Document upload to Pinecone
- [x] Chat interface with Gemini 2.5 Flash
- [x] Bilingual support (English/Bangla)
- [x] Source citation in responses
- [x] PDF document processing

### In Progress 🚧
- [ ] Web interface for easier access
- [ ] Conversation history export

### Planned 📋
- [ ] Multi-turn conversation with memory
- [ ] Streaming responses for real-time output
- [ ] Document comparison features
- [ ] PDF export of consultation history
- [ ] RESTful API endpoints
- [ ] Mobile app (Android/iOS)
- [ ] Advanced search filters
- [ ] Document update notifications
- [ ] Circular tracking and alerts

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
1. **Add More Documents**: Upload additional Bangladesh Bank circulars
2. **Improve Prompts**: Enhance the system prompts in `prompt_leftout.txt`
3. **Bug Fixes**: Report and fix bugs
4. **Feature Requests**: Suggest new features
5. **Documentation**: Improve README or add guides

### Contribution Steps
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/NewFeature`
3. Make your changes
4. Test thoroughly
5. Commit: `git commit -m 'Add NewFeature'`
6. Push: `git push origin feature/NewFeature`
7. Open a Pull Request

## 📄 License

Licensed under the Apache 2.0 License.

## 👥 Author

**Hasan Mahmood (Nayem Hasan)**
- GitHub: [@NayemHasanLoLMan](https://github.com/NayemHasanLoLMan)
- Email: hasanmahmudnayeem3027@gmail.com

## 🙏 Acknowledgments

- **Bangladesh Bank** for official documentation and guidelines
- **Google** for Gemini 2.5 Flash AI model
- **Pinecone** for vector database infrastructure
- **Open source community** for Python libraries

## 📞 Support & Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/NayemHasanLoLMan/Bangladesh-Bank-Rules-and-Regulation/issues)
- **Email**: hasanmahmudnayeem3027@gmail.com
- **Discussions**: Use GitHub Discussions for general questions

## 🔗 Related Resources

- [Bangladesh Bank Official Website](https://www.bb.org.bd/)
- [Bangladesh Bank Circulars](https://www.bb.org.bd/en/index.php/publication/circuler)
- [Bangladesh Bank Guidelines](https://www.bb.org.bd/en/index.php/about/guidelist)
- [Google Gemini Documentation](https://ai.google.dev/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)

## ⚠️ Important Disclaimer

**Legal Notice**: This is an AI-powered consultation tool for informational and educational purposes only.

### Please Note:
- ✋ **Not Official Legal Advice**: This tool does not provide official legal, financial, or regulatory advice
- 🔍 **Verification Required**: Always verify critical information with official Bangladesh Bank circulars and regulations
- 👨‍💼 **Professional Consultation**: For official regulatory compliance and critical decisions, consult with qualified banking professionals and legal experts
- 🤖 **AI Limitations**: AI responses may contain errors, may be incomplete, or may be based on outdated information
- 📋 **Official Sources**: Always refer to official Bangladesh Bank publications and circulars for final, authoritative decisions
- 📅 **Currency of Information**: Regulations change frequently; verify with the latest official circulars
- ⚖️ **No Liability**: The authors and contributors assume no liability for decisions made based on this tool's output

### Recommended Usage:
- Use as a **starting point** for research and understanding
- Always **cross-reference** with official Bangladesh Bank sources
- Consult **qualified professionals** for compliance matters
- Stay updated with **latest circulars** and amendments


## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/NayemHasanLoLMan/Bangladesh-Bank-Rules-and-Regulation.git
cd Bangladesh-Bank-Rules-and-Regulation

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file and add:
# GOOGLE_API_KEY=your_key
# PINECONE_API_KEY=your_key

# Upload documents (one-time setup)
python upload.py

# Start chatting
python chat.py
```

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Status**: ✅ Active Development  
**Language**: 🐍 Python 100%
