# Resume Parser

Resume Parser is a tool designed to automatically extract relevant information from resumes/CVs. It helps recruiters and HR professionals quickly identify candidate details such as contact information, education, work experience, and skills, streamlining the hiring process.

## Features

- **Automated Extraction**: Parse resumes in various formats (PDF, DOCX, TXT) to pull out useful candidate information.
- **Structured Output**: Outputs data in a structured format (JSON, CSV, etc.) for easy integration with other systems.
- **Customizable Fields**: Configure which fields to extract based on your hiring needs.
- **Multi-language Support**: Handles resumes written in different languages.
- **Easy Integration**: Can be integrated into existing HR workflows or ATS platforms.
- **Vector Database Embeddings**: Store and search resume data using vector embeddings for semantic retrieval.

## Installation

Clone the repository:

```bash
git clone https://github.com/Ashmit456/resume-parser.git
cd resume-parser
```

Install the required dependencies (see `requirements.txt`):

```bash
pip install -r requirements.txt
```

## Usage

To run the Resume Parser API:

```bash
python api.py
```

## Vector Database & Embeddings

This project supports storing resume data as vector embeddings for advanced semantic search.  
Typical workflow for embedding and vector DB operations (see `commands.txt` for reference):

- **Create Embeddings**: Generate vector embeddings from resume text.
- **Store in Vector DB**: Save embeddings to a supported vector database (e.g., Pinecone, FAISS).
- **Semantic Search**: Query the database using embeddings for fast, relevant candidate retrieval.

Refer to the `commands.txt` file for commands and usage examples related to embeddings and vector DB operations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or support, feel free to open an issue or contact [Ashmit456](https://github.com/Ashmit456).
