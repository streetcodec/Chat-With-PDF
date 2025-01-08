# Chat-With-PDF

## Demo :
https://drive.google.com/file/d/1Rbse1tlaqHx4JpjimpPngpk7MhUpCAV9/view?usp=drive_link
## Hosted at : 
https://chattin-pdf.streamlit.app/

This is Chat with PDF application, hosted on streamlit, which supports question answers from multiple PDFs.


Core features:

* Allow users to upload multiple PDF document.
* Extract text from the PDFs.
* Convert the text to embeddings.
* Enable users to ask questions about the document's content, with responses generated via an LLM, in this case Gemini. (Gemini-pro, temperature = 0.4)

Technical Specifications:

* Streamlit is used for interface.
* Displays the uploaded PDF’s extracted content.
* Accepts user questions and provides accurate, context-aware answers.
* Handle large PDFs efficiently by splitting them into manageable chunks for embedding and querying.

Bonus Specifications:

* The application highlights sections in the original PDF that are relevant to the answers.
* The application supports multiple PDF uploads and enable cross-document querying.
* The application is deployed to streamlit, https://chattin-pdf.streamlit.app/


## Installation

First clone the repository.

```bash
git clone https://github.com/streetcodec/Chat-With-PDF.git
```

Go to the project directory and start a virtual environment. Make sure your python version is 3.11.

```bash
cd Chat-With-PDF
python -m venv .venv
# For Linux/Mac users
source .venv/bin/activate
# For Windows users
.\.venv\Scripts\activate.bat
```

Run the following command to install the required packages.

```bash
pip install -r requirements.txt
```

## Instructions :
1) If desired answer is obtained, remove the pdf by clicking the cross, to ensure accuracy.
2) The highlighting feature works for one PDF at a time so upload one if you are looking to download the highlighted PDF. 
3) The highlighting feature will highlight many wrong instaces but the relevant part will be where the frequency of highlighted words is very high. ( Almost entire para/sentence is highlighted.)

## To dos :
1) The highlighting feature can be perfected. During data ingestion, a pdf file is split to passages and the passages' positions are calculated. The passage text and positions are saved in a search index. When a question comes, the indexer returns relevant passages which are used by the LLM to generate final responses. In the meantime, the passages used to generate such responses are highlighted in PDF files. I am not sure if it is possible in Gemini. Maybe in Claude or OpenAI. 
2) Improvement in UI, removing the sidebar.
