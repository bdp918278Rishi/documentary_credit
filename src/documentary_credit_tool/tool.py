"""Tool to verify document and field validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Type
from PIL import Image
import fitz  # PyMuPDF
import os
import io
import requests
import json
import argparse
from openai import AzureOpenAI
import base64
import time
from pathlib import Path
import base64
import pandas as pd
from pydantic import BaseModel, PrivateAttr, field_validator
import shutil
from typing import List, Optional, Any

count = 0 

AZURE_OPENAI_ENDPOINT = "https://adib-azn-openai01.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_KEY = "0451276aae764a9e8509f304fcea881d"
AZURE_OPENAI_API_VERSION = "2024-10-21"

class UserParameters(BaseModel):
    dummy: Optional[str] = Field(default=None, description="No parameters needed for this tool.")


class ToolParameters(BaseModel):
    """
    Arguments of a tool call.
    """
    lc_pdf_path: str = Field(description="Path to the LC PDF file.")
    swift_pdf_path: str = Field(description="Path to the PDF file.")

def _pdf_to_images(pdf_path: str) -> List[Image.Image]:
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    pdf_document.close()
    return images

def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _call_azure_vision_api(image_base64: str, prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }

    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def _extract_required_documents(pdf_path: str) -> List[str]:
    images = _pdf_to_images(pdf_path)
    prompt = """
    As you are specialist in analysing MT format swift documents, Analyse the document,
    Recognize the required documents with specified conditions and  Give the document names listed in this SWIFT message that are explicitly required for trade.
    Look for Field 46A, 47A, and similar. Respond only with a comma-separated list.
    Example: "Commercial Invoice, Packing List"
    Do not include proforma invoice.
    If nothing is found, respond with "NONE".
    """
    results = []

    for image in images:
        img_base64 = _image_to_base64(image)
        response = _call_azure_vision_api(img_base64, prompt)
        if response.strip().upper() != "NONE":
            docs = [doc.strip() for doc in response.split(",") if doc.strip()]
            results.extend(docs)

    return list(set(results))




def validate_image_path_url(image_path_url: str) -> str:
    if image_path_url.startswith("http"):
        return image_path_url

    path = Path(image_path_url)
    if not path.exists():
        raise ValueError(f"Image file does not exist: {image_path_url}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        raise ValueError("Unsupported image format.")
    return image_path_url


def vision_tool(image_path_url) -> Any:
    """
    Executes OCR using Azure OpenAI Vision.
    """
    global count
    client = AzureOpenAI(
        api_key="0451276aae764a9e8509f304fcea881d",
        api_version="2024-10-21",
        azure_endpoint="https://adib-azn-openai01.openai.azure.com/",
    )

    try:
        image_path_url = validate_image_path_url(image_path_url)

        if image_path_url.startswith("http"):
            image_data = image_path_url
        else:
            base64_image = encode_image(image_path_url)
            image_data = f"data:image/jpeg;base64,{base64_image}"

        # Step 1: Initial OCR
        response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an OCR/ICR agent who will extract the text "
                        "in any language (e.g., English, Hindi, Chinese, Arabic etc.) "
                        "from the image, including lines, tables, stamps, seals and numbers in currencies. Keep them as they are and identify them clearly. "
                        "DO NOT TRANSLATE ON YOUR OWN. Return the result as it is in the same format."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        }
    ],
    max_tokens=4096,
)


        ocr_text = response.choices[0].message.content

        count += response.usage.total_tokens
        # Step 2: Iterative evaluation
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            eval_prompt = (
                f"You are the OCR evaluator. Review the extracted text below:\n\n{ocr_text}\n\n"
                "Check if anything is missing from the original image (text, tables, stamps, etc). "
                "If something is missing, add the missing part exactly. "
                "If the image contains unreadable or unclear text, give a reason why it cannot be extracted. "
                "If everything is extracted correctly, reply with 'COMPLETED'."
            )

            feedback_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": eval_prompt},
                            {"type": "image_url", "image_url": {"url": image_data}},
                        ],
                    }
                ],
                max_tokens=4096,
            )

            feedback = feedback_response.choices[0].message.content.strip()
            if "COMPLETED" in feedback.upper():

                count += response.usage.total_tokens
                print("Token count :::: ", count)
                return {"result": f"{ocr_text}"}
            else:
                ocr_text += f"\n\n# Correction Round {attempt}\n{feedback}"
                time.sleep(10)

        return {"result": f"⚠️ Max attempts reached. Final extracted content:\n\n{ocr_text}"}

    except Exception as e:
        return {"error": str(e)}
def validate_txt_path(text_path: str) -> str:
    with open(text_path, "r") as txt_file:
        return txt_file.read()

# lc image document extraction 

def pdf_to_img_run_tool(PdfImgToolParameters) -> Any:
    """
    Converts each page of the PDF into images and saves them to 'lcimg' directory.
    Returns the list of image file paths.
    """
    pdf_path = PdfImgToolParameters
    output_folder = "lcimg"
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    return {
        "message": f"{len(image_paths)} images generated.",
        "images": image_paths
    }

    
"This will clean and return the Required Data Eleemnts which are all Mandatory, Optional and Conditional"
def required_data_elements(doc_abbr: str) -> str:
    """
    This function reads the Data_elements.xlsx file, filters the data elements 
    for the given document abbreviation (e.g., 'INV'), and returns a formatted 
    string containing mandatory, optional, and conditional fields.
    """
    # Load Excel file with second row as header

    elements_xlsx_path = Path("Data_elements.xlsx")
    if not elements_xlsx_path.is_absolute():
        elements_xlsx_path = Path(__file__).parent / elements_xlsx_path
    elements_xlsx_path = elements_xlsx_path.resolve()
    df = pd.read_excel(elements_xlsx_path, header=1)

    # Strip column names of leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # Drop irrelevant columns if present
    for col in ["Unnamed: 0", "UID"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Clean doc_abbr and check if the column exists
    doc_abbr = doc_abbr.strip()
    print(doc_abbr)
    required_columns = ['Data element', 'Description', doc_abbr]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # Filter only required columns
    df = df.loc[:, required_columns]

    # Define field types
    required_fields = ["M", "O", "C"]

    # Filter rows where the field is one of M, O, C
    df_filter = df[df[doc_abbr].isin(required_fields)]

    # Sort by field type (M, O, C)
    df_filter = df_filter.sort_values(by=doc_abbr)

    # Group results
    groups = {"M": "**Mandatory fields:**\n", "O": "**Optional fields:**\n", "C": "**Conditional fields:**\n"}

    for label in required_fields:
        subset = df_filter[df_filter[doc_abbr] == label]
        for _, row in subset.iterrows():
            name = row['Data element']
            desc = str(row['Description']).strip()
            if desc and desc.lower() != name.lower():
                groups[label] += f"- {name} (description='{desc}')\n"
            else:
                groups[label] += f"- {name}\n"

    # Combine result string
    data_element_result = groups["M"] + "\n" + groups["O"]
    if "C" in df_filter[doc_abbr].values:
        data_element_result += "\n" + groups["C"]

    return data_element_result

class SWIFTClauseExtractorTool():
    name: str = "SWIFT Clause Extractor Tool"
    description: str = "Summarizes key clause requirements for a given document type using a SWIFT reference document"

    _client: AzureOpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        self._client = AzureOpenAI(
            api_version="2024-10-21",
            azure_endpoint="https://adib-azn-openai01.openai.azure.com/",
            api_key="0451276aae764a9e8509f304fcea881d",
        )
        self._deployment = "gpt-4o"

    def _run(self, **kwargs) -> str:
        doc_type = kwargs.get("document_type")
        file_path = kwargs.get("swift_clauses_path")

        # Automatically find the file if not specified
        if not file_path:
            extracted_dir = Path("./extracted_text")
            txt_files = sorted(extracted_dir.glob("*.txt"))
            if not txt_files:
                return "No .txt files found in ./extracted_text"
            file_path = txt_files[0]  # Pick the first file
        else:
            file_path = Path(file_path)

        try:
            clauses = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading SWIFT text file: {str(e)}"

        try:
            system_msg = (
                "You are an expert in trade finance compliance and SWIFT messaging rules. "
                "Based on the following content from a SWIFT standards document, "
                "extract the specific clauses or message content that are relevant to the given document type."
            )

            user_msg = f"""
Document Type: {doc_type}

SWIFT Clauses:
{clauses}

Summarize the key points or requirements that apply to the above document type.
"""

            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=1024,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error during GPT summarization: {e}"

class IdentifyDocumentTool():
    name: str = "Document Identifier Tool"
    description: str = "Identifies trade document type using Azure GPT-4o Vision model and Excel reference"

    _client: AzureOpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        self._client = AzureOpenAI(
            api_version="2024-10-21",
            azure_endpoint="https://adib-azn-openai01.openai.azure.com/",
            api_key="0451276aae764a9e8509f304fcea881d",
        )
        self._deployment = "gpt-4o"

    def _run(self, **kwargs) -> tuple[str,str]:
        image_path = kwargs.get("image_path_url")
        excel_path = kwargs.get("excel_path")

        try:
            df = pd.read_excel(excel_path)
            documents = df.to_dict(orient='records')
            doc_list_str = "\n".join(
                [f"{row['Document name']} ({row['Code']})" for row in documents]
            )
        except Exception as e:
            return f"Error loading Excel: {e}"

        try:
            if image_path.startswith("http"):
                image_data = image_path
            else:
                image_data = f"data:image/jpeg;base64,{self._encode_image(image_path)}"
        except Exception as e:
            return f"Error reading image: {e}"

        try:
            user_prompt = f"""
 As a trade and Letter of Credit expert, Be cautious in IDENTIFYING the document,
    Identify the type of document name if there is document name specified on the document explicitly or
    based on layout, headers, and content.
    Use the following types for guidance:\n{doc_list_str}
    Respond with the document name or 'UNKNOWN'.
    Example: "Commercial Invoice"
Return the best match in the format:
[Document Name, Code]

If you are unsure, cannot confidently identify the document: 
[Unrecognized, Unrecognized]

Do not add any explanations or comments.
"""


            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.2,
            )

            tolist = response.choices[0].message.content.strip("[]").split(",")
            print("DOC_CLASSIFIER :::: ",user_prompt,tolist)
            return tolist
            

        except Exception as e:
            return f"Error during classification: {e}"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

def _identify_documents_in_package(pdf_path: str) -> List[str]:
    images = _pdf_to_images(pdf_path)
    format_xlsx_path = Path("Format.xlsx")
    if not format_xlsx_path.is_absolute():
        format_xlsx_path = Path(__file__).parent / format_xlsx_path
    format_xlsx_path = format_xlsx_path.resolve()
    df = pd.read_excel(format_xlsx_path)
    documents = df.to_dict(orient='records')
    document_names = "\n".join([f"{row['Document name']} ({row['Code']})" for row in documents])

    prompt = f"""
    As a trade and Letter of Credit expert, Be cautious in IDENTIFYING the document,
    Identify the type of document name if there is document name specified on the document explicitly or
    based on layout, headers, and content.
    Use the following types for guidance:\n{document_names}
    Respond with the document name or 'UNKNOWN'.
    Example: "Commercial Invoice"
    """

    found_docs = []
    for i, image in enumerate(images):
        img_base64 = _image_to_base64(image)
        response = _call_azure_vision_api(img_base64, prompt)
        doc_type = response.strip()
        if doc_type.upper() != "UNKNOWN":
            found_docs.append(f"{doc_type} (Page {i+1})")

    return found_docs

def _llm_document_match(required_doc: str, found_doc: str) -> bool:
    prompt = f"""
You are a trade document compliance specialist. Determine whether these two document names refer to the same trade document type.

Required Document: "{required_doc}"
Found Document: "{found_doc}"

Respond with only "Yes" or "No".
"""
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 5
    }

    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"].strip().lower()
    return reply.startswith("yes")

def _compare_documents(required_docs: List[str], found_docs: List[str]) -> dict:
    def normalize(text: str) -> str:
        return text.strip().strip('"').strip("'")

    clean_required = [normalize(doc) for doc in required_docs if normalize(doc).lower() not in {"none", ""}]
    matched = []
    missing = []

    for req in clean_required:
        match_found = False
        for fd in found_docs:
            if _llm_document_match(req, fd.split(" (Page")[0]):
                matched.append(f"{req} ✓ (LLM matched: {fd})")
                match_found = True
                break
        if not match_found:
            missing.append(req)

    return {
        "required_documents": [doc.title() for doc in clean_required],
        "found_documents": found_docs,
        "missing_documents": missing,
        "matched_documents": matched,
        "summary": {
            "required": len(clean_required),
            "found": len(found_docs),
            "matched": len(matched),
            "missing": len(missing),
            "match_rate_percent": round(len(matched) / len(clean_required) * 100, 1) if clean_required else 0
        }
    }



def Discrepancy_run_tool(Discrepancy_ToolParameters) -> Any:
    """
    Executes OCR using Azure OpenAI Vision.
    """

    format_xlsx_path = Path("Format.xlsx")
    if not format_xlsx_path.is_absolute():
        format_xlsx_path = Path(__file__).parent / format_xlsx_path
    format_xlsx_path = format_xlsx_path.resolve()
    
    identify_doc_tool = IdentifyDocumentTool()
    doc_type = identify_doc_tool._run(
        image_path_url=Discrepancy_ToolParameters,
        excel_path=format_xlsx_path
    )

    if isinstance(doc_type, str):  # in case an error string is returned
        raise ValueError(f"Document classification failed: {doc_type}")

    doc_name = doc_type[0].strip()
    doc_abbr = doc_type[1].strip()
        # Extract image name from path/URL
    image_name = os.path.splitext(os.path.basename(Discrepancy_ToolParameters))[0]

    # Load the document reference from Excel to verify match
    df_ref = pd.read_excel(format_xlsx_path)
    doc_refs = {(str(row['Document name']).strip(), str(row['Code']).strip()) for _, row in df_ref.iterrows()}
    
    if (doc_name, doc_abbr) not in doc_refs:
        os.makedirs("NonConforming", exist_ok=True)
        with open(f"NonConforming/unidentified_doc.txt", "a", encoding="utf-8") as f:
            f.write(f"[{doc_name}, {doc_abbr}, {os.path.basename(Discrepancy_ToolParameters)}]\n")
        return {"skipped": True, "reason": "Unrecognized document type"}

    output_txt_path = f"discrepancy_individual/Discrepancy {doc_name}{image_name}.txt"
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    data_element_result = required_data_elements(doc_abbr)
    print(data_element_result)
    swift_clause_tool = SWIFTClauseExtractorTool()

    swift_clause_result = swift_clause_tool._run(
        document_type=doc_name
    )
    print("swift_clause_result ::::", swift_clause_result)
    client = AzureOpenAI(
        api_key="0451276aae764a9e8509f304fcea881d",
        api_version="2024-10-21",
        azure_endpoint="https://adib-azn-openai01.openai.azure.com/",
    )

    try:
        image_path_url = validate_image_path_url(Discrepancy_ToolParameters)

        if image_path_url.startswith("http"):
            image_data = image_path_url
        else:
            base64_image = encode_image(image_path_url)
            image_data = f"data:image/jpeg;base64,{base64_image}"
   
        # Step 1: Initial OCR
        response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": (
                (
    f"You are an {doc_name} analyzer. Given the image of a document, your task is to extract and analyze whether all required {doc_name} fields are present.\n\n"
    f"You will also validate the {doc_name} against the condition defined in the corresponding SWIFT message.\n\n"
    f"{data_element_result}\n"
    "✅ Your job is to:\n"
    "1. Identify these fields (even if the name used in the document is different but means the same).\n"
    "2. List which mandatory fields are missing and why (e.g., not found, unclear).\n"
    "3. List the missing optional fields (if any), and list the missing conditional fields (if applicable).\n"
    "4. You are a super intelligent assistant. Understand the given image and do not hallucinate values. Provide values exactly as they appear. Pay special attention to countries, dates, LC numbers, and places.\n"
    f"5. Perform **Discrepancy Detection** by comparing extracted data with the {doc_name} and {swift_clause_result} with its conditions:\n"
    "   - Flag discrepancies such as:\n"
    "     1. Mismatched amounts.\n"
    "     2. Expiry date violations.\n"
    "     3.. Non-compliance with shipping terms.\n"
    "   - Categorize discrepancies by severity (e.g., Critical, Minor).\n"
    "   - Explain each discrepancy clearly.\n"
    "7. If everything is present and clear, say: '✅ All mandatory fields are present. Document is good.'\n\n"
    "Return the extracted fields as structured markdown in this format:\n"
    "### Mandatory Fields:\n- Field Name: Value (or '❌ Missing')\n...\n\n"
    "### Optional Fields:\n- Field Name: Value (or '❌ Missing')\n...\n\n"
    "### Swift Fields:\n- Field Name: Value (or '❌ Missing')\n...\n\n"
    "### Discrepancy Summary:\n- Discrepancy: Description (Severity: Critical/Minor)\n- ...\n"
)

            ),
        },
        {"type": "image_url", "image_url": {"url": image_data}},
    ],
}

    ],
    max_tokens=4096,
)

        ocr_text = response.choices[0].message.content
        with open(output_txt_path, "w", encoding="utf-8") as out_file:
            out_file.write(str(ocr_text) + "\n")
        return {"result": f"{ocr_text}"}

    except Exception as e:
        return {"error": str(e)}


def run_tool(config: UserParameters, args: ToolParameters) -> Any:
    """
    Converts each page of the PDF into images and saves them to 'img' directory.
    Returns the list of image file paths.
    """
    folders_to_remove = ['img', 'lcimg', 'NonConforming', 'extracted_text', 'discrepancy_individual']
    import shutil
    for folder in folders_to_remove:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"Removed folder: {folder}")
        else:
            print(f"Folder not found (skipped): {folder}")

    # Assume the user provides only the file name
    file_name = args.swift_pdf_path  # e.g., "ILCAE00221000098-1.pdf"
    
    # Define the base folder where swift_data lives
    swift_base_folder = Path(__file__).parent / "swift_data"

    swift_pdf_path = Path(args.swift_pdf_path)
    if not swift_pdf_path.is_absolute():
       swift_pdf_path = Path(__file__).parent / swift_pdf_path
    swift_pdf_path = swift_pdf_path.resolve()
    required = _extract_required_documents(swift_pdf_path)
    output_folder = Path("img")  # Convert string to Path
    output_folder.mkdir(exist_ok=True)
    
    # Delete all files (and subfolders) inside the img folder
    for item in output_folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    print("SWIFT_VISION : ",swift_pdf_path)        
    doc = fitz.open(swift_pdf_path)
    image_paths = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path) 

    # Prepare output file
    extracted_text_folder_name = "extracted_text"
    os.makedirs(extracted_text_folder_name, exist_ok=True)
    output_txt_path = os.path.join(extracted_text_folder_name, "extracted.txt")
    
    with open(output_txt_path, "w", encoding="utf-8") as out_file:
        # Sort filenames to ensure correct order: page_1.png, page_2.png, ...
        for idx, image_file in enumerate(sorted(os.listdir(output_folder))):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(output_folder, image_file)
                print(f"Processing: {image_path}")

                result = vision_tool(image_path)
                text = result.get("result") or result.get("error", "[No result]")
                out_file.write(str(text) + "\n")

    print("TOTAL Count", count)
    extracted_text_folder_path = os.path.abspath("extracted_text")
    discrepancy_individual_folder_path = os.path.abspath("discrepancy_individual")
    os.makedirs("Missing_document", exist_ok=True)
    missing_document_folder_path = os.path.abspath("Missing_document")
    # User provides only the file name (e.g., "ILCAE00221000098-2-single-page.pdf")
    lc_pdf_path = Path(args.lc_pdf_path)
    lc_number = lc_pdf_path.stem.split("-")[0]
    print(lc_number)
    if not lc_pdf_path.is_absolute():
       lc_pdf_path = Path(__file__).parent / lc_pdf_path
    lc_pdf_path = lc_pdf_path.resolve()
    found = _identify_documents_in_package(lc_pdf_path)
    result = _compare_documents(required, found)
    lc_pdf_to_img_result = pdf_to_img_run_tool(PdfImgToolParameters=lc_pdf_path)
    image_folder_lc = Path("lcimg").resolve()

    for idx, image_file in enumerate(sorted(os.listdir(image_folder_lc))):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder_lc, image_file)
            # print(f"Processing: {image_path}")
    
            discrepancy_result = Discrepancy_run_tool(Discrepancy_ToolParameters  = image_path)
            # print(result)
            if discrepancy_result is None or discrepancy_result.get("skipped", False):
                print(f"Skipped processing {image_path}")
                continue
    comparison_txt_path = os.path.join(missing_document_folder_path, "Missing Document.txt")
    with open(comparison_txt_path , "w", encoding="utf-8") as f:
        f.write("📄 SWIFT Document Analyzer Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"📌 LC Number: {lc_number}\n\n")

        f.write("🧾 Required Documents:\n")
        for doc in result["required_documents"]:
            f.write(f"  - {doc}\n")

        f.write("\n📂 Found Documents in Package:\n")
        for doc in result["found_documents"]:
            f.write(f"  - {doc}\n")

        f.write("\n✅ Matched Documents:\n")
        for doc in result["matched_documents"]:
            f.write(f"  - {doc}\n")

        f.write("\n❌ Missing Documents:\n")
        for doc in result["missing_documents"]:
            f.write(f"  - {doc}\n")

        f.write("\n📊 Summary:\n")
        for key, value in result["summary"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

        f.write("\n" + "=" * 40 + "\n")

    print(f"\n📁 Result saved to: {output_txt_path }")
    
    print("extracted_text_folder_path :::: ", extracted_text_folder_path)
    print("discrepancy_individual_folder_path :::: ", discrepancy_individual_folder_path)
    
    return {
        "status": "completed",
        "extracted_text_folder_path": extracted_text_folder_path,
        "discrepancy_individual_folder_path":discrepancy_individual_folder_path,
        "missing_document":result
        
    }


OUTPUT_KEY = "tool_output"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-params", required=True)
    parser.add_argument("--tool-params", required=True)
    args = parser.parse_args()

    user_dict = json.loads(args.user_params)
    tool_dict = json.loads(args.tool_params)

    config = UserParameters(**user_dict)
    params = ToolParameters(**tool_dict)

    output = run_tool(config, params)
    print(OUTPUT_KEY, output)