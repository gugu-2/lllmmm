# Features & Capabilities

The Legal "Red Flag" Scanner is an AI-powered legal assistant designed specifically for freelancers, small businesses, and agencies who frequently sign Vendor Agreements, NDAs, and Service Contracts.

## Core Features

### 1. 100% Offline & Private
Unlike using ChatGPT or Claude, this application runs entirely locally on your own hardware. 
*   **Benefit:** Zero risk of exposing sensitive, confidential, or proprietary corporate data to third-party cloud providers. It strictly adheres to NDA and privacy constraints.

### 2. Expert-Level Clause Extraction
The AI is fine-tuned on the **CUAD (Contract Understanding Atticus Dataset)**, which was annotated by real corporate lawyers and researchers from UC Berkeley.
*   **Benefit:** The AI doesn't just "guess" what a clause is; it has been explicitly trained by legal professionals to spot industry-standard legal language.

### 3. Key Risk Detection
The interface currently allows users to instantly scan for the most critical B2B contract elements:
*   **Termination Clauses:** Discover exactly how and when either party can cancel the agreement.
*   **Non-compete Clauses:** Identify restrictions on your future work or hiring practices.
*   **Governing Law:** Instantly see which state or country's laws govern the contract.
*   **Indemnification:** Highlight areas where you are taking on massive financial liability for the other party's mistakes.
*   **Liability Caps:** Check if there is a ceiling on how much you can be sued for.
*   **Exclusivity:** Ensure you aren't accidentally signing away your right to work with other clients.
*   **Audit Rights:** Find out if the client has the right to inspect your books or systems.

### 4. Hallucination Resistance
Because it is fine-tuned for *extraction* rather than *generation*, and uses strict low-temperature inference constraints, the model is highly resistant to "hallucinating" or making up legal advice. If a clause does not exist in the text, it will state "No relevant clause found."

### 5. Beautiful, Easy-to-Use UI
You do not need to be a programmer to use the final product. The Streamlit web interface provides a simple text box and dropdown menu. Just paste, select, and scan.
