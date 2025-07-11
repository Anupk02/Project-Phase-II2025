# 🔍 Dual-Mode Plagiarism Detection System (Text + Image + PDF)

This project is an advanced **web-based plagiarism detection system** that supports:

- 📄 **Text and PDF** content checking
- 🖼️ **Image plagiarism** detection using visual similarity
- 🔁 Smart **web crawling** to identify copied or paraphrased content
- 📊 Detailed reporting with source breakdown and percentage scoring
- 📥 Real-time **upload support** for `.txt`, `.pdf`, and `.docx` files

---

## 🚀 Features

| Capability                      | Description                                                   |
|----------------------------------|---------------------------------------------------------------|
| ✅ Text-based detection          | Checks entire document using sentence chunking and crawling   |
| ✅ PDF support                   | Extracts both text and images from uploaded PDFs              |
| ✅ Image-based detection         | Detects visual similarity using ORB and SSIM methods          |
| ✅ Full-text web crawling        | Compares content beyond just previews/snippets                |
| ✅ Parallel processing           | Increases speed using multithreading                          |
| ✅ Plagiarism report             | Detailed breakdown with % scores and sources                  |
| ✅ Caching                       | Saves previous results to avoid redundant processing          |
| ✅ QR and chart integration      | PDF reports include QR codes and pie charts                   |

---

## 📁 Project Structure

├── app.py # Flask backend server
├── plagiarism_detector.py # Text plagiarism logic
├── search_cache.json # Cached search results
├── plagiarism_log.txt # Detailed logs of activity
├── requirements.txt # Python dependencies
├── static/ # CSS, JS, uploads, chart outputs
└── templates/ index.html  # Web UI frontend

---

## 🛠️ Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/plagiarism-detector
cd plagiarism-detector
2. Install required packages
Make sure you have Python 3.8+ installed.

bash
Copy
Edit
pip install -r requirements.txt
3. Run the Flask app
bash
Copy
Edit
python app.py
Then visit: http://localhost:5000
🔍 How It Works
The system crawls public web sources to find matching content.

For PDFs, it extracts text and embedded images, and checks both.

For images, it uses ORB keypoint detection and SSIM comparison to identify duplication.

It highlights all matching sources and shows plagiarism % in a clean, visual layout.

You can download a PDF report with pie chart and source list.

📊 Output Details
✅ % of content matched

✅ Sources of match (link + snippet)

✅ Side-by-side matched text

✅ Pie chart for visual display

✅ Downloadable PDF with QR code to result

🛡 Notes
This project uses intelligent web crawling, not public APIs.

Internet access is required to perform live similarity checking.

You may need to handle User-Agent headers or bypass captchas if targeting protected sites.

🧠 Technologies Used
Flask for backend and routing

NLTK for sentence tokenization

PyMuPDF for PDF extraction

OpenCV, scikit-image for image similarity

matplotlib, reportlab for visual reports

BeautifulSoup, requests for crawling

📄 License
This project is released under the MIT License.

Built for research, education, and academic integrity.

yaml
Copy
Edit

---

Let me know if you'd like:
- 🐳 A `Dockerfile`
- 🌍 A hosted deployment setup (Render, Heroku, AWS)
- 📥 Admin dashboard to track user uploads

You're ready to launch!