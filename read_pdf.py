from pypdf import PdfReader

def main():
    pdf_path = 'c:/Users/user/Desktop/RalphTradeBot/методическое-пособие.pdf'
    reader = PdfReader(pdf_path)
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        print(f"--- PAGE {i+1} ---\n{text}\n")

if __name__ == "__main__":
    main()
