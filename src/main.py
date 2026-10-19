from pipeline import run_assessment

if __name__ == "__main__":

    result = run_assessment( 
        pdf_path="../PDF/Percobaan3.pdf"
    )

    print("\nHASIL AKHIR")
    print("=" * 50)

    print(
        "Status Kecukupan:",
        result["classification"]
    )

    print(
        "Assessment Score:",
        result["percentage"]
    )

    print(
        "Raw Score:",
        result["total_score"],
        "/",
        result["max_score"]
    )