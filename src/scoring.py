from rubric import RUBRIC

# HITUNG SKOR AKHIR
def hitung_skor_akhir(assessment_results):
    total_score = 0
    max_score = 0

    # LOOP RUBRIC
    for category in RUBRIC.values():
        for subcategory_id, subcategory in (
            category["subcategories"].items()
        ):
            weight = subcategory["weight"]

            # AMBIL HASIL LLM
            assessment = assessment_results.get(
                subcategory_id
            )
            if assessment is None:
                assessment_level = None
            else:
                assessment_level = assessment.get(
                    "assessment_level"
                )

            # VALIDASI ASSESSMENT LEVEL
            if assessment_level not in (
                1,
                2,
                3
            ):
                assessment_level = None

            # HITUNG SCORE
            if assessment_level is None:
                score = 0
            else:
                score = (
                    weight *
                    assessment_level
                )

            # TOTAL SCORE
            total_score += score

            # MAXIMUM SCORE
            max_score += (
                weight * 3
            )

    # PERCENTAGE
    if max_score == 0:
        percentage = 0
    else:
        percentage = (
            total_score /
            max_score
        ) * 100

    # CLASSIFICATION
    if percentage >= 85:
        classification = "Memadai"
    elif percentage >= 60:
        classification = "Cukup Memadai"
    elif percentage >= 30:
        classification = "Kurang Memadai"
    else:
        classification = "Tidak Memadai"

    # RETURN
    return {
        "total_score":
            total_score,
        "max_score":
            max_score,
        "percentage":
            round(
                percentage,
                2
            ),
        "classification":
            classification
    }