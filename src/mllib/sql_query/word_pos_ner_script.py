

def create_ads_survey_answers_word_segmentation():
    """Create DS.ads_survey_answers_word_segmentation."""
    bq_query = """
        CREATE OR REPLACE TABLE DS.ads_survey_answers_word_segmentation (
            assess_date DATE OPTIONS(description="模型執行時間")
            answer_id STRING OPTIONS(description="問題代號"),
            ws STRING OPTIONS(description="斷詞"),
            pos STRING OPTIONS(description="詞性標記"),
        )
    """
    return bq_query


def create_ads_survey_answers_named_entity_recognition():
    """Create DS.ads_survey_answers_named_entity_recognition."""
    bq_query = """
        CREATE OR REPLACE TABLE DS.ads_survey_answers_named_entity_recognition (
            assess_date DATE OPTIONS(description="模型執行時間")
            answer_id STRING OPTIONS(description="問題代號"),
            words STRING OPTIONS(description="字詞"),
            ners STRING OPTIONS(description="實體辨識,如人或物品"),
        )
    """
    return bq_query
