"""
目的: VOC 文字雲 ETL
輸出: HLH_DW.dbo.ads_survey_answers_word_segmentation, ads_survey_answers_named_entity_recognition.
"""
import pandas as pd
from ckip_transformers.nlp import CkipNerChunker, CkipPosTagger, CkipWordSegmenter
from mllib.database import connect_to_mssql
from mllib.ml_utils.utils import upload_df_to_bq
from mllib.sql_query.word_pos_ner_script import (
    create_ads_survey_answers_named_entity_recognition,
    create_ads_survey_answers_word_segmentation,
)
from prefect import flow, get_run_logger, task

from utils.gcp.client import get_bigquery_client
from utils.prefect import generate_flow_name


@task(name="with_seasonal_training")
def e_dwd_survey_answers() -> pd.DataFrame:
    """取得survey資料."""
    logging = get_run_logger()
    sql = """
    SELECT answer_id,
        answer_content
    FROM dwd_survey_answers
    WHERE question_id IN (
        '984988b1-0916-5731-aeef-85973b7f078d',
        '70d4ee9d-293d-5a8d-99ca-4b7b29e29f95',
        'e6ada374-b6e8-54b0-8a6c-ffc6d66b2808')
        AND answer_content <> ''
        AND answer_id NOT IN (
            SELECT DISTINCT answer_id
            FROM ads_survey_answers_word_segmentation
            )
    """
    logging.info("start to prepare survey_answers data...")
    with connect_to_mssql("hlh_dw") as conn:
        answers_df = pd.read_sql(sql, conn)
    return answers_df


def gen_ws_and_pos(answers_df: pd.DataFrame) -> pd.DataFrame:
    """生成斷詞和詞性標記."""
    # 數據準備
    text = answers_df["answer_content"].astype(str).tolist()

    # 初始化模型
    ws_driver = CkipWordSegmenter(model="albert-base")
    pos_driver = CkipPosTagger(model="albert-base")

    # 斷詞與詞性標註
    ws_list = ws_driver(text)
    pos_list = pos_driver(ws_list)
    answers_df["ws"] = pd.Series(ws_list)
    answers_df["pos"] = pd.Series(pos_list)
    words_ws_df = answers_df[["answer_id", "ws"]].explode("ws").reset_index(drop=True)
    words_pos_df = (
        answers_df[["answer_id", "pos"]].explode("pos").reset_index(drop=True)
    )
    words_df = words_ws_df.join(words_pos_df[["pos"]]).dropna()
    return words_df


def gen_words_and_ners(answers_df: pd.DataFrame) -> pd.DataFrame:
    """生成字詞和實體辨識名稱."""
    # 數據準備
    text = answers_df["answer_content"].astype(str).tolist()

    # 初始化模型
    ner_driver = CkipNerChunker(model="albert-base")

    # 專有名詞辨識,或名為實體辨識(Named Entity Recognition, NER)
    ner_token_list = ner_driver(text)
    words_list = [
        [ner_token.word for ner_token in ner_tokens] for ner_tokens in ner_token_list
    ]
    ners_list = [
        [ner_token.ner for ner_token in ner_tokens] for ner_tokens in ner_token_list
    ]
    answers_df["words"] = pd.Series(words_list)
    answers_df["ners"] = pd.Series(ners_list)
    ners_words_df = (
        answers_df[["answer_id", "words"]].explode("words").reset_index(drop=True)
    )
    ners_ners_df = (
        answers_df[["answer_id", "ners"]].explode("ners").reset_index(drop=True)
    )
    ners_df = ners_words_df.join(ners_ners_df[["ners"]]).dropna()
    return ners_df


@flow(name=generate_flow_name())
def words_and_ners_dwd_to_ads_daily_flow(init: bool = False) -> None:
    """Flow for words_and_ners."""
    bigquery_client = get_bigquery_client()
    answers_df = e_dwd_survey_answers()
    assess_date = pd.Timestamp.now("Asia/Taipei").tz_localize(None).strftime("%Y-%m-%d")

    if init:
        create_ads_survey_answers_word_segmentation(bigquery_client)
        create_ads_survey_answers_named_entity_recognition(bigquery_client)

    if len(answers_df) > 0:
        words_df = (
            gen_ws_and_pos(answers_df)
            .insert(loc=0, column="assess_date", value=assess_date)
        )
        ners_df = (
            gen_words_and_ners(answers_df)
            .insert(loc=0, column="assess_date", value=assess_date)
        )
        # 上傳資料到 BQ
        upload_df_to_bq(
            bigquery_client=bigquery_client,
            upload_df=words_df,
            bq_table="DS.ads_survey_answers_word_segmentation",
            bq_project="data-warehouse-369301",
            write_disposition="WRITE_APPEND",
        )
        upload_df_to_bq(
            bigquery_client=bigquery_client,
            upload_df=ners_df,
            bq_table="DS.ads_survey_answers_named_entity_recognition",
            bq_project="data-warehouse-369301",
            write_disposition="WRITE_APPEND",
        )

if __name__ == "__main__":
    words_and_ners_dwd_to_ads_daily_flow(True)
