from pydantic import BaseModel, ConfigDict, Field


class NlpSettings(BaseModel):
    """NPL：专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化 reranker，检索直接使用融合打分结果。
    enable: bool = Field(default=False, description="是否启用NLP")

    nlp_name: str = Field(default="zh_core_web_md", description="nlp模型名称")