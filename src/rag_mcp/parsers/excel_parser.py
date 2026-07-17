"""Excel 矩阵序列化器。

特性:
    - 基于 pandas 将半结构化表格无损转化为元数据对
    - 格式: "列名A: 值A; 列名B: 值B;"
    - 多行合并为段 (20行/段)，减少 embedding 调用
    - 保留表格结构边界
"""

from pathlib import Path

import pandas as pd

from ..logger import get_logger

log = get_logger(__name__)

_MAX_ROW_LENGTH = 2000
_ROWS_PER_SECTION = 20  # 每段合并行数


class ExcelParser:
    """基于 pandas 的 Excel 矩阵序列化器。"""

    def parse(self, file_path: str) -> list[dict]:
        path = Path(file_path)
        if not path.exists():
            log.error("[ExcelParser] 文件不存在: {}", file_path)
            return []

        log.info("[ExcelParser] 解析: {}", path.name)

        try:
            xls = pd.ExcelFile(str(path))
        except Exception:
            log.exception("[ExcelParser] 无法打开 Excel: {}", file_path)
            return []

        sections = []
        try:
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                except Exception:
                    log.warning("[ExcelParser] Sheet '{}' 读取失败，跳过", sheet_name)
                    continue

                if df.empty:
                    continue

                df.columns = [str(c).strip() for c in df.columns]

                # 合并多行 → 减少 embedding 调用次数
                batch_texts: list[str] = []
                batch_start = 0

                for row_idx, (_, row) in enumerate(df.iterrows()):
                    pairs = []
                    for col_name in df.columns:
                        val = row[col_name]
                        if pd.isna(val):
                            continue
                        val_str = str(val).strip()
                        if val_str:
                            pairs.append(f"{col_name}: {val_str}")

                    if not pairs:
                        continue

                    text = "; ".join(pairs)
                    batch_texts.append(f"[行{row_idx + 1}] {text}")

                    if len(batch_texts) >= _ROWS_PER_SECTION:
                        merged = "\n".join(batch_texts)
                        sections.append({
                            "text": merged[:_MAX_ROW_LENGTH],
                            "source": str(path.resolve()),
                            "section": f"{sheet_name}_rows_{batch_start + 1}-{row_idx + 1}",
                            "metadata": {
                                "sheet": sheet_name,
                                "row_range": f"{batch_start + 1}-{row_idx + 1}",
                            },
                        })
                        batch_texts = []
                        batch_start = row_idx + 1

                if batch_texts:
                    merged = "\n".join(batch_texts)
                    sections.append({
                        "text": merged[:_MAX_ROW_LENGTH],
                        "source": str(path.resolve()),
                        "section": f"{sheet_name}_rows_{batch_start + 1}-{len(df)}",
                        "metadata": {
                            "sheet": sheet_name,
                            "row_range": f"{batch_start + 1}-{len(df)}",
                        },
                    })

                log.info("[ExcelParser] Sheet '{}': {} 行 → {} 段",
                         sheet_name, len(df),
                         sum(1 for s in sections
                             if s["metadata"]["sheet"] == sheet_name))
        finally:
            xls.close()

        log.info("[ExcelParser] 解析完成: {} 段 (共 {} sheets)",
                 len(sections), len(xls.sheet_names))
        return sections
