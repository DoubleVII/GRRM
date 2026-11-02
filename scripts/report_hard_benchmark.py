"""Render the benchmark artifacts into a reviewable Chinese report."""
import json
from pathlib import Path


def main():
    root = Path('results/hard_pipeline_20261102')
    read = lambda path: json.loads(path.read_text())
    strict = read(root / 'strict_parsing_summary.json')
    variant = read(root / 'prompt_variants' / 'strict_parsing_summary.json')
    live = read(root / 'relaxed' / 'parsing_summary.json')
    metadata = read(root / 'metadata.json')
    lines = [
        '# Difficulty prediction pipeline 实测报告', '',
        f"测试基线：`{metadata['commit']}`，2026-11-02。", '',
        '使用真实模型执行 candidate → direct / QE prediction → OSS golden → OSS semantic matching / scoring。',
        '6 个数据集各 20 条，共 120 条；每个数据集随机 16 条，加最长原文 4 条，seed=20261102。',
        '方向覆盖 zh→en、en→zh、ja→en、de→en，最长原文 3025 字符。', '',
        f"测试模型：`{metadata['model_paths']['test']}`，cuda:1。",
        f"评估模型：`{metadata['model_paths']['evaluation']}`，cuda:1,4，tensor parallel=2。",
        'gpu_memory_utilization=0.85；max_model_len=16384；max_num_seqs=32；',
        'temperature=0，top_p=1，max_tokens=8192；retry=3 表示首轮加最多 3 次重试。OSS reasoning_effort=medium。', '',
        '## 解析结果', '',
        '下表为 JSON 解析与字段/片段校验共同通过的比例。预测和标注以 112 条有效候选为分母；',
        '候选翻译阶段另有 8 条失败，后续保留其 failed 记录并排除出有效指标。', '',
        '| 配置 / 阶段 | 首轮通过 | 最终通过 |',
        '| --- | ---: | ---: |']
    def row(label, result):
        first = result['attempts'][0]['success']
        n = result['total']
        final = result['final_success']
        lines.append(f'| {label} | {first}/{n} ({first/n:.1%}) | {final}/{n} ({final/n:.1%}) |')
    for stage, value in strict.items():
        row(f'原始严格校验 {stage}', value)
    for stage, value in live.items():
        row(f'宽松校验实跑 {stage}', value)
    lines += ['', '宽松校验前后的重新实跑会受批处理和输出波动影响；为隔离校验规则的影响，',
              '另保存了完全相同历史输出的重新校验统计：`relaxed_revalidation.json`。', '',
              '## Prompt 对照', '',
              '生产 prompt 保持原样。实验补丁完整保存在 `prompt_variants/prompt_changes.json`。',
              '候选翻译增加末尾代码块提醒；difficulty 增加短连续引用、保留大小写/换行的显式说明。',
              'difficulty 对照固定使用相同的 112 条候选，使用原始严格校验：', '',
              '| 实验 / 阶段 | 首轮通过 | 最终通过 |', '| --- | ---: | ---: |']
    for stage, value in variant.items():
        row(stage, value)
    lines += ['',
        '这些补充说明没有改善本次样本的通过率，候选翻译提醒还明显退化，因此未写入生产 prompt。',
        '原始候选翻译首轮有 18 条解析失败，最终 8 条失败；已观察的主要原因是缺失最终代码块。',
        'difficulty 严格校验失败主要是原文引用发生大小写、换行、改写或纠错差异，未观察到 JSON 语法失败或 token 截断。',
        'QE 另有语义误报：seedx_challenge_enzh row_id=16 把原文 str3aming 标为错误，',
        '而候选译文已经表达 streaming 的含义。建议后续单独对照：要求明确指出候选译文的具体错误，',
        '接受正确的源文纠错和合理替代表达，不因源文不常见或存在拼写变体就判错。此项语义 prompt 建议尚未 A/B 验证。', '',
        '## 宽松校验与部分结果保留', '',
        '修改 `utils/hard_protocol.py`：片段匹配使用 Unicode casefold、折叠空白，并兼容词内插入的换行。',
        'source_span、golden translation_span 匹配后均回填原文/候选中的实际连续子串。',
        '不放松 JSON 结构、解释非空、错误类型、重复项和语义改写校验。',
        '仅失败的 item 重试，已成功的 item 保持成功；重试耗尽的 item 保留 failed 诊断记录，',
        '评估时跳过，其他成功结果继续保存并参与统计。保留记录是为了维持 prediction / golden 的身份对齐。',
        '13 项单元测试全部通过，新增测试覆盖大小写/换行回填、重复项、真实改写拒绝，',
        '以及 instruct / OSS 两个 backend 在部分 item 重试耗尽时仍保留其他成功项。', '',
        '宽松重跑剩余失败：direct 的 wmt23_de_en row_id=418 将 fur 引成 für；',
        'QE 的 wmt23_de_en row_id=434 引用了原文不存在的长句。二者都超出大小写/换行差异范围。', '',
        '## OSS 评估', '']
    for mode in ['direct', 'qe']:
        path = root / 'relaxed' / f'{mode}.summary.json'
        if path.exists():
            overall = read(path)['overall']
            lines.append(f"- {mode}: coverage={overall['valid']}/{overall['total']} "
                         f"({overall['coverage']:.1%}), precision={overall['precision']:.4f}, "
                         f"recall={overall['recall']:.4f}, F1={overall['f1']:.4f}, "
                         f"TP/FP/FN={overall['tp']}/{overall['fp']}/{overall['fn']}。")
    lines += ['', '以上分数以本次 OSS 标注与语义匹配为参照，并非人工金标；',
              '样本刻意包含长文本，不是全量数据集的无偏质量估计。失败项不作为零分混入 F1，coverage 单列。', '',
              '原始 prompt 的 direct / QE 重复实验中，共同有效样本的原文片段集合完全一致数分别为',
              '87/108、53/107。它只衡量引用集合一致性，不等同于语义一致性。', '',
              '## 产物与复跑', '',
              '- `metadata.json`、`selection.json`：环境配置与完整样本身份。',
              '- `strict_parsing_summary.json`：原始校验、两次真实 Qwen 运行。',
              '- `inst.attempts.json`：原始所有轮次输出、finish_reason、token 数和耗时。',
              '- `prompt_variants/`：prompt 对照的完整输出与严格校验统计。',
              '- `relaxed/`：新校验规则下 Qwen 与 OSS 实跑产物、分数据集评估和解析失败。',
              '- `oss.log`、`qwen*.log`：完整模型加载和运行日志。', '',
              '```sh',
              'CUDA_VISIBLE_DEVICES=1 USE_TF=0 PYTHONPATH=. .venv/bin/python scripts/benchmark_hard_pipeline.py qwen --output <new-directory>',
              '```',
              '现有 prompt 对照和宽松重跑脚本使用本次结果目录；再次执行前应改为新目录，避免覆盖已冻结结果。',
              '原始阶段入口仍拒绝覆盖已有 artifact。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
