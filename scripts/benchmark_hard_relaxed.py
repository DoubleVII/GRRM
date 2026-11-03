"""Live Qwen run after relaxing quote validation, with frozen baseline candidates."""
from pathlib import Path
from scripts.benchmark_hard_pipeline import QWEN, Recorder, read_stage, save_stage
from inference.run_inst_hard_prediction import predict


def main():
    from inference.run_inst_mt import init_inst_model
    root = Path('results/hard_pipeline_20261102')
    output = root / 'relaxed'
    output.mkdir(exist_ok=True)
    candidates = read_stage(root, 'candidates')
    save_stage(output, 'candidates', candidates)
    engine = init_inst_model(QWEN, gpu_memory_utilization=0.85, max_model_len=16384,
                             max_num_seqs=32, seed=20261102)
    recorder = Recorder(engine.model, output, 'inst')
    engine.model = recorder
    for mode in ['direct', 'qe']:
        recorder.start(f'{mode}.round0')
        predictions = predict(candidates, mode, QWEN, model=engine, temperature=0.9,
                              top_p=1.0, max_tokens=8192, retry=3)
        save_stage(output, 'predictions', predictions, mode)


if __name__ == '__main__':
    main()
