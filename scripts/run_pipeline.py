from trading_pipeline.pipeline import run_full_pipeline

if __name__ == "__main__":
    summary = run_full_pipeline(refresh_config=True)
    print(summary)
