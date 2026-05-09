from trading_pipeline.pipeline import run_pipeline

if __name__ == "__main__":
    summary = run_pipeline()
    #summary = run_pipeline(refresh_config=True)
    print(summary)
