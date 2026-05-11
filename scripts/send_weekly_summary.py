from trading_pipeline.notifications import send_weekly_summary


if __name__ == "__main__":
    sent = send_weekly_summary()
    print({"sent": sent})
