import unittest
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO

import pandas as pd


def _sample_kline(rows=70):
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": [10 + i * 0.1 for i in range(rows)],
        "high": [11 + i * 0.1 for i in range(rows)],
        "low": [9 + i * 0.1 for i in range(rows)],
        "close": [10.5 + i * 0.1 for i in range(rows)],
        "volume": [1000 + i for i in range(rows)],
    })


class StabilityV111Tests(unittest.TestCase):
    def test_fetch_kline_retries_and_honors_end_date(self):
        import chan_lib.data as data

        calls = []
        original_fetch = data._fetch_baostock
        original_sleep = data.time.sleep

        def fake_fetch(symbol, period, adjust, start_date, end_date):
            calls.append(end_date)
            if len(calls) < 3:
                raise RuntimeError("temporary network failure")
            return _sample_kline(rows=5)

        data._fetch_baostock = fake_fetch
        data.time.sleep = lambda seconds: None
        try:
            df = data.fetch_kline(
                "601689",
                period="daily",
                start_date="20200101",
                end_date="20200103",
                use_cache=False,
                max_retries=3,
            )
        finally:
            data._fetch_baostock = original_fetch
            data.time.sleep = original_sleep

        self.assertEqual(calls, ["20200103", "20200103", "20200103"])
        self.assertEqual(len(df), 3)
        self.assertEqual(df["date"].max(), pd.Timestamp("2020-01-03"))

    def test_fetch_kline_limits_retries_when_stale_cache_exists(self):
        import chan_lib.data as data

        calls = []
        stale = _sample_kline(rows=5)
        original_fetch = data._fetch_baostock
        original_read_cache = data._read_cache
        original_cache_age = data._cache_age
        original_sleep = data.time.sleep

        def fake_fetch(symbol, period, adjust, start_date, end_date):
            calls.append(end_date)
            raise RuntimeError("network unavailable")

        data._fetch_baostock = fake_fetch
        data._read_cache = lambda symbol, period, adjust="qfq": stale
        data._cache_age = lambda symbol, period, adjust="qfq": data.TTL_MAP["daily"] + 1
        data.time.sleep = lambda seconds: None
        try:
            with redirect_stdout(StringIO()):
                df = data.fetch_kline(
                    "601689",
                    period="daily",
                    start_date="20200101",
                    end_date="20200103",
                    max_retries=5,
                )
        finally:
            data._fetch_baostock = original_fetch
            data._read_cache = original_read_cache
            data._cache_age = original_cache_age
            data.time.sleep = original_sleep

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(df), 3)
        self.assertEqual(df["date"].max(), pd.Timestamp("2020-01-03"))

    def test_analyze_stock_forwards_end_date_to_data_fetch(self):
        import chan_analyzer

        captured = {}
        original_fetch = chan_analyzer.fetch_kline

        def fake_fetch(symbol, period="daily", start_date="20200101", end_date=None, **kwargs):
            captured["end_date"] = end_date
            df = _sample_kline(rows=70)
            df["date"] = pd.to_datetime(df["date"])
            if end_date:
                df = df[df["date"] <= pd.Timestamp(datetime.strptime(end_date, "%Y%m%d"))]
                df.reset_index(drop=True, inplace=True)
            return df

        chan_analyzer.fetch_kline = fake_fetch
        try:
            result = chan_analyzer.analyze_stock(
                "601689",
                "拓普集团",
                quiet=True,
                start_date="20200101",
                end_date="20200305",
            )
        finally:
            chan_analyzer.fetch_kline = original_fetch

        self.assertEqual(captured["end_date"], "20200305")
        self.assertTrue(result)
        self.assertLessEqual(result["_raw_df"]["date"].max(), pd.Timestamp("2020-03-05"))

    def test_multi_timeframe_default_includes_monthly(self):
        import chan_analyzer

        captured = {}
        original_fetch_multi = chan_analyzer.fetch_multi_timeframe

        def fake_fetch_multi(symbol, periods=None, start_date="20200101", end_date=None, **kwargs):
            captured["periods"] = list(periods)
            return {}

        chan_analyzer.fetch_multi_timeframe = fake_fetch_multi
        try:
            chan_analyzer.analyze_multi_timeframe("601689", "拓普集团", quiet=True)
        finally:
            chan_analyzer.fetch_multi_timeframe = original_fetch_multi

        self.assertEqual(captured["periods"], ["30", "daily", "weekly", "monthly"])

    def test_fetch_multi_timeframe_passes_end_date_and_uses_short_retries(self):
        import chan_lib.data as data

        calls = []
        original_fetch = data.fetch_kline

        def fake_fetch(symbol, period="daily", start_date="20200101",
                       end_date=None, max_retries=5, **kwargs):
            calls.append({
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "max_retries": max_retries,
            })
            df = _sample_kline(rows=70)
            df["date"] = pd.to_datetime(df["date"])
            return df

        data.fetch_kline = fake_fetch
        try:
            result = data.fetch_multi_timeframe(
                "601689",
                periods=["daily"],
                start_date="20200101",
                end_date="20200305",
            )
        finally:
            data.fetch_kline = original_fetch

        self.assertEqual(calls[0]["end_date"], "20200305")
        self.assertEqual(calls[0]["max_retries"], 2)
        self.assertIn("daily", result)


if __name__ == "__main__":
    unittest.main()
