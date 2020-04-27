"""Testing performance issue #29.

See: https://github.com/pdpipe/pdpipe/issues/29
"""

import time

import numpy as np
import pandas as pd
import pdpipe as pdp


COLUMNS = [
    "SKC",
    "date",
    "sales",
    "sales_amount",
    "passenger_flow",
    "plus_purchase",
    "activity_level",
    "S",
    "A",
    "B",
    "N",
    "shelf_date",
    "end_of_season",
    "category_group",
    "category",
    "original_price",
    "season",
]


def _original_code():
    start = time.time()
    salesdata = pd.read_csv("processed_salesdata.csv")
    pline = pdp.PdPipeline([
        pdp.Schematize(COLUMNS),
        pdp.ApplyByCols(
            "category_group", lambda x: "tops" if x == "tops" else "other"),
        pdp.ApplyByCols(
            ["date", "shelf_date", "end_of_season"], pd.to_datetime),
        pdp.ApplyToRows(lambda row: pd.Series({
            "standard_amount": row["original_price"] * row["sales"],
            "sales_discount": 0 if (
                row["original_price"] * row["sales"] <= 0
            ) else row["sales_amount"] / (
                (row["original_price"] * row["sales"])
            ),
            "week": int(row["date"].strftime('%W')),
            "days_on_counter": (
                row["date"] - row["shelf_date"]) / np.timedelta64(1, 'D'),
            "life_cycle": (row["end_of_season"] - row["shelf_date"]) / (
                np.timedelta64(1, 'D')),
            "C1": 1 if row["category_group"] == "tops" else 0,
            "C2": 1 if row["category_group"] == "other" else 0,
            "sales": 0 if row["sales"] < 0 else row["sales"],
            "passenger_flow": 0 if row["passenger_flow"] < 0 else (
                row["passenger_flow"]),
            "plus_purchase": 0 if row["plus_purchase"] < 0 else (
                row["plus_purchase"]),
        })),
        pdp.AdHocStage(
            lambda df: df[df["days_on_counter"] <= df["life_cycle"]]),
        pdp.ColDrop("activity_level")
    ])
    salesdata = pline.apply(salesdata, verbose=True, exraise=True)

    salesdata_cumitems = salesdata[
        ["SKC", "date", "sales", "passenger_flow", "plus_purchase"]
    ].sort_values(by=["SKC", "date"]).groupby(['SKC']).cumsum()
    salesdata_cumitems.columns = [
        "total_sales", "total_passenger_flow", "total_plus_purchase"]
    salesdata["total_sales"] = salesdata_cumitems["total_sales"]
    salesdata["total_passenger_flow"] = salesdata_cumitems[
        "total_passenger_flow"]
    salesdata["total_plus_purchase"] = salesdata_cumitems[
        "total_plus_purchase"]
    print("consumed time(s)=", time.time() - start)
