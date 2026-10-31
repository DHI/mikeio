"""View, trim and export a dfs0 file in the browser."""
# /// script
# requires-python = "==3.12.*"
# dependencies = ["mikeio==3.3.0", "streamlit==1.66.0"]
# [tool.uv]
# exclude-newer = "2026-10-18T00:00:00Z"
# ///

import sys
import tempfile
from pathlib import Path

import mikeio
import numpy as np
import pandas as pd
import streamlit as st


def demo_dataset() -> mikeio.Dataset:
    """Two weeks of synthetic 10-minute data: M2 + S2 tide plus a surge event."""
    time = pd.date_range("2026-01-01", periods=14 * 24 * 6, freq="10min")
    hours = (time - time[0]).total_seconds().to_numpy() / 3600
    m2 = 0.9 * np.cos(2 * np.pi * hours / 12.42)
    s2 = 0.3 * np.cos(2 * np.pi * hours / 12.00)
    surge = 0.6 * np.exp(-(((hours - 200) / 18) ** 2))
    level = m2 + s2 + surge
    # Tidal current leads the level by a quarter period; U before V.
    u = 0.8 * np.gradient(m2 + s2, hours)
    v = 0.3 * u + 0.05 * np.random.default_rng(1).standard_normal(len(time))
    items = [
        ("Water level", mikeio.EUMType.Water_Level, level),
        ("U velocity", mikeio.EUMType.u_velocity_component, u),
        ("V velocity", mikeio.EUMType.v_velocity_component, v),
    ]
    return mikeio.Dataset(
        [
            mikeio.DataArray(data, time=time, item=mikeio.ItemInfo(name, eum))
            for name, eum, data in items
        ]
    )


@st.cache_data
def read_dfs0(content: bytes) -> mikeio.Dataset:
    """Read an uploaded dfs0 file."""
    # mikeio reads from a path, so park the upload in a temp file.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "upload.dfs0"
        path.write_bytes(content)
        return mikeio.read(path)


def to_dfs0_bytes(ds: mikeio.Dataset) -> bytes:
    """Write a dataset to dfs0 and return the file content."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "out.dfs0"
        ds.to_dfs(path)
        return path.read_bytes()


def main() -> None:
    """Lay out the app."""
    st.set_page_config(page_title="dfs0 explorer", layout="wide")
    st.title("dfs0 explorer")

    upload = st.sidebar.file_uploader("dfs0 file", type="dfs0")
    if upload is None:
        st.sidebar.caption("No file uploaded: showing a synthetic tide station.")
        ds = demo_dataset()
        name = "demo"
    else:
        ds = read_dfs0(upload.getvalue())
        name = Path(upload.name).stem

    st.subheader("Items")
    st.dataframe(
        pd.DataFrame(
            {
                "Item": [i.name for i in ds.items],
                "Type": [i.type.display_name for i in ds.items],
                "Unit": [i.unit.display_name for i in ds.items],
            }
        ),
        hide_index=True,
    )
    st.caption(f"{ds.n_timesteps} timesteps, {ds.time[0]} to {ds.time[-1]}")

    names = [i.name for i in ds.items]
    selected = st.sidebar.multiselect("Items to plot", names, default=names[:1])
    start, end = st.sidebar.slider(
        "Time window",
        min_value=ds.time[0].to_pydatetime(),
        max_value=ds.time[-1].to_pydatetime(),
        value=(ds.time[0].to_pydatetime(), ds.time[-1].to_pydatetime()),
        format="YYYY-MM-DD HH:mm",
    )
    if not selected:
        st.info("Pick at least one Item in the sidebar.")
        return

    sub = ds[selected].sel(time=slice(start, end))
    df = sub.to_dataframe()

    st.subheader("Time series")
    st.line_chart(df)

    st.subheader("Statistics")
    st.dataframe(df.describe().T)

    st.subheader("Export the selection")
    col1, col2 = st.columns(2)
    col1.download_button("dfs0", to_dfs0_bytes(sub), file_name=f"{name}_subset.dfs0")
    col2.download_button("CSV", df.to_csv().encode(), file_name=f"{name}_subset.csv")


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        # Started with `uv run dfs0_app.py`: hand over to `streamlit run` on this file.
        from streamlit.web import cli

        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(cli.main())
