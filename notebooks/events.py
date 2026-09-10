import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from oxyde import db

    from eh30pt.esthealth.models import Event
    from eh30pt.esthealth.queries import select_events


@app.cell
def _():
    user = "dev"
    password = "dev"  # noqa: S105
    database = "dev"
    port = 5432

    schema = Event.Meta.schema
    url = (
        f"postgresql://{user}:{password}@localhost:{port}/{database}"
        f"?options=-c%20search_path%3D{schema}%2Cpublic"
    )
    return (url,)


@app.cell
async def _(url):
    async with db.connect(url):
        events = [event async for event in select_events(page_size=50)]
    return (events,)


@app.cell
def _(events):
    mo.ui.table(
        [
            {
                "id": event.id,
                "epi_id": event.epi_id,
                "epi_type": event.epi_type,
                "effective_time": event.effective_time,
                "specialty": event.specialty,
                "hidden": event.hidden,
                "text": (event.data or {}).get("text", ""),
            }
            for event in events
        ],
        label="Est-Health mock events",
    )
    return


if __name__ == "__main__":
    app.run()
