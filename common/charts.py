import plotly.graph_objects as go


def line_chart(frame, x, y, title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame[x], y=frame[y], mode="lines", name=y))
    fig.update_layout(title=title, template="plotly_white")
    return fig
