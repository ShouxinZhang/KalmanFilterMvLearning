# Visual Debugging

## Symptoms and fixes

### You are tempted to debug inside the main PDF

- Stop.
- Copy the figure into a scratch file.
- Render only that scratch file.
- Return to the integrated PDF only after the single figure is already clean.

### Text overlaps inside nodes

- Make labels shorter.
- Increase node distance before increasing font size.
- Use `\shortstack{...}` for multi-line text.
- Move long prose to caption.

### Edge labels collide with nodes

- Remove the label first and verify geometry.
- Re-add labels on fewer edges.
- Prefer one explanatory label per flow, not per arrow.

### Too many edge crossings

- Split the figure into lanes: input lane, state lane, output lane.
- Route skip connections as dashed arcs or outer paths.
- If a block diagram still crosses badly, separate it into two smaller figures.

### Background box hides content

- Add background `fit` nodes only in `on background layer`.
- Keep the background box lighter than the foreground.
- Fit the smallest meaningful group, not the whole diagram by default.

### Isolated render looks fine but PDF page looks wrong

- Inspect the full document page with `render_pdf_page.py`.
- Check surrounding figure width, scaling, and caption spacing.
- Verify that the integrated document uses the same preamble styles as the isolated render.
- If the integrated page is wrong, treat it as an integration problem, not a figure-design problem.

## Minimum acceptance bar

A TikZ figure is not done until:

- the node labels are readable without zooming,
- the main information flow is visually obvious,
- the image survives raster inspection,
- and the integrated PDF page still looks correct.
