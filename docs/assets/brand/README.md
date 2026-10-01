# Chelis brand art

Line art for the README and docs, drawn in the same style as chelis.ch. Each file
has literal colors, because GitHub renders README images without inheriting the
page's text color.

| File | Use |
|---|---|
| `chelis-banner-{light,dark}.svg` | README banner: a green sea turtle, kelp, and seafloor |
| `chelis-turtle-{light,dark}.svg` | The turtle alone |
| `chelis-mark-{light,dark}.svg` | The scute-shield mark, 48px and up |
| `chelis-mark-small-{light,dark}.svg` | The mark at 24 to 32px |
| `chelis-mark-16-{light,dark}.svg` | The mark's 16px bitmap |
| `chelis-mark-512.png` | The mark on its dark tile, for avatars and app icons |

`-light` files are for light backgrounds and `-dark` files for dark ones. In
Markdown, pick between them with a `<picture>` element and a
`prefers-color-scheme` source, as the root README does.

The art is generated, not hand-edited. `scripts/export_brand.py` in the chelis.ch
website repository draws it deterministically from the site's art generators
(`scripts/draw_reef.py`, seed 1729, and `scripts/draw_mark.py`). Regenerate it
there and copy the files here. The mdBook favicon in `docs/book/theme/` comes from
the same script.
