"""Stream games from the public Lichess database.

The monthly dumps are large (tens of GB compressed), but they are a single zstd stream,
so we decompress *incrementally* and stop as soon as we have enough games. Capping the
game count therefore only pulls a small compressed prefix over the network — a few MB
buys thousands of games — no full download or extraction.

URL pattern:
    https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst
"""

from __future__ import annotations

import io
import urllib.request

import zstandard

LICHESS_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)


def lichess_pgn_stream(month: str, timeout: int = 60) -> io.TextIOBase:
    """Return a decompressed PGN *text* stream for a Lichess month (e.g. "2016-04").

    The stream is lazy: bytes are fetched and decompressed only as they are read, so a
    downstream reader that stops early (a game cap) never downloads the whole file.
    """
    url = LICHESS_URL.format(month=month)
    request = urllib.request.Request(url, headers={"User-Agent": "cwm/0.1"})
    response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 (fixed host)
    dctx = zstandard.ZstdDecompressor()
    binary = dctx.stream_reader(response)
    return io.TextIOWrapper(binary, encoding="utf-8", errors="ignore")
