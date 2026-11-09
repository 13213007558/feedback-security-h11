# Regression tests for request/response message boundaries under arbitrary
# network fragmentation. h11 is an in-memory state machine, so the way a byte
# stream is split into recv() calls must not change where one message ends,
# which headers/body/trailer bytes belong to which message, or whether a
# corrupted body gets silently re-interpreted as the start of the next request.
#
# Every corpus below carries hand-written expectations derived from RFC 7230
# (and the project's documented compatibility policy); they are not snapshots
# of whatever the current parser happened to emit. Allowed combinations list
# the exact events they must produce; rejected combinations assert that the
# peer side of the connection lands in the ERROR state and that no second
# request is ever delivered after the failure.
import random
from dataclasses import dataclass, field
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Type

import pytest

from .._connection import Connection, NEED_DATA, PAUSED
from .._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from .._state import CLIENT, DONE, ERROR, SERVER, SWITCHED_PROTOCOL
from .._util import RemoteProtocolError
from .helpers import normalize_data_events

# Fixed upper bounds so the suite stays a fast, deterministic set of short
# messages rather than an unbounded fuzzing campaign.
MULTI_CUT_SCHEDULES_PER_CORPUS = 12
MULTI_CUT_RNG_SEED = 20261109
# Hard ceiling on the number of fragmented executions across the whole module,
# guarding against accidental corpus growth making the suite explode.
MAX_TOTAL_EXECUTIONS = 5000


@dataclass
class Corpus:
    name: str
    # Which side of the connection we are driving; the wire bytes are what
    # that side receives from its peer.
    role: Type[Any]
    wire: bytes
    # Hand-written expectation for every event delivered up to the terminal
    # point of the corpus.
    expected_events: List[Event]
    # When True, b""" is fed after the wire bytes to signal peer EOF.
    eof: bool = False
    # For corpora that expect RemoteProtocolError: the exact event prefix that
    # must have been delivered before the error was raised.
    expect_error: bool = False
    events_before_error: List[Event] = field(default_factory=list)
    # CLIENT corpora first have to emit a request before they can receive a
    # response. These events are sent (not received).
    initial_send: Optional[List[Event]] = None
    # Called whenever the receive side reports PAUSED while more of the wire
    # remains to be processed. This is the only legal place to advance the
    # conversation: send the scripted response and/or start the next cycle.
    # It must never rebuild the Connection.
    on_paused: Optional[Callable[[Connection], None]] = None
    # Called once all wire bytes (and optional EOF) have been driven through.
    after: Optional[Callable[[Connection], None]] = None


#
# Server-side corpora: the wire bytes are requests received by a SERVER
# connection.
#


def _respond_simple(conn: Connection) -> None:
    # A minimal keep-alive response that lets the completed request cycle be
    # reset with start_next_cycle() without rebuilding the connection.
    conn.send(Response(status_code=200, headers=[]))
    conn.send(EndOfMessage())
    conn.start_next_cycle()


SERVER_CORPORA: List[Corpus] = [
    # --- Fixed-length body -------------------------------------------------
    Corpus(
        name="content_length_basic",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 5\r\n\r\n"
            b"hello"
        ),
        expected_events=[
            Request(
                method="POST",
                target="/a",
                headers=[("Host", "example.com"), ("Content-Length", "5")],
            ),
            Data(data=b"hello"),
            EndOfMessage(),
        ],
        on_paused=_respond_simple,
    ),
    Corpus(
        name="content_length_zero",
        role=SERVER,
        wire=(
            b"GET /empty HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 0\r\n\r\n"
        ),
        expected_events=[
            Request(
                method="GET",
                target="/empty",
                headers=[("Host", "example.com"), ("Content-Length", "0")],
            ),
            EndOfMessage(),
        ],
        on_paused=_respond_simple,
    ),
    # --- Chunked body, with and without trailer fields ---------------------
    Corpus(
        name="chunked_two_chunks_with_trailer",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhello\r\n"
            b"6\r\n world\r\n"
            b"0\r\n"
            b"x-checksum: deadbeef\r\n\r\n"
        ),
        expected_events=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
            Data(data=b"hello", chunk_start=True, chunk_end=True),
            Data(data=b" world", chunk_start=True, chunk_end=True),
            EndOfMessage(headers=[("x-checksum", "deadbeef")]),
        ],
        on_paused=_respond_simple,
    ),
    Corpus(
        name="chunked_empty_trailer_section",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"3\r\nabc\r\n"
            b"0\r\n\r\n"
        ),
        expected_events=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
            Data(data=b"abc", chunk_start=True, chunk_end=True),
            EndOfMessage(),
        ],
        on_paused=_respond_simple,
    ),
]
SERVER_CORPORA += [
    # --- Duplicate Content-Length: identical values are accepted, the
    # duplicate field is folded into one normalized header. ----------------
    Corpus(
        name="duplicate_content_length_consistent",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 5\r\n"
            b"Content-Length: 5\r\n\r\n"
            b"hello"
        ),
        expected_events=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Content-Length", "5"),
                ],
            ),
            Data(data=b"hello"),
            EndOfMessage(),
        ],
        on_paused=_respond_simple,
    ),
    # Conflicting duplicates must be rejected at header parse time, before any
    # body byte is delivered.
    Corpus(
        name="duplicate_content_length_conflicting",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 5\r\n"
            b"Content-Length: 6\r\n\r\n"
            b"hello"
        ),
        expected_events=[],
        expect_error=True,
        events_before_error=[],
    ),
    # A single field with comma-separated conflicting values is the same
    # smuggling vector and must be rejected the same way.
    Corpus(
        name="comma_content_length_conflicting",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 5, 6\r\n\r\n"
            b"hello"
        ),
        expected_events=[],
        expect_error=True,
        events_before_error=[],
    ),
    # --- Transfer-Encoding and Content-Length together --------------------
    # RFC 7230 3.3.3: TE wins; the CL field is retained on the parsed request
    # but ignored for framing. The body MUST be delimited by chunk boundaries,
    # not by the (spoofable) Content-Length value.
    Corpus(
        name="transfer_encoding_and_content_length_chunked_wins",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 99\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhello\r\n"
            b"0\r\n\r\n"
        ),
        expected_events=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Content-Length", "99"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
            Data(data=b"hello", chunk_start=True, chunk_end=True),
            EndOfMessage(),
        ],
        on_paused=_respond_simple,
    ),
]
SERVER_CORPORA += [
    # --- Premature EOF while reading a fixed-length body ------------------
    Corpus(
        name="content_length_premature_eof",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 5\r\n\r\n"
            b"hel"
        ),
        eof=True,
        expected_events=[],
        expect_error=True,
        events_before_error=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Content-Length", "5"),
                ],
            ),
            Data(data=b"hel"),
        ],
    ),
    # Premature EOF inside a chunk: the partial chunk is delivered as Data,
    # then EOF has to raise rather than completing the message.
    Corpus(
        name="chunked_premature_eof_inside_chunk",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhel"
        ),
        eof=True,
        expected_events=[],
        expect_error=True,
        events_before_error=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
            Data(data=b"hel", chunk_start=True, chunk_end=False),
        ],
    ),
    # Premature EOF before the header section is complete.
    Corpus(
        name="header_section_premature_eof",
        role=SERVER,
        wire=b"POST /a HTTP/1.1\r\nHost: example.com",
        eof=True,
        expected_events=[],
        expect_error=True,
        events_before_error=[],
    ),
    # --- Corrupt chunk framing ---------------------------------------------
    Corpus(
        name="illegal_chunk_header",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"ZZ\r\n"
        ),
        expected_events=[],
        expect_error=True,
        events_before_error=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
        ],
    ),
    Corpus(
        name="malformed_chunk_footer",
        role=SERVER,
        wire=(
            b"POST /a HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhelloXX"
        ),
        expected_events=[],
        expect_error=True,
        events_before_error=[
            Request(
                method="POST",
                target="/a",
                headers=[
                    ("Host", "example.com"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
            Data(data=b"hello", chunk_start=True, chunk_end=True),
        ],
    ),
]
SERVER_CORPORA += [
    # --- Two complete, pipelined keep-alive requests -----------------------
    # The boundary between the two messages is defined by Content-Length;
    # fragmentation must never move it. The driver responds to the first
    # request and calls start_next_cycle() before the second can be read.
    Corpus(
        name="two_pipelined_requests",
        role=SERVER,
        wire=(
            b"POST /1 HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n"
            b"12345"
            b"POST /2 HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n"
            b"67890"
        ),
        expected_events=[
            Request(
                method="POST",
                target="/1",
                headers=[("Host", "a"), ("Content-Length", "5")],
            ),
            Data(data=b"12345"),
            EndOfMessage(),
            Request(
                method="POST",
                target="/2",
                headers=[("Host", "a"), ("Content-Length", "5")],
            ),
            Data(data=b"67890"),
            EndOfMessage(),
        ],
        on_paused=_respond_simple,
    ),
    # A short first body immediately followed by a second request. With a
    # correct Content-Length boundary the body is exactly "12345" and the
    # next byte must start a brand new request. This is the primary corpus
    # that catches length-boundary weakening and cross-message state bugs.
    Corpus(
        name="pipelined_request_boundary_back_to_back",
        role=SERVER,
        wire=(
            b"POST /1 HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n"
            b"12345"
            b"GET /2 HTTP/1.1\r\nHost: a\r\n\r\n"
        ),
        expected_events=[
            Request(
                method="POST",
                target="/1",
                headers=[("Host", "a"), ("Content-Length", "5")],
            ),
            Data(data=b"12345"),
            EndOfMessage(),
            Request(
                method="GET",
                target="/2",
                headers=[("Host", "a")],
            ),
            EndOfMessage(),
        ],
        on_paused=_respond_simple,
    ),
    # --- Truncated body with a second request smuggled behind it ----------
    # The first request claims 5 bytes but the stream only contains 3 before
    # what the attacker hopes is a new request line. When EOF arrives the
    # parser must surface RemoteProtocolError and enter ERROR; the smuggled
    # bytes must never be delivered as a complete second Request event.
    Corpus(
        name="truncated_body_smuggled_request_eof",
        role=SERVER,
        wire=(
            b"POST /1 HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n"
            b"123"
            b"POST /2 HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\n\r\n"
            b"ab"
        ),
        eof=True,
        expected_events=[],
        expect_error=True,
        events_before_error=[
            Request(
                method="POST",
                target="/1",
                headers=[("Host", "a"), ("Content-Length", "5")],
            ),
            # The declared length over-runs the real body into the smuggled
            # request line; this malformed stream is the test fixture.
            Data(data=b"123PO"),
            EndOfMessage(),
            # The desync surfaces as a second request whose method has been
            # eaten by the first body, followed by the EOF-induced error.
            Request(
                method="ST",
                target="/2",
                headers=[("Host", "a"), ("Content-Length", "3")],
            ),
            Data(data=b"ab"),
        ],
        on_paused=_respond_simple,
    ),
    # Chunked variant: an unfinished chunk followed by a smuggled request.
    # There is no valid end-of-chunk, so EOF must error out; the smuggled
    # request line is only ever body/trailing bytes, never a Request event.
    Corpus(
        name="truncated_chunk_smuggled_request_eof",
        role=SERVER,
        wire=(
            b"POST /1 HTTP/1.1\r\nHost: a\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhel"
            b"POST /2 HTTP/1.1\r\nHost: a\r\n\r\n"
        ),
        eof=True,
        expected_events=[],
        expect_error=True,
        events_before_error=[
            Request(
                method="POST",
                target="/1",
                headers=[
                    ("Host", "a"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
            Data(data=b"helPO", chunk_start=True, chunk_end=True),
        ],
    ),
]
#
# Client-side corpora: the connection first sends a request (initial_send),
# then the wire bytes are responses received by a CLIENT connection.
#


def _assert_switched_and_paused(conn: Connection) -> None:
    # A successful protocol switch freezes the HTTP state machine: further
    # bytes are tunnel data and must keep being reported via PAUSED +
    # trailing_data, never parsed as HTTP events.
    assert conn.their_state is SWITCHED_PROTOCOL
    assert conn.next_event() is PAUSED


CLIENT_CORPORA: List[Corpus] = [
    # 100 Continue is an InformationalResponse that leaves the connection in
    # SEND_RESPONSE; the final Response still owns the body boundary.
    Corpus(
        name="informational_100_then_final_response",
        role=CLIENT,
        initial_send=[
            Request(
                method="POST",
                target="/upload",
                headers=[
                    ("Host", "example.com"),
                    ("Content-Length", "2"),
                    ("Expect", "100-continue"),
                ],
            ),
            Data(data=b"hi"),
            EndOfMessage(),
        ],
        wire=(
            b"HTTP/1.1 100 Continue\r\n\r\n"
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n"
            b"hi"
        ),
        expected_events=[
            InformationalResponse(
                status_code=100,
                headers=[],
                reason="Continue",
            ),
            Response(
                status_code=200,
                headers=[("Content-Length", "2")],
                reason="OK",
            ),
            Data(data=b"hi"),
            EndOfMessage(),
        ],
    ),
    # Multiple 1xx responses are all InformationalResponse events and must
    # not be confused with message boundaries; only the 2xx ends the response
    # headers and starts the body.
    Corpus(
        name="two_informational_responses_then_final",
        role=CLIENT,
        initial_send=[
            Request(
                method="GET",
                target="/job",
                headers=[("Host", "example.com")],
            ),
            EndOfMessage(),
        ],
        wire=(
            b"HTTP/1.1 102 Processing\r\n\r\n"
            b"HTTP/1.1 100 Continue\r\n\r\n"
            b"HTTP/1.1 204 No Content\r\n\r\n"
        ),
        expected_events=[
            InformationalResponse(
                status_code=102,
                headers=[],
                reason="Processing",
            ),
            InformationalResponse(
                status_code=100,
                headers=[],
                reason="Continue",
            ),
            Response(
                status_code=204,
                headers=[],
                reason="No Content",
            ),
            EndOfMessage(),
        ],
    ),
    # 101 Switching Protocols: the InformationalResponse switches the
    # connection; everything after the blank line is opaque tunnel data.
    Corpus(
        name="upgrade_101_switch_with_trailing_bytes",
        role=CLIENT,
        initial_send=[
            Request(
                method="GET",
                target="/chat",
                headers=[
                    ("Host", "example.com"),
                    ("Upgrade", "websocket"),
                ],
            ),
            EndOfMessage(),
        ],
        wire=(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n\r\n"
            b"raw-tunnel-bytes"
        ),
        expected_events=[
            InformationalResponse(
                status_code=101,
                headers=[("Upgrade", "websocket")],
                reason="Switching Protocols",
            ),
        ],
        after=_assert_switched_and_paused,
    ),
    # CONNECT accepted with a 200: a plain Response (no 1xx) performs the
    # switch; trailing tunnel bytes stay in trailing_data.
    Corpus(
        name="connect_200_switch_with_trailing_bytes",
        role=CLIENT,
        initial_send=[
            Request(
                method="CONNECT",
                target="example.com:443",
                headers=[("Host", "example.com:443")],
            ),
            EndOfMessage(),
        ],
        wire=b"HTTP/1.1 200 OK\r\n\r\ntunnel-bytes",
        expected_events=[
            Response(
                status_code=200,
                headers=[],
                reason="OK",
            ),
        ],
        after=_assert_switched_and_paused,
    ),
]


def _assert_reusable_after_denied_connect(conn: Connection) -> None:
    # A denied switch proposal is cleared by the non-switch Response, so once
    # both sides reach DONE the same Connection object must start a fresh,
    # independent request cycle via start_next_cycle().
    assert conn.states == {CLIENT: DONE, SERVER: DONE}
    conn.start_next_cycle()
    conn.send(
        Request(
            method="GET",
            target="/fallback",
            headers=[("Host", "example.com")],
        )
    )
    conn.send(EndOfMessage())


CLIENT_CORPORA += [
    Corpus(
        name="connect_denied_then_cycle_reused",
        role=CLIENT,
        initial_send=[
            Request(
                method="CONNECT",
                target="example.com:443",
                headers=[("Host", "example.com:443")],
            ),
            EndOfMessage(),
        ],
        wire=(
            b"HTTP/1.1 404 Not Found\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"0\r\n\r\n"
        ),
        expected_events=[
            Response(
                status_code=404,
                headers=[("Transfer-Encoding", "chunked")],
                reason="Not Found",
            ),
            EndOfMessage(),
        ],
        after=_assert_reusable_after_denied_connect,
    ),
]

ALL_CORPORA: List[Corpus] = SERVER_CORPORA + CLIENT_CORPORA


# A "cut" is a byte offset where one recv() chunk ends. The same wire bytes
# are always fed in order; the schedule only changes where the boundaries
# between receive_data() calls fall.
def _split_at_cuts(wire: bytes, cuts: Sequence[int]) -> List[bytes]:
    pieces: List[bytes] = []
    start = 0
    for cut in cuts:
        if cut > start:
            pieces.append(wire[start:cut])
            start = cut
    if start < len(wire):
        pieces.append(wire[start:])
    return pieces


def _single_cut_schedules(wire: bytes) -> List[Tuple[int, ...]]:
    # Every boundary position 1..len(wire)-1: each test splits the stream in
    # exactly two non-empty pieces once.
    return [(cut,) for cut in range(1, len(wire))]


def _multi_cut_schedules(wire: bytes, name: str) -> List[Tuple[int, ...]]:
    # Deterministic, seeded schedules. Randomness is only used to pick cut
    # positions; the seed is fixed so runs (and failure replay) are stable.
    rng = random.Random(MULTI_CUT_RNG_SEED + sum(name.encode("ascii")))
    max_cuts = min(len(wire) - 1, 8)
    schedules: List[Tuple[int, ...]] = []
    seen = set()
    attempts = 0
    while len(schedules) < MULTI_CUT_SCHEDULES_PER_CORPUS and attempts < 200:
        attempts += 1
        if max_cuts <= 0:
            break
        cut_count = rng.randint(2, max(2, max_cuts))
        cuts = tuple(sorted(rng.sample(range(1, len(wire)), cut_count)))
        if cuts in seen:
            continue
        seen.add(cuts)
        schedules.append(cuts)
    return schedules


def _schedules_for(wire: bytes, name: str) -> List[Tuple[str, int, Tuple[int, ...]]]:
    schedules: List[Tuple[str, int, Tuple[int, ...]]] = []
    for cuts in _single_cut_schedules(wire):
        schedules.append(("single", 0, cuts))
    for index, cuts in enumerate(_multi_cut_schedules(wire, name)):
        schedules.append(("multi", index, cuts))
    return schedules


def _parametrize_cases() -> List[Any]:
    cases: List[Any] = []
    for corpus in ALL_CORPORA:
        for kind, seed_index, cuts in _schedules_for(corpus.wire, corpus.name):
            seed = (
                MULTI_CUT_RNG_SEED + sum(corpus.name.encode("ascii"))
                if kind == "multi"
                else 0
            )
            case_id = f"{corpus.name}[{kind}-{seed_index}-cuts{list(cuts)}]".replace(
                " ", "_"
            )
            cases.append(pytest.param(corpus, kind, seed, seed_index, cuts, id=case_id))
    return cases


def _drive_corpus(
    corpus: Corpus, cuts: Tuple[int, ...], kind: str, seed: int
) -> Tuple[List[Event], Optional[RemoteProtocolError], Connection]:
    conn = Connection(corpus.role)
    if corpus.initial_send is not None:
        for event in corpus.initial_send:
            conn.send(event)

    delivered: List[Event] = []
    error: Optional[RemoteProtocolError] = None

    def pump() -> None:
        # Drain every event currently available. PAUSED means we cannot make
        # receive progress until the local side advances the cycle; NEED_DATA
        # means more wire bytes are required. We never rebuild conn here.
        nonlocal error
        for _ in range(1000):
            try:
                event = conn.next_event()
            except RemoteProtocolError as exc:
                error = exc
                return
            if event is NEED_DATA:
                return
            if event is PAUSED:
                if corpus.on_paused is None:
                    return
                try:
                    corpus.on_paused(conn)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    raise AssertionError(
                        f"corpus={corpus.name!r} fragmentation={kind} seed={seed} "
                        f"cut_points={list(cuts)}: failure while advancing the "
                        f"conversation from PAUSED: {exc!r}"
                    ) from exc
                continue
            delivered.append(cast(Event, event))
            if type(event) is ConnectionClosed:
                # ConnectionClosed is idempotent; one copy is the terminal
                # receive event, so stop pumping to avoid an infinite loop.
                return
        raise AssertionError("event pump did not converge")

    for piece in _split_at_cuts(corpus.wire, cuts):
        conn.receive_data(piece)
        pump()
        if error is not None:
            break
    if error is None and corpus.eof:
        conn.receive_data(b"")
        pump()

    if corpus.after is not None and error is None:
        try:
            corpus.after(conn)
        except Exception as exc:  # pragma: no cover - diagnostic path
            raise AssertionError(
                f"corpus={corpus.name!r} fragmentation={kind} seed={seed} "
                f"cut_points={list(cuts)}: failure in post-drive check: {exc!r}"
            ) from exc

    return delivered, error, conn


def _assert_events_match(
    corpus_name: str,
    kind: str,
    seed: int,
    schedule_index: int,
    cuts: Tuple[int, ...],
    actual: List[Event],
    expected: List[Event],
) -> None:
    # Adjacent Data events are allowed to differ because their boundaries are
    # an artifact of how the recv() calls happened. Everything else -- header
    # fields, trailer fields, message boundaries, event order -- must match
    # exactly.
    normalized_actual = normalize_data_events(actual)
    normalized_expected = normalize_data_events(expected)
    if normalized_actual == normalized_expected:
        return
    for index, (got, want) in enumerate(zip(normalized_actual, normalized_expected)):
        if got != want:
            first_difference = (
                f"first different event at index {index}:\n"
                f"  got:      {got!r}\n"
                f"  expected: {want!r}"
            )
            break
    else:
        first_difference = (
            f"event count differs: got {len(normalized_actual)} "
            f"events, expected {len(normalized_expected)}"
        )
    # The parameterized node id embeds the exact cut list; quoting it lets a
    # single command replay the exact failing fragmentation schedule.
    cut_signature = "cuts" + str(list(cuts)).replace(" ", "_")
    node_id = (
        "h11/tests/test_message_fragmentation.py::"
        f"test_message_boundaries[{corpus_name}[{kind}-{schedule_index}-"
        f"{cut_signature}]]"
    )
    replay = f"pytest -q '{node_id}'"
    pytest.fail(
        "\n".join(
            [
                f"corpus={corpus_name!r} fragmentation={kind} seed={seed} "
                f"cut_points={list(cuts)}",
                first_difference,
                f"replay: {replay}",
            ]
        )
    )


FRAGMENTATION_CASES = _parametrize_cases()


def test_execution_count_is_bounded() -> None:
    # Keeps the module a fixed-size short-message suite instead of a fuzzer.
    assert len(FRAGMENTATION_CASES) <= MAX_TOTAL_EXECUTIONS


@pytest.mark.parametrize(
    "corpus, kind, seed, schedule_index, cuts", FRAGMENTATION_CASES
)
def test_message_boundaries(
    corpus: Corpus,
    kind: str,
    seed: int,
    schedule_index: int,
    cuts: Tuple[int, ...],
) -> None:
    delivered, error, conn = _drive_corpus(corpus, cuts, kind, seed)

    if corpus.expect_error:
        assert error is not None, (
            f"corpus={corpus.name!r} seed={seed} cut_points={list(cuts)}: "
            f"expected RemoteProtocolError but got events {delivered!r}"
        )
        # The peer side must be in ERROR, and once the error has fired every
        # later receive attempt must keep failing: a corrupted message can
        # never become a second, deliverable request on this connection. Two
        # consecutive calls make the stickiness of the ERROR state explicit.
        assert conn.their_state is ERROR
        with pytest.raises(RemoteProtocolError):
            conn.next_event()
        with pytest.raises(RemoteProtocolError):
            conn.next_event()
        _assert_events_match(
            corpus.name,
            kind,
            seed,
            schedule_index,
            cuts,
            delivered,
            corpus.events_before_error,
        )
    else:
        assert error is None, (
            f"corpus={corpus.name!r} seed={seed} cut_points={list(cuts)}: "
            f"unexpected RemoteProtocolError: {error!r}"
        )
        _assert_events_match(
            corpus.name,
            kind,
            seed,
            schedule_index,
            cuts,
            delivered,
            corpus.expected_events,
        )
