"""Resolve EPTS channel declarations without assuming XML roster order.

This supports the unnamespaced Metrica EPTS export. It reads format metadata
only, never event, score-value, possession or live/dead fields. Errors omit
source values.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Layout:
    first: int
    last: int
    # For each raw player group: output slot, tuple width, x index, y index.
    players: tuple
    # Tuple width and x/y indices in the ball group.
    ball: tuple


def _integer(value):
    try:
        number = int(float(value))      # "25.0" is a valid whole number; "25.5" is not
        if float(value) != number:
            raise ValueError
        return number
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Invalid EPTS frame bound") from None


def resolve(root, n_frames):
    """Return team membership and declared coordinate layouts for recorded frames.

    The minimal no-format case supports the older static x,y,s convention.
    If declarations exist, they are authoritative and must be consistent.
    """
    metadata = root.find("Metadata")
    if metadata is None:
        metadata = root
    teams = metadata.findall("Teams/Team") or metadata.findall("Team")
    ids = [node.get("id") for node in teams]
    if len(ids) != 2 or any(not x for x in ids) or len(set(ids)) != 2:
        raise ValueError("Expected two distinct EPTS teams")
    # The public EPTS loader uses these role attributes when present; score
    # numbers are neither read nor used. Otherwise XML team order is retained.
    session = metadata.find("Sessions/Session")
    score = session.find("MatchParameters/Score") if session is not None else metadata.find("Score")
    if score is not None:
        roles = [score.get("idLocalTeam"), score.get("idVisitingTeam")]
        if set(roles) != set(ids) or len(set(roles)) != 2:
            raise ValueError("Invalid EPTS team roles")
        ids = roles
    players = metadata.findall("Players/Player") or metadata.findall("Player")
    if not players or any(node.get("teamId") not in ids for node in players):
        raise ValueError("Invalid EPTS team membership")
    membership = np.asarray([ids.index(node.get("teamId")) for node in players], np.int8)
    rates = metadata.findall("GlobalConfig/FrameRate")
    if len(rates) > 1 or rates and _integer(rates[0].text) != 25:
        raise ValueError("Expected 25 Hz tactical tracking")
    specifications = root.findall("DataFormatSpecifications/DataFormatSpecification")
    if not specifications:
        if (root.find("DataFormatSpecifications") is not None
                or metadata.find("PlayerChannels") is not None
                or metadata.find("Devices") is not None or metadata.find("Sensors") is not None):
            raise ValueError("Incomplete modern EPTS format declarations")
        return membership, (Layout(1, n_frames, tuple((i, 3, 0, 1) for i in range(len(players))), (3, 0, 1)),)
    player_ids = [node.get("id") for node in players]
    if any(not x for x in player_ids) or len(set(player_ids)) != len(player_ids):
        raise ValueError("Invalid EPTS track-slot identifiers")
    slots = {identity: index for index, identity in enumerate(player_ids)}
    units = {}
    device = metadata.find("Devices/Device")
    sensors = device.find("Sensors") if device is not None else metadata.find("Sensors")
    channel_nodes = sensors.findall("Sensor/Channels/Channel") if sensors is not None else []
    for node in channel_nodes:
        identity = node.get("id")
        if not identity or identity in units:
            raise ValueError("Invalid EPTS channel identifiers")
        units[identity] = (node.findtext("Unit") or "").strip()
    if any(units.get(axis) != "normalized" for axis in ("x", "y")):
        raise ValueError("Expected normalized EPTS x/y coordinates")
    channels = {}
    assignments = set()
    for node in metadata.findall("PlayerChannels/PlayerChannel"):
        identity, player, axis = node.get("id"), node.get("playerId"), node.get("channelId")
        if not identity or identity in channels or player not in slots or axis not in units:
            raise ValueError("Unresolved EPTS player channel")
        if (player, axis) in assignments:
            raise ValueError("Duplicate EPTS player channel")
        channels[identity] = (slots[player], axis)
        assignments.add((player, axis))
    layouts = []
    for spec in specifications:
        first, last = _integer(spec.get("startFrame")), _integer(spec.get("endFrame"))
        if first < 0 or last < first:
            raise ValueError("Invalid EPTS format interval")
        children = list(spec)
        if (spec.get("separator") != ":" or [node.tag for node in children] !=
                ["StringRegister", "SplitRegister", "SplitRegister"]):
            raise ValueError("Unsupported EPTS record layout")
        frame, groups, ball = children
        if not frame.get("name") or groups.get("separator") != ";":
            raise ValueError("Unsupported EPTS frame/player layout")
        resolved = []
        for group in groups:
            if group.tag != "SplitRegister" or group.get("separator") != ",":
                raise ValueError("Unsupported EPTS coordinate tuple")
            refs = []
            for ref in group:
                if ref.tag != "PlayerChannelRef" or ref.get("playerChannelId") not in channels:
                    raise ValueError("Unresolved EPTS coordinate reference")
                refs.append(channels[ref.get("playerChannelId")])
            axes = [axis for _, axis in refs]
            if (not refs or len({slot for slot, _ in refs}) != 1
                    or axes.count("x") != 1 or axes.count("y") != 1 or len(set(axes)) != len(axes)):
                raise ValueError("Mixed or duplicate EPTS coordinate reference")
            resolved.append((refs[0][0], len(refs), axes.index("x"), axes.index("y")))
        if not resolved or len({entry[0] for entry in resolved}) != len(resolved):
            raise ValueError("Duplicate or empty active EPTS slot layout")
        axes = [ref.get("channelId") for ref in ball]
        if (ball.get("separator") != "," or any(ref.tag != "BallChannelRef" for ref in ball)
                or any(axis not in units for axis in axes) or axes.count("x") != 1 or axes.count("y") != 1
                or len(set(axes)) != len(axes)):
            raise ValueError("Invalid EPTS ball-channel layout")
        layouts.append(Layout(first, last, tuple(resolved), (len(axes), axes.index("x"), axes.index("y"))))
    if any(a.last >= b.first for a, b in zip(layouts[:-1], layouts[1:])):
        raise ValueError("Overlapping or unordered EPTS format intervals")
    active = [layout for layout in layouts if layout.last >= 1 and layout.first <= n_frames]
    if (not active or active[0].first > 1 or active[-1].last < n_frames
            or any(a.last+1 != b.first for a, b in zip(active[:-1], active[1:]))):
        raise ValueError("EPTS formats do not cover every recorded frame")
    return membership, tuple(active)


def split_fields(value, separator, expected):
    """Accept one optional trailing declared separator, retaining empty fields."""
    fields = value.split(separator)
    if len(fields) == expected+1 and not fields[-1].strip():
        fields.pop()
    if len(fields) != expected:
        raise ValueError("EPTS field count differs from metadata")
    return fields


def frame_number(value):
    """Parse without allowing a conversion exception to quote a restricted row."""
    return _integer(value)


def coordinates(group, layout):
    """Read declared x/y fields; missing empty fields and NaN remain missing."""
    width, x, y = layout
    fields = split_fields(group, ",", width)
    try:
        values = tuple(float(fields[i]) if fields[i].strip() else np.nan for i in (x, y))
    except ValueError:
        raise ValueError("Nonnumeric EPTS coordinate") from None
    if np.isinf(values).any():
        raise ValueError("Infinite EPTS coordinate")
    return values
