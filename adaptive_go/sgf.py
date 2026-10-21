"""SGF(바둑 기보 표준 형식) 읽기·쓰기. 본 수순(main line)만 다룬다.

SGF 좌표는 왼쪽 위가 "aa"이고 I를 건너뛰지 않는다: "pd" = 16열, 위에서 4번째 줄.
GTP로는 열 COLUMNS[15] = "Q", 행 19 - 3 = 16 → "Q16".
패스는 빈 값 B[] 또는 (19줄 이하에서) B[tt].
"""

from .board import COLUMNS

META_KEYS = ("GN", "EV", "RO", "DT", "PC", "PB", "PW", "BR", "WR", "RE", "RU")


class SgfError(ValueError):
    pass


def _parse_tree(text, i):
    """'(' 위치부터 게임 트리 하나를 읽는다. (노드 목록, 자식 트리 목록, 다음 위치)"""
    assert text[i] == "("
    i += 1
    nodes, children = [], []
    while i < len(text):
        ch = text[i]
        if ch == ";":
            node, i = _parse_node(text, i + 1)
            nodes.append(node)
        elif ch == "(":
            child, i = _parse_tree(text, i)
            children.append(child)
        elif ch == ")":
            return {"nodes": nodes, "children": children}, i + 1
        else:
            i += 1
    raise SgfError("괄호가 닫히지 않음")


def _parse_node(text, i):
    node = {}
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] in ";()":
            return node, i
        start = i
        while i < len(text) and text[i].isalpha():
            i += 1
        ident = "".join(c for c in text[start:i] if c.isupper())  # 옛 형식의 소문자 섞인 속성명 대비
        if not ident:
            raise SgfError(f"속성 이름이 없음: 위치 {i}")
        values = []
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text) or text[i] != "[":
                break
            i += 1
            buf = []
            while i < len(text) and text[i] != "]":
                if text[i] == "\\" and i + 1 < len(text):
                    i += 1
                buf.append(text[i])
                i += 1
            values.append("".join(buf))
            i += 1
        node.setdefault(ident, []).extend(values)
    return node, i


def sgf_to_gtp(value, size):
    if value == "" or (value == "tt" and size <= 19):
        return "pass"
    col, row = ord(value[0]) - ord("a"), ord(value[1]) - ord("a")
    if not (0 <= col < size and 0 <= row < size):
        raise SgfError(f"판 밖 좌표: {value}")
    return f"{COLUMNS[col]}{size - row}"


def gtp_to_sgf(vertex, size):
    if vertex.lower() == "pass":
        return ""
    col = COLUMNS.index(vertex[0].upper())
    row = size - int(vertex[1:])
    return chr(ord("a") + col) + chr(ord("a") + row)


def parse(text):
    """SGF 텍스트 → {size, komi, meta, moves: [(color, gtp)], comments: {수 번호: 글}, setup}"""
    start = text.find("(")
    if start < 0:
        raise SgfError("SGF가 아님")
    tree, _ = _parse_tree(text, start)
    # 본 수순: 각 단계에서 첫 번째 자식만 따라간다
    nodes = []
    while True:
        nodes += tree["nodes"]
        if not tree["children"]:
            break
        tree = tree["children"][0]
    if not nodes:
        raise SgfError("노드가 없음")
    root = nodes[0]
    size = int(root.get("SZ", ["19"])[0].split(":")[0])
    komi_raw = root.get("KM", ["0"])[0]
    try:
        komi = float(komi_raw)
    except ValueError:
        komi = 0.0
    meta = {k: root[k][0] for k in META_KEYS if k in root}
    moves, comments = [], {}
    setup = any(k in n for n in nodes for k in ("AB", "AW", "AE"))
    if "C" in root:
        comments[0] = root["C"][0]
    for node in nodes:
        for color in ("B", "W"):
            if color in node:
                moves.append((color, sgf_to_gtp(node[color][0], size)))
                if "C" in node:
                    comments[len(moves)] = node["C"][0]
    return {"size": size, "komi": komi, "meta": meta, "moves": moves, "comments": comments, "setup": setup}


def dump(size, komi, moves, meta=None):
    """수순 → SGF 텍스트 (내 대국 내보내기용)."""
    esc = lambda v: str(v).replace("\\", "\\\\").replace("]", "\\]")
    head = f"(;GM[1]FF[4]CA[UTF-8]AP[adaptive-go]SZ[{size}]KM[{komi:g}]RU[Chinese]"
    head += "".join(f"{k}[{esc(v)}]" for k, v in (meta or {}).items())
    body = "".join(f";{c}[{gtp_to_sgf(v, size)}]" for c, v in moves)
    return head + body + ")\n"
