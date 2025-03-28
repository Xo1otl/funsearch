import re


def fix_single_quote_line(code: str) -> str:
    """
    単体の '"' が存在する行を検出し、その行の '"' を '\"""' に置換します。

    この関数は、特にdocstringの終了記号として誤って使われた単一の '"' を正しく修正するためのものです。

    Args:
        code (str): 元のコード文字列

    Returns:
        str: 修正後のコード文字列
    """
    # パターンの説明:
    # ^              : 行頭
    # (?P<indent>\s*): インデント（空白文字）をキャプチャ
    # "              : 単体のダブルクオート
    # (?P<tail>\s*)  : 行末までの空白文字をキャプチャ
    # $              : 行末
    pattern = r'^(?P<indent>\s*)"(?P<tail>\s*)$'
    fixed_code = re.sub(pattern, lambda m: m.group(
        'indent') + '"""' + m.group('tail'), code, flags=re.MULTILINE)
    return fixed_code


def remove_docstring(code: str) -> str:
    # トリプルクォートのdocstring（シングル・ダブル両方）を削除
    pattern = r'("""|\'\'\')(.*?)(\1)'
    new_fn_code = re.sub(pattern, '', code, flags=re.DOTALL)
    return new_fn_code


def remove_empty_lines(code: str) -> str:
    # 空行を削除する正規表現
    pattern = r'\n\s*\n'
    new_fn_code = re.sub(pattern, '\n', code)
    return new_fn_code


def set_fn_name(fn_code: str, version: int) -> str:
    pattern = r"^(def\s+)\w+(\s*\(.*?\).*:)"
    new_name = f"equation_v{version}"
    new_fn_code = re.sub(pattern, rf"\1{new_name}\2", fn_code)
    return new_fn_code


def fix_missing_def(code: str) -> str:
    """
    関数定義のヘッダーに 'def' キーワードや末尾のコロンが欠落している場合に補完する。
    - 行頭のインデントを保持しつつ、関数名と引数リスト（さらにオプションで戻り値の型注釈）がある場合、
      先頭に 'def ' を追加し、末尾に ':' がなければ追加する。
    - 既に 'def' が存在する行は変更しない。
    """
    # パターンの説明:
    # ^(?P<indent>\s*)           : 行頭のインデントをキャプチャ
    # (?P<header>(?!def\s)\w+\(.*\)) : "def " で始まらない関数っぽいヘッダー（関数名と引数リスト）
    # (?P<rest>.*)               : ヘッダーの残り（例：戻り値注釈など）
    pattern = r"^(?P<indent>\s*)(?P<header>(?!def\s)\w+\(.*\))(?P<rest>.*)$"

    def replacement(match: re.Match) -> str:
        indent = match.group("indent")
        header = match.group("header")
        rest = match.group("rest")
        fixed_line = f"{indent}def {header}{rest}"
        # 末尾にコロンがなければ追加
        if not fixed_line.rstrip().endswith(':'):
            fixed_line += ':'
        return fixed_line

    # 1行目のみ補正対象
    fixed_code = re.sub(pattern, replacement, code,
                        count=1, flags=re.MULTILINE)
    return fixed_code
