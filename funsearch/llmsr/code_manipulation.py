import re
import ast


def fix_wrong_escape(code: str) -> str:
    try:
        code = code.replace("â\x80\x98â\x80\x98â\x80\x98", '"""')
        code = re.sub(r'[^\x00-\x7F]+', ' ', code)
        code = code.encode("utf-8").decode("unicode_escape")
        return code
    except Exception as e:
        raise ValueError("Unicode escape decoding failed.\n", code) from e


def extract_fn_header(code: str) -> str:
    # トリプルクォートのdocstring（シングル・ダブル両方）を削除
    match = re.search(r'def\s+\w+\s*\((.*?)\)', code, re.DOTALL)
    if match:
        # Return the captured group with any extra surrounding whitespace removed.
        return match.group(1).strip()
    raise ValueError("No function header found in the provided code.", code)


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


def fix_missing_fn_header(code: str, example: str) -> str:
    """
    Fixes the answer if it is missing a function header by extracting the header from the example.

    If the provided answer starts with a 'return' statement (indicating that only the return
    portion of the function is present), this function extracts the function header from the
    example and prepends it to the answer with proper indentation for the function body.
    Otherwise, it returns the answer unchanged.

    Args:
        code (str): The response string that may be missing the function header.
        example (str): A complete function definition used as a reference to extract the header.

    Returns:
        str: The complete function definition with the header, or the original answer if the header is present.

    Raises:
        ValueError: If the answer is missing the header and no valid function header can be found in the example.
    """
    import re

    # Check if the code starts with a "return" statement after stripping leading whitespace.
    if code.lstrip().startswith("return"):
        # Use regex to capture the function header from the example.
        header_match = re.search(
            r'^(def\s+\w+\(.*\).*:)', example, re.MULTILINE)
        if header_match:
            header = header_match.group(0)
            # Indent each line of the code block by 4 spaces to match standard Python formatting.
            indented_code = "\n".join(
                "    " + line if line.strip() != "" else "" for line in code.splitlines())
            # Prepend the header to the indented code block.
            return header + "\n" + indented_code
        else:
            raise ValueError(
                "No function header found in the provided example")
    # If code already includes a function header, return it unchanged.
    return code


def extract_last_function(code: str) -> str:
    """
    Extracts the portion of the code starting from the last function definition header.

    This function searches for the last occurrence of a function definition (a line that starts with "def"
    and ends with a colon). If found, it returns the code starting from that header (thus cutting off anything
    above it). If no function header is found, it returns the original code unchanged so that subsequent
    fixers can handle it.

    Args:
        code (str): The code string to process.

    Returns:
        str: The processed code starting from the last function definition, or the original code if none is found.
    """
    # Regular expression pattern to match a function definition header.
    # It matches lines beginning with 'def ', followed by a function name, parameters in parentheses, and ending with a colon.
    pattern = r'^(def\s+\w+\(.*\).*:)'

    # Find all matches of the function header in the code.
    matches = list(re.finditer(pattern, code, re.MULTILINE))

    # If at least one function header is found, return the substring starting from the last one.
    if matches:
        last_match = matches[-1]
        start_pos = last_match.start()
        return code[start_pos:]
    else:
        # If no function header is found, return the original code to let subsequent fixers process it.
        return code


def fix_missing_header_and_ret(code: str, example: str) -> str:
    """
    Wraps code in a function header extracted from 'example' only if the code is a single assignment
    or a single return statement. If the code already contains a valid function definition, it is
    returned unchanged. This ensures that only cases like:

        'return params[0] * width * wavelength * np.cos(width * np.pi / wavelength) + params[1] * np.sin(width * np.pi / wavelength)'
        'shg_efficieny = params[0] * (params[1] * wavelength - params[2])**2 * width * np.sin(params[3] * width) * np.exp(-params[4] * width) * np.cos(params[5] * wavelength) * np.sin(params[6] * wavelength)'

    are fixed, while complete function definitions remain intact.

    Args:
        code (str): Code that may be missing a function header.
        example (str): A complete function definition used to extract a valid function header.

    Returns:
        str: A complete function definition if applicable, or the original code unchanged.
    """
    # First, try to parse the code to see if it already defines a function.
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            # Valid function definition found; do not modify.
            return code
    except Exception:
        # If parsing fails here, we assume the code isn't a full function definition.
        pass

    # Attempt to parse the code to check its structure.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If parsing fails, return the code unchanged.
        return code

    # Only apply the fix if the code consists of exactly one statement which is either:
    #   - a single assignment statement, or
    #   - a single return statement.
    if (isinstance(tree.body[0], ast.Assign) or isinstance(tree.body[0], ast.Return)):
        # Extract the function header from the example.
        header_match = re.search(
            r'^(def\s+\w+\(.*\).*:)', example, re.MULTILINE)
        if not header_match:
            raise ValueError("提供されたexampleから関数ヘッダーを抽出できませんでした。")
        header = header_match.group(0)

        # Indent the original code by 4 spaces.
        indented_code = "\n".join(
            "    " + line if line.strip() else "" for line in code.splitlines())

        # If it's an assignment statement, add a return statement automatically.
        if isinstance(tree.body[0], ast.Assign):
            assign_node = tree.body[0]
            # Only add return if there's a single target variable.
            if len(assign_node.targets) == 1 and isinstance(assign_node.targets[0], ast.Name):
                var_name = assign_node.targets[0].id
                indented_code += "\n    return " + var_name

        # For a return statement, no additional return is needed.
        return header + "\n" + indented_code

    # If the code doesn't match the two specific cases, return it unchanged.
    return code


def fix_indentation(code: str) -> str:
    """
    与えられたコード内の各行について、先頭の空白数をタブ→スペース変換後に計算し、
    4の倍数になるように四捨五入で調整します。

    例:
      - 5スペース -> 4スペース
      - 3スペース -> 4スペース
      - 7スペース -> 8スペース

    タブは expandtabs(4) を使って4スペース相当に変換するので、大丈夫です。

    Args:
        code (str): 修正前のコード文字列

    Returns:
        str: インデントが調整されたコード文字列
    """
    new_lines = []
    for line in code.splitlines():
        # タブを4スペースに変換
        expanded_line = line.expandtabs(4)
        # 先頭の空白部分を取得
        match = re.match(r'^(\s*)', expanded_line)
        current_indent = match.group(1)  # type: ignore
        num_spaces = len(current_indent)

        # 4の倍数でない場合、四捨五入で最も近い倍数に調整
        if num_spaces % 4 != 0:
            # 四捨五入（半分以上は切り上げ）するために、0.5を足してから整数に変換
            adjusted_spaces = int(num_spaces / 4 + 0.5) * 4
            new_indent = " " * adjusted_spaces
            # 元のインデント部分を新しいインデントに置換
            fixed_line = new_indent + expanded_line[len(current_indent):]
        else:
            fixed_line = expanded_line
        new_lines.append(fixed_line)
    return "\n".join(new_lines)
