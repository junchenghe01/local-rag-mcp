"""构建 Wheel 包 --- Cython .pyd + .whl。

用法:
    .venv/Scripts/python.exe build/build_release.py                  # 构建
    .venv/Scripts/python.exe build/build_release.py --clean          # 清理后构建
    .venv/Scripts/python.exe build/build_release.py --install        # 构建并安装
    .venv/Scripts/python.exe build/build_release.py --no-cython      # 跳过 Cython
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "rag_mcp"
DIST = ROOT / "dist"

_CYTHON_TARGETS = ["engine.py", "lancedb_store.py", "chunker.py"]


def clean():
    for d in (DIST, ROOT / "build" / "wheel_tmp"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    # 清理 Cython 构建产物
    for build_dir in list(SRC.glob("build")) + list((SRC.parent).glob("build")):
        if build_dir.is_dir():
            shutil.rmtree(build_dir, ignore_errors=True)
    for p in list(SRC.glob("*.pyd")) + list(SRC.glob("*.c")):
        p.unlink()
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
    print("[clean] 完成")


def cython_compile() -> bool:
    """编译 _CYTHON_TARGETS → .pyd。"""
    print(f"\n{'=' * 50}")
    print("  Cython (.py → .pyd)")
    print(f"{'=' * 50}")

    try:
        from Cython.Build import cythonize  # noqa: F401
    except ImportError:
        print("[Cython] 未安装 (pip install cython setuptools)")
        return False

    extensions = []
    for t in _CYTHON_TARGETS:
        src = SRC / t
        if src.exists():
            extensions.append((f"rag_mcp.{t[:-3]}", str(src)))
            print(f"[Cython] {t}")

    if not extensions:
        return False

    # 用临时 .py 脚本执行 Cython 编译 (比 -c 内联更可靠)
    setup_code = (
        "import sys\n"
        f"sys.path.insert(0, r'{SRC.parent}')\n"
        "from setuptools import setup, Extension\n"
        "from Cython.Build import cythonize\n\n"
        f"exts = [Extension(n, [s]) for n, s in {extensions!r}]\n"
        "setup(\n"
        "    name='_',\n"
        "    ext_modules=cythonize(exts, compiler_directives={\n"
        "        'language_level': '3',\n"
        "        'boundscheck': False,\n"
        "        'wraparound': False,\n"
        "        'cdivision': True,\n"
        "    }),\n"
        "    script_args=['build_ext', '--inplace'],\n"
        ")\n"
    )
    tmp_setup = ROOT / "build" / "_cython_setup.py"
    tmp_setup.write_text(setup_code, encoding="utf-8")

    print("[Cython] 编译中...")
    r = subprocess.run(
        [sys.executable, str(tmp_setup)],
        cwd=str(SRC.parent),
        capture_output=True, text=True,
    )
    tmp_setup.unlink()

    if r.returncode != 0:
        print("[Cython] 编译失败 (缺少 MSVC/gcc):")
        for line in r.stderr.strip().split("\n")[-8:]:
            print(f"  {line}")
        return False

    for c_file in SRC.glob("*.c"):
        c_file.unlink()

    ok = 0
    for t in _CYTHON_TARGETS:
        base = t[:-3]  # engine.py → engine
        matches = list(SRC.glob(f"{base}*.pyd"))
        if matches:
            pyd = matches[0]
            # 去掉平台后缀: engine.cp313-win_amd64.pyd → engine.pyd
            clean_name = SRC / f"{base}.pyd"
            if pyd != clean_name:
                shutil.move(str(pyd), str(clean_name))
            pyd = clean_name
            print(f"[Cython] OK {pyd.name} ({pyd.stat().st_size / 1024**2:.1f} MB)")
            ok += 1
        else:
            print(f"[Cython] MISS {t}")
    print(f"[Cython] {ok}/{len(_CYTHON_TARGETS)} 编译成功")
    return ok > 0


def build_wheel(cython_ok: bool) -> Path | None:
    """构建 .whl。"""
    print(f"\n{'=' * 50}")
    print("  Wheel 打包")
    print(f"{'=' * 50}")

    try:
        import build  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "build", "-q"], check=True)

    DIST.mkdir(parents=True, exist_ok=True)
    for whl in DIST.glob("*.whl"):
        try:
            whl.unlink()
        except (PermissionError, OSError):
            print(f"[wheel] 跳过被占用的文件: {whl.name}")
            continue

    tmp = ROOT / "build" / "wheel_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    pkg = tmp / "rag_mcp"
    pkg.mkdir(parents=True)

    for f in SRC.glob("*.py"):
        shutil.copy2(f, pkg / f.name)
    parsers = SRC / "parsers"
    if parsers.is_dir():
        shutil.copytree(parsers, pkg / "parsers")

    if cython_ok:
        print("[wheel] .pyd 替换 .py")
        for t in _CYTHON_TARGETS:
            pyd = SRC / t.replace(".py", ".pyd")
            if pyd.exists():
                (pkg / t).unlink()
                shutil.copy2(pyd, pkg / pyd.name)
                print(f"[wheel]   {t} → {pyd.name}")
    else:
        print("[wheel] 纯 Python")

    (tmp / "setup.py").write_text(
        "from setuptools import setup, find_packages\n"
        "setup(name='rag-mcp', version='2.1.0', packages=find_packages(),\n"
        "      package_data={'rag_mcp': ['*.pyd', 'parsers/*.py']},\n"
        "      python_requires='>=3.10',\n"
        "      entry_points={'console_scripts': ['rag-mcp=rag_mcp.server:main']})\n"
    )

    r = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(DIST)],
        cwd=str(tmp), capture_output=True, text=True,
    )
    shutil.rmtree(tmp, ignore_errors=True)

    if r.returncode != 0:
        print("[FAIL]")
        print(r.stderr.strip()[-800:])
        return None

    whls = list(DIST.glob("*.whl"))
    if not whls:
        print("[FAIL] .whl 未生成")
        return None

    w = whls[0]
    import zipfile
    with zipfile.ZipFile(w, "r") as zf:
        pyd_n = sum(1 for n in zf.namelist() if n.endswith(".pyd"))
        py_n = sum(1 for n in zf.namelist() if n.endswith(".py"))
    print(f"[OK] {w.name} ({w.stat().st_size / 1024:.0f} KB, {pyd_n}.pyd + {py_n}.py)")
    return w


def main():
    import argparse
    p = argparse.ArgumentParser(description="RAG MCP Server Wheel 构建")
    p.add_argument("--clean", action="store_true", help="构建前清理")
    p.add_argument("--install", action="store_true", help="构建后 pip install")
    p.add_argument("--no-cython", action="store_true", help="跳过 Cython")
    args = p.parse_args()

    print("=" * 60)
    print("  RAG MCP Server V2.0 - Wheel 构建")
    print("=" * 60)
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    if args.clean:
        clean()

    cython_ok = False
    if not args.no_cython:
        if any((SRC / t.replace(".py", ".pyd")).exists() for t in _CYTHON_TARGETS):
            print("[SKIP] .pyd 已存在")
            cython_ok = True
        else:
            cython_ok = cython_compile()

    whl = build_wheel(cython_ok)
    if whl is None:
        sys.exit(1)

    if args.install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(whl), "--force-reinstall", "--no-deps"],
        )
        print("[OK] 安装完成")

    prot = "Cython .pyd" if cython_ok else "纯 Python 源码"
    print(f"\n保护: {prot}")
    print(f"安装: pip install {whl}")


if __name__ == "__main__":
    main()
