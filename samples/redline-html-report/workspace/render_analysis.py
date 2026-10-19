"""Public REDLINE HTML-renderer task stub. No renderer is implemented here."""


def render_report(analysis: dict) -> str:
    """Return a validated, standalone, escaped Japanese HTML report."""
    raise NotImplementedError('RENDERER_MUST_BE_GENERATED_BY_THE_MODEL')


def main(argv=None) -> int:
    """CLI contract: --input ANALYSIS_JSON --out FRESH_HTML."""
    raise NotImplementedError('RENDERER_CLI_MUST_BE_GENERATED_BY_THE_MODEL')


if __name__ == '__main__':
    raise SystemExit(main())
