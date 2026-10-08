"""Create deterministic multilingual retrieval documents."""

from __future__ import annotations

import argparse
from pathlib import Path

DOCUMENTS = {
    "operations.md": """# Cooling controls

The emergency cooling valve is painted cobalt blue. Pulling it isolates the
primary loop and starts the reserve pump.

# Secure service

Encrypted client traffic is accepted on port 8443. The listener requires TLS
and rejects clear-text connections.
""",
    "spanish.md": """# Política de respaldo

Las copias de seguridad se ejecutan cada noche a las 02:30. Los archivos se
conservan durante catorce días antes de su eliminación programada.
""",
    "german.md": """# Alarmsteuerung

Der Drucksensor P-17 löst den Alarm aus, sobald der Leitungsdruck unter
vier bar fällt. Der Temperatursensor ist nur für die Anzeige zuständig.
""",
    "research.md": """# Catalyst trial C-9

The catalyst trial ran at 68 degrees Celsius for twelve minutes. The control
sample used the same duration without the nickel additive.
""",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, content in DOCUMENTS.items():
        (root / name).write_text(content, encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
