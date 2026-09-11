"""
FiniRAG Validation Schemas.

Defines the structure of validation checks produced by the
financial validation service. Each check compares a calculated
value against a reported value with tolerance.

Statuses: PASS | FAIL | NOT_APPLICABLE (Trap P3, P4)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_serializer


class ValidationCheck(BaseModel):
    """A single validation check result."""
    name: str                                    # "line_total_check_1", "assets_vs_liabilities"
    formula: str = ""                            # "quantity × unit_price", "sum(assets)"
    operands: dict[str, Any] = {}                # {"quantity": "24", "unit_price": "23.50"}
    calculated_value: Decimal | None = None
    reported_value: Decimal | None = None
    variance: Decimal | None = None
    tolerance: Decimal | None = None
    status: str = "NOT_APPLICABLE"               # "PASS" | "FAIL" | "NOT_APPLICABLE"
    reason: str = ""                             # Explanation when NOT_APPLICABLE

    @field_serializer(
        "calculated_value", "reported_value", "variance", "tolerance",
        when_used="json",
    )
    def _serialize_decimal(self, v: Decimal | None) -> str | None:
        """
        Emit exact decimal strings. Serializing money through float would
        reintroduce the binary rounding the Decimal pipeline exists to avoid.
        """
        return None if v is None else str(v)


class ValidationResult(BaseModel):
    """Aggregated validation result for a document."""
    checks: list[ValidationCheck] = []
    overall_status: str = "NOT_APPLICABLE"       # "PASS" | "FAIL" | "NOT_APPLICABLE"
    summary: dict[str, int] = Field(
        default_factory=lambda: {"pass": 0, "fail": 0, "not_applicable": 0}
    )

    def compute_summary(self) -> None:
        """Compute summary counts and overall status from individual checks."""
        self.summary = {"pass": 0, "fail": 0, "not_applicable": 0}
        for check in self.checks:
            if check.status == "PASS":
                self.summary["pass"] += 1
            elif check.status == "FAIL":
                self.summary["fail"] += 1
            else:
                self.summary["not_applicable"] += 1

        # Overall: FAIL if any check fails, PASS if at least one passes and none fail
        if self.summary["fail"] > 0:
            self.overall_status = "FAIL"
        elif self.summary["pass"] > 0:
            self.overall_status = "PASS"
        else:
            self.overall_status = "NOT_APPLICABLE"
