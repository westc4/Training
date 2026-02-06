"""
Reporting Module for Preflight Checks

Generates JSON and pretty text reports for preflight results.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any
import json
import datetime


@dataclass
class PreflightReport:
    """Container for preflight report data."""
    
    passed: bool
    timestamp: str
    fatal_errors: list[str]
    warnings: list[str]
    info: list[str]
    checks: dict[str, dict]
    resolved_config: dict[str, Any]
    epoch_meaning: str = ""
    
    @classmethod
    def from_results(cls, results: "PreflightResults") -> "PreflightReport":
        """Create report from PreflightResults."""
        from .preflight import get_epoch_meaning_banner
        
        return cls(
            passed=results.passed,
            timestamp=datetime.datetime.now().isoformat(),
            fatal_errors=results.fatal_errors.copy(),
            warnings=results.warnings.copy(),
            info=results.info.copy(),
            checks=dict(results.checks),
            resolved_config=dict(results.resolved_config),
            epoch_meaning=get_epoch_meaning_banner(results),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "timestamp": self.timestamp,
            "summary": {
                "fatal_errors": len(self.fatal_errors),
                "warnings": len(self.warnings),
                "passed_checks": len([c for c in self.checks.values() if c.get("status") == "OK"]),
            },
            "fatal_errors": self.fatal_errors,
            "warnings": self.warnings,
            "info": self.info,
            "checks": self.checks,
            "resolved_config": self.resolved_config,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_text(self) -> str:
        """Convert to pretty text format."""
        lines = []
        
        lines.append("=" * 70)
        lines.append("PREFLIGHT CHECK REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"Status: {'✓ PASSED' if self.passed else '❌ FAILED'}")
        lines.append("")
        
        # Summary
        lines.append("-" * 70)
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"  Fatal errors: {len(self.fatal_errors)}")
        lines.append(f"  Warnings: {len(self.warnings)}")
        lines.append(f"  Passed checks: {len([c for c in self.checks.values() if c.get('status') == 'OK'])}")
        lines.append("")
        
        # Fatal errors
        if self.fatal_errors:
            lines.append("-" * 70)
            lines.append("❌ FATAL ERRORS")
            lines.append("-" * 70)
            for err in self.fatal_errors:
                lines.append(f"  • {err}")
            lines.append("")
        
        # Warnings
        if self.warnings:
            lines.append("-" * 70)
            lines.append("⚠️  WARNINGS")
            lines.append("-" * 70)
            for warn in self.warnings:
                lines.append(f"  • {warn}")
            lines.append("")
        
        # Passed checks
        if self.info:
            lines.append("-" * 70)
            lines.append("✓ PASSED CHECKS")
            lines.append("-" * 70)
            for info in self.info:
                lines.append(f"  • {info}")
            lines.append("")
        
        # Resolved configuration
        if self.resolved_config:
            lines.append("-" * 70)
            lines.append("RESOLVED CONFIGURATION")
            lines.append("-" * 70)
            
            for section, values in self.resolved_config.items():
                lines.append(f"\n  [{section.upper()}]")
                if isinstance(values, dict):
                    for key, value in values.items():
                        lines.append(f"    {key}: {value}")
                else:
                    lines.append(f"    {values}")
            lines.append("")
        
        # Epoch meaning
        if self.epoch_meaning:
            lines.append("")
            lines.append(self.epoch_meaning)
        
        # Check details
        lines.append("")
        lines.append("-" * 70)
        lines.append("CHECK DETAILS")
        lines.append("-" * 70)
        
        for check_name, check_data in self.checks.items():
            status = check_data.get("status", "UNKNOWN")
            message = check_data.get("message", "")
            
            status_icon = {
                "OK": "✓",
                "WARNING": "⚠️",
                "FATAL": "❌",
            }.get(status, "?")
            
            lines.append(f"\n  {status_icon} {check_name}: {message}")
            
            details = check_data.get("details", {})
            if details:
                for key, value in details.items():
                    if isinstance(value, dict):
                        lines.append(f"      {key}:")
                        for k2, v2 in value.items():
                            lines.append(f"        {k2}: {v2}")
                    elif isinstance(value, list) and len(value) > 5:
                        lines.append(f"      {key}: [{len(value)} items]")
                    else:
                        lines.append(f"      {key}: {value}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def save_report(
    report: PreflightReport,
    xp_dir: Path,
    json_filename: str = "preflight.json",
    text_filename: str = "preflight.txt",
) -> tuple[Path, Path]:
    """
    Save preflight report to XP directory.
    
    Args:
        report: The preflight report
        xp_dir: XP directory to save to
        json_filename: Name for JSON file
        text_filename: Name for text file
        
    Returns:
        (json_path, text_path): Paths to saved files
    """
    xp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = xp_dir / json_filename
    with open(json_path, 'w') as f:
        f.write(report.to_json())
    
    # Save text
    text_path = xp_dir / text_filename
    with open(text_path, 'w') as f:
        f.write(report.to_text())
    
    return json_path, text_path


def load_report(json_path: Path) -> PreflightReport:
    """Load preflight report from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return PreflightReport(
        passed=data.get("passed", False),
        timestamp=data.get("timestamp", ""),
        fatal_errors=data.get("fatal_errors", []),
        warnings=data.get("warnings", []),
        info=data.get("info", []),
        checks=data.get("checks", {}),
        resolved_config=data.get("resolved_config", {}),
    )


def print_report_summary(report: PreflightReport):
    """Print a concise summary of the report."""
    status = "✓ PASSED" if report.passed else "❌ FAILED"
    print(f"\nPreflight: {status}")
    print(f"  Fatal: {len(report.fatal_errors)}, Warnings: {len(report.warnings)}")
    
    if report.fatal_errors:
        print("\n  Fatal errors:")
        for err in report.fatal_errors[:3]:
            print(f"    • {err}")
        if len(report.fatal_errors) > 3:
            print(f"    ... and {len(report.fatal_errors) - 3} more")


def compare_reports(report1: PreflightReport, report2: PreflightReport) -> dict:
    """Compare two preflight reports."""
    changes = {
        "status_changed": report1.passed != report2.passed,
        "new_errors": [],
        "resolved_errors": [],
        "new_warnings": [],
        "resolved_warnings": [],
    }
    
    # Compare errors
    errors1 = set(report1.fatal_errors)
    errors2 = set(report2.fatal_errors)
    changes["new_errors"] = list(errors2 - errors1)
    changes["resolved_errors"] = list(errors1 - errors2)
    
    # Compare warnings
    warnings1 = set(report1.warnings)
    warnings2 = set(report2.warnings)
    changes["new_warnings"] = list(warnings2 - warnings1)
    changes["resolved_warnings"] = list(warnings1 - warnings2)
    
    return changes
