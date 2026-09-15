"""Display a transparent phishing-risk screen from a raw URL."""

import argparse
import html
import ipaddress
import re
import string
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "url_prediction.html"

FEATURE_NAMES = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
    "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation",
    "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL",
    "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS"
]


def normalized_url(raw_url):
    value = raw_url.strip()
    if not value:
        raise ValueError("URL cannot be empty.")
    if "://" not in value:
        value = "https://" + value
    parsed = urlsplit(value)
    if not parsed.hostname:
        raise ValueError("Please enter a valid URL with a domain name.")
    return value, parsed


def is_ip_address(hostname):
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def longest_repeated_run(value):
    if not value:
        return 0
    longest = current = 1
    for previous, current_char in zip(value, value[1:]):
        current = current + 1 if current_char == previous else 1
        longest = max(longest, current)
    return longest


def extract_features(raw_url):
    value, parsed = normalized_url(raw_url)
    hostname = parsed.hostname or ""
    path_and_query = parsed.path + ("?" + parsed.query if parsed.query else "")
    length = max(len(value), 1)
    letters = sum(character.isalpha() for character in value)
    digits = sum(character.isdigit() for character in value)
    punctuation = set(string.punctuation)
    allowed_specials = set("/:.?&=#-_~%")
    other_specials = sum(
        character in punctuation and character not in allowed_specials
        for character in value
    )
    encoded_matches = re.findall(r"%[0-9a-fA-F]{2}", value)
    suspicious_tokens = ("@", "//", "login", "verify", "secure", "update", "account")
    has_obfuscation = bool(encoded_matches or "@" in value or "\\x" in value)
    tld = "" if is_ip_address(hostname) else hostname.rsplit(".", 1)[-1]
    domain_parts = [part for part in hostname.split(".") if part]
    subdomains = max(len(domain_parts) - 2, 0) if not is_ip_address(hostname) else 0
    allowed_character_probability = sum(
        character.isalnum() or character in ".-_/:?=&%" for character in value
    ) / length
    feature_values = {
        "URLLength": len(value),
        "DomainLength": len(hostname),
        "IsDomainIP": int(is_ip_address(hostname)),
        "CharContinuationRate": longest_repeated_run(value) / length,
        "URLCharProb": allowed_character_probability,
        "TLDLength": len(tld),
        "NoOfSubDomain": subdomains,
        "HasObfuscation": int(has_obfuscation),
        "NoOfObfuscatedChar": sum(len(match) for match in encoded_matches),
        "ObfuscationRatio": sum(len(match) for match in encoded_matches) / length,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / length,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / length,
        "NoOfEqualsInURL": value.count("="),
        "NoOfQMarkInURL": value.count("?"),
        "NoOfAmpersandInURL": value.count("&"),
        "NoOfOtherSpecialCharsInURL": other_specials,
        "SpacialCharRatioInURL": other_specials / length,
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
    }
    return value, parsed, feature_values, suspicious_tokens


def heuristic_reasons(parsed, feature_values):
    phishing_reasons = []
    legitimate_signals = []
    if feature_values["IsDomainIP"]:
        phishing_reasons.append("The hostname is an IP address rather than a normal domain.")
    if feature_values["IsHTTPS"] == 0:
        phishing_reasons.append("The URL does not use HTTPS.")
    else:
        legitimate_signals.append("The URL uses HTTPS.")
    if feature_values["URLLength"] > 75:
        phishing_reasons.append("The URL is unusually long.")
    if feature_values["NoOfSubDomain"] >= 2:
        phishing_reasons.append("The domain contains multiple subdomain levels.")
    if feature_values["HasObfuscation"]:
        phishing_reasons.append("The URL contains encoded or obfuscation-like characters.")
    if feature_values["DegitRatioInURL"] > 0.20:
        phishing_reasons.append("The URL contains an unusually high proportion of digits.")
    if feature_values["SpacialCharRatioInURL"] > 0.08:
        phishing_reasons.append("The URL contains many uncommon special characters.")
    if feature_values["NoOfQMarkInURL"] > 0 or feature_values["NoOfEqualsInURL"] >= 2:
        phishing_reasons.append("The URL contains query parameters often used in redirect or tracking links.")
    if not phishing_reasons:
        legitimate_signals.append("No strong URL-level warning signal was detected.")
    return phishing_reasons, legitimate_signals


def screening_probability(feature_values):
    risk_points = 4
    risk_points += 28 if feature_values["IsDomainIP"] else 0
    risk_points += 12 if feature_values["IsHTTPS"] == 0 else 0
    risk_points += 15 if feature_values["URLLength"] > 75 else 0
    risk_points += 10 if feature_values["NoOfSubDomain"] >= 2 else 0
    risk_points += 18 if feature_values["HasObfuscation"] else 0
    risk_points += 8 if feature_values["DegitRatioInURL"] > 0.20 else 0
    risk_points += 8 if feature_values["SpacialCharRatioInURL"] > 0.08 else 0
    risk_points += 4 if feature_values["NoOfQMarkInURL"] > 0 else 0
    risk_points += 4 if feature_values["NoOfEqualsInURL"] >= 2 else 0
    return min(risk_points, 99) / 100.0


def build_html(url, probability, feature_values, phishing_reasons, legitimate_signals):
    label = "PHISHING" if probability >= 0.5 else "LIKELY LEGITIMATE"
    color = "#b42318" if probability >= 0.5 else "#087f5b"
    reason_items = phishing_reasons if probability >= 0.5 else legitimate_signals
    reason_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in reason_items)
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(value))}</td></tr>"
        for name, value in feature_values.items()
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>URL Prediction Result</title>
<style>
body {{ margin: 0; background: #f4f6f5; color: #17212b; font-family: Georgia, serif; }}
main {{ width: min(820px, calc(100% - 28px)); margin: 0 auto; padding: 42px 0 64px; }}
.kicker {{ color: #087f8c; font: 700 12px Arial, sans-serif; letter-spacing: 1.5px; text-transform: uppercase; }}
h1 {{ font-size: clamp(2.2rem, 7vw, 4.4rem); line-height: 1; font-weight: 500; margin: 12px 0; }}
.url {{ overflow-wrap: anywhere; color: #65727e; }}
.result {{ background: white; border-top: 8px solid {color}; padding: 24px; margin: 26px 0; border: 1px solid #dce3e8; }}
.label {{ color: {color}; font: 700 18px Arial, sans-serif; letter-spacing: 1px; }}
.probability {{ font: 700 clamp(3rem, 10vw, 6.5rem)/1 Arial, sans-serif; margin: 12px 0; }}
.meter {{ height: 14px; background: #e6ecea; }}
.meter span {{ display: block; height: 100%; width: {probability * 100:.2f}%; background: {color}; }}
section {{ background: white; border: 1px solid #dce3e8; padding: 20px; margin-top: 18px; }}
h2 {{ margin-top: 0; font-size: 1.35rem; font-weight: 500; }}
li {{ margin: 9px 0; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; font: 13px Arial, sans-serif; }}
td {{ border-bottom: 1px solid #e4e9eb; padding: 8px; text-align: left; }}
.note {{ color: #65727e; font: 13px/1.5 Arial, sans-serif; }}
</style></head><body><main>
<div class="kicker">URL-only phishing detector</div><h1>Prediction result</h1><p class="url">{html.escape(url)}</p>
<div class="result"><div class="label">{label}</div><div class="probability">{probability * 100:.2f}%</div><div class="meter"><span></span></div><p>Estimated probability that this URL is phishing.</p></div>
<section><h2>Risk signals</h2><ul>{reason_html}</ul><p class="note">This is a transparent URL-structure screening score. It is not the trained ANN probability because the ANN requires the original dataset's engineered features.</p></section>
<section><h2>Extracted URL features</h2><table>{rows}</table></section>
<p class="note">This prediction uses URL structure only. It does not visit the website, inspect its HTML, or guarantee safety. Do not open a suspicious URL just to test it.</p>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description="Predict phishing probability for a URL.")
    parser.add_argument("url", nargs="?", help="URL to evaluate")
    parser.add_argument("--no-browser", action="store_true", help="Only print the result")
    args = parser.parse_args()
    raw_url = args.url or input("Enter a website URL: ")
    url, parsed, feature_values, _ = extract_features(raw_url)
    probability = screening_probability(feature_values)
    phishing_reasons, legitimate_signals = heuristic_reasons(parsed, feature_values)
    label = "PHISHING" if probability >= 0.5 else "LIKELY LEGITIMATE"

    print("=" * 72)
    print(f"URL: {url}")
    print(f"Result: {label}")
    print(f"Phishing probability: {probability * 100:.2f}%")
    print(f"Legitimate probability: {(1.0 - probability) * 100:.2f}%")
    print("\nExplanation:")
    for reason in (phishing_reasons if probability >= 0.5 else legitimate_signals):
        print(f"- {reason}")
    print("\nNote: this is a URL-structure screening score, not the trained ANN probability.")

    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(
        build_html(url, probability, feature_values, phishing_reasons, legitimate_signals),
        encoding="utf-8",
    )
    print(f"\nBrowser report: {OUTPUT_PATH}")
    if not args.no_browser:
        webbrowser.open(OUTPUT_PATH.as_uri())


if __name__ == "__main__":
    main()
