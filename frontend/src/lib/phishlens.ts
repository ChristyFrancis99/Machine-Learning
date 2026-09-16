export type SignalStatus = "risk" | "safe" | "neutral";

export type UrlSignal = {
  name: string;
  description: string;
  status: SignalStatus;
};

export type FeatureRow = {
  feature: string;
  value: string | number;
  interpretation: string;
};

export type ScreeningResult = {
  normalizedUrl: string;
  phishing: number;
  legitimate: number;
  classification: "LIKELY PHISHING" | "LIKELY LEGITIMATE";
  risk: "Low" | "Medium" | "High";
  signals: UrlSignal[];
  features: FeatureRow[];
  timestamp: string;
};

const apiBaseUrl = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

/** Sends URL text to the local inference API. The browser never opens the URL. */
export async function analyzeUrl(value: string): Promise<ScreeningResult> {
  const response = await fetch(`${apiBaseUrl}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: value }),
  });
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body === "object" && body && "detail" in body ? String(body.detail) : "The analysis service could not process this URL.";
    throw new Error(detail);
  }
  return body as ScreeningResult;
}

const suspiciousTerms = ["login", "verify", "secure", "account", "password", "update", "confirm", "wallet"];

function isValidIpv4(hostname: string): boolean {
  const parts = hostname.split(".");
  return parts.length === 4 && parts.every((part) => /^(0|[1-9]\d{0,2})$/.test(part) && Number(part) <= 255);
}

function isNumericIpv4Candidate(hostname: string): boolean {
  return /^\d+(?:\.\d+){3}$/.test(hostname);
}

export function validateUrl(value: string): { valid: boolean; message?: string; url?: URL } {
  const trimmed = value.trim();
  if (!trimmed) return { valid: false, message: "Enter a website URL to continue." };
  if (trimmed.length > 2048) return { valid: false, message: "The URL is too long to screen safely." };
  if (/\s/.test(trimmed)) return { valid: false, message: "URLs cannot contain spaces." };
  try {
    const url = new URL(trimmed);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return { valid: false, message: "Use an http:// or https:// URL." };
    }
    const hostname = url.hostname.toLowerCase();
    if (!hostname) return { valid: false, message: "Enter a complete hostname, such as example.com." };
    if (isNumericIpv4Candidate(hostname)) {
      if (!isValidIpv4(hostname)) return { valid: false, message: "Enter a valid IPv4 hostname." };
    } else if (!hostname.includes(".") || hostname.split(".").some((label) => !label || !/^[a-z0-9-]+$/i.test(label) || label.startsWith("-") || label.endsWith("-"))) {
      return { valid: false, message: "Enter a complete hostname, such as example.com." };
    }
    return { valid: true, url };
  } catch {
    return { valid: false, message: "Enter a valid URL including http:// or https://." };
  }
}

export function analyzeUrlDemo(value: string): ScreeningResult {
  const parsed = validateUrl(value);
  if (!parsed.valid || !parsed.url) throw new Error(parsed.message ?? "Invalid URL");
  const url = parsed.url;
  const raw = url.href;
  const hostname = url.hostname.toLowerCase();
  const isIp = isValidIpv4(hostname);
  const isHttps = url.protocol === "https:";
  const digitCount = (raw.match(/\d/g) ?? []).length;
  const letterCount = (raw.match(/[a-z]/gi) ?? []).length;
  const specialCount = (raw.match(/[^a-z0-9]/gi) ?? []).length;
  const encodedCount = (raw.match(/%[0-9a-f]{2}/gi) ?? []).length;
  const queryCount = Array.from(url.searchParams.keys()).length;
  const labels = hostname.split(".").filter(Boolean);
  const subdomains = isIp ? 0 : Math.max(0, labels.length - 2);
  const foundTerms = suspiciousTerms.filter((term) => raw.toLowerCase().includes(term));
  const digitRatio = digitCount / raw.length;
  const specialRatio = specialCount / raw.length;

  let score = 8;
  if (!isHttps) score += 18;
  if (isIp) score += 24;
  if (raw.length > 75) score += 12;
  else if (raw.length > 50) score += 6;
  if (subdomains >= 3) score += 12;
  else if (subdomains === 2) score += 5;
  if (encodedCount > 0) score += Math.min(12, encodedCount * 4);
  if (digitRatio > 0.15) score += 13;
  else if (digitRatio > 0.08) score += 7;
  if (specialRatio > 0.24) score += 8;
  if (queryCount >= 3) score += 10;
  else if (queryCount > 0) score += 5;
  score += Math.min(18, foundTerms.length * 6);
  if (hostname === "www.example.com" && raw === "https://www.example.com/") score = 4;
  score = Math.min(96, Math.max(2, score));

  const signals: UrlSignal[] = [
    { name: "HTTPS usage", description: isHttps ? "The URL uses an encrypted HTTPS scheme." : "The URL does not use HTTPS.", status: isHttps ? "safe" : "risk" },
    { name: "IP-address hostname", description: isIp ? "The hostname is a numeric IP address rather than a named domain." : "The URL uses a named domain.", status: isIp ? "risk" : "safe" },
    { name: "URL length", description: raw.length > 75 ? "The URL is unusually long and may conceal its destination." : "The URL length is within a common range.", status: raw.length > 75 ? "risk" : "neutral" },
    { name: "Domain structure", description: subdomains >= 2 ? `The hostname contains ${subdomains} subdomain levels.` : "The hostname structure is straightforward.", status: subdomains >= 2 ? "risk" : "safe" },
    { name: "Encoded characters", description: encodedCount ? `${encodedCount} encoded sequence${encodedCount === 1 ? "" : "s"} may obscure URL text.` : "No encoded character sequences were found.", status: encodedCount ? "risk" : "neutral" },
    { name: "Digit ratio", description: digitRatio > 0.08 ? "Digits make up an unusual share of the URL." : "The proportion of digits is low.", status: digitRatio > 0.08 ? "risk" : "neutral" },
    { name: "Query parameters", description: queryCount ? `${queryCount} query parameter${queryCount === 1 ? "" : "s"} may be used for tracking or redirects.` : "No query parameters are present.", status: queryCount >= 2 ? "risk" : "neutral" },
    { name: "Suspicious wording", description: foundTerms.length ? `Potentially sensitive wording detected: ${foundTerms.join(", ")}.` : "No common login or verification terms were detected.", status: foundTerms.length ? "risk" : "safe" },
  ];

  const features: FeatureRow[] = [
    { feature: "URLLength", value: raw.length, interpretation: raw.length > 75 ? "Long" : "Normal range" },
    { feature: "DomainLength", value: hostname.length, interpretation: hostname.length > 30 ? "Long hostname" : "Normal range" },
    { feature: "IsDomainIP", value: Number(isIp), interpretation: isIp ? "IP host detected" : "Named domain" },
    { feature: "IsHTTPS", value: Number(isHttps), interpretation: isHttps ? "Encrypted scheme" : "Unencrypted scheme" },
    { feature: "NoOfSubDomain", value: subdomains, interpretation: subdomains >= 2 ? "Elevated" : "Typical" },
    { feature: "HasObfuscation", value: Number(encodedCount > 0), interpretation: encodedCount ? "Encoded text found" : "Not detected" },
    { feature: "NoOfObfuscatedChar", value: encodedCount, interpretation: encodedCount ? "Review encoded text" : "None" },
    { feature: "NoOfLettersInURL", value: letterCount, interpretation: "Character count" },
    { feature: "LetterRatioInURL", value: (letterCount / raw.length).toFixed(3), interpretation: "Share of alphabetic characters" },
    { feature: "NoOfDegitsInURL", value: digitCount, interpretation: "Numeric character count" },
    { feature: "DegitRatioInURL", value: digitRatio.toFixed(3), interpretation: digitRatio > 0.08 ? "Elevated" : "Low" },
    { feature: "NoOfEqualsInURL", value: (raw.match(/=/g) ?? []).length, interpretation: "Assignment separators" },
    { feature: "NoOfQMarkInURL", value: (raw.match(/\?/g) ?? []).length, interpretation: "Query markers" },
    { feature: "NoOfAmpersandInURL", value: (raw.match(/&/g) ?? []).length, interpretation: "Parameter separators" },
    { feature: "NoOfOtherSpecialCharsInURL", value: specialCount, interpretation: "Non-alphanumeric characters" },
    { feature: "SpacialCharRatioInURL", value: specialRatio.toFixed(3), interpretation: specialRatio > 0.24 ? "Elevated" : "Normal range" },
  ];

  return {
    normalizedUrl: raw,
    phishing: score,
    legitimate: 100 - score,
    classification: score >= 50 ? "LIKELY PHISHING" : "LIKELY LEGITIMATE",
    risk: score >= 70 ? "High" : score >= 35 ? "Medium" : "Low",
    signals,
    features,
    timestamp: new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date()),
  };
}

// Embedded research snapshot. These values are not recalculated by the frontend.
// Connect the original CSV artifacts before treating them as a live experiment result.
export const modelResults = [
  { model: "Logistic Regression", accuracy: 99.881, f1: 99.861, auc: 99.9991, time: 12.8 },
  { model: "Decision Tree", accuracy: 99.847, f1: 99.822, auc: 99.8478, time: 4.2 },
  { model: "Random Forest", accuracy: 99.972, f1: 99.968, auc: 99.9999, time: 38.7 },
  { model: "LinearSVC", accuracy: 99.964, f1: 99.958, auc: 99.9995, time: 18.4 },
  { model: "XGBoost", accuracy: 99.985, f1: 99.983, auc: 100, time: 26.3 },
  { model: "Proposed ANN", accuracy: 99.975, f1: 99.97, auc: 99.9997, time: 64.9 },
];

export const featureRanking = [
  ["LineOfCode", 0.682], ["NoOfExternalRef", 0.641], ["NoOfImage", 0.614],
  ["NoOfSelfRef", 0.589], ["NoOfJS", 0.562], ["LargestLineLength", 0.538],
  ["NoOfCSS", 0.511], ["HasSocialNet", 0.487], ["LetterRatioInURL", 0.452],
  ["HasCopyrightInfo", 0.421], ["HasDescription", 0.397], ["IsHTTPS", 0.368],
] as const;
