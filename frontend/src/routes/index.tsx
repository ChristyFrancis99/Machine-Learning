import "@/premium.css";
import { createFileRoute } from "@tanstack/react-router";
import { PhishLensDashboard } from "@/components/PhishLensDashboard";

export const Route = createFileRoute("/")({
  // The research dashboard is entirely client-side. Keeping it out of SSR
  // prevents a browser-only dashboard dependency from taking down the root
  // document before the client can render the application.
  ssr: false,
  head: () => ({
    meta: [
      { title: "PhishLens — Phishing Detection Research Dashboard" },
      {
        name: "description",
        content:
          "Explore URL risk signals and phishing detection research results in the PhishLens dashboard.",
      },
      { property: "og:title", content: "PhishLens — Phishing Detection Dashboard" },
      {
        property: "og:description",
        content: "URL screening demonstrations and phishing detection research results.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return <PhishLensDashboard />;
}
